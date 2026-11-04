"""Environment-collision inputs for the experimental tree (G1) solver path (humanoid-v1 M6a).

Two runtime inputs, deliberately separate from the kinematic artifact and from each other:

* ROBOT SPHERE TABLE ``[NS, 5] = (anchor, ox, oy, oz, r)``: a GRiD foam-format sphere model
  (``python -m grid_codegen.spherize``; one ``<sphere>`` per ``<collision>``, centre in its LINK frame)
  re-expressed on the tree model's frames. ``anchor`` is the movable joint index (GRiD order) whose
  child-link frame carries the sphere, or ``-1`` for the base (root link) frame; the offset is the
  sphere centre in that frame (fixed-joint chains folded in). On the device the centre is
  ``p_W = R_B (X_anchor o) + p_B``, with ``X_anchor`` the same tree FK the LM uses.

  GRiD's own ``sphere_anchor_frames`` maps ROOT-welded spheres to a skip sentinel ("pedestal"). That
  is right for a bolted arm and wrong for a floating base (the pelvis can hit the wall), so root
  spheres are kept here with anchor -1. Non-root anchors are cross-checked against GRiD's mapping in
  the tests.

* ENVIRONMENT ``dict(spheres=(M,4), capsules=(M,7), cuboids=(M,10), planes=(M,4))`` in the WORLD frame,
  GRiD ``make_host_environment`` row layouts (cuboid = centre, unit quaternion wxyz, FULL dims; plane
  = n.p >= d). Uploaded per call into an owned ``grid_collision::DeviceEnvironment<T>``.

Tool overrides only move IK contact targets; they never touch this table.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ENV_KINDS = (("spheres", 4), ("capsules", 7), ("cuboids", 10), ("planes", 4))


def _origin(joint):
    o = joint.find("origin")
    xyz = [float(v) for v in ((o.get("xyz") if o is not None and o.get("xyz") else "0 0 0").split())]
    rpy = [float(v) for v in ((o.get("rpy") if o is not None and o.get("rpy") else "0 0 0").split())]
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    R = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                  [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                  [-sp, cp * sr, cp * cr]])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = xyz
    return T


def parse_sphere_model(spherized_urdf):
    """``{link: [(x, y, z, r), ...]}`` in document order (GRiD's foam parser, no re-implementation)."""
    from grid_codegen.algorithms._collision import parse_spherized_urdf
    return parse_spherized_urdf(str(spherized_urdf))


def link_anchors(urdf, joint_names):
    """``{link: (anchor, T_anchor_link)}`` for every link of the kinematic URDF: the nearest ancestor frame
    that is a movable joint's child (anchor = its index in ``joint_names``) or the root (anchor = -1)."""
    root = ET.parse(str(urdf)).getroot()
    parent_joint = {j.find("child").get("link"): j for j in root.findall("joint")}
    index = {n: i for i, n in enumerate(joint_names)}
    out = {}
    for link in root.findall("link"):
        name = link.get("name")
        T = np.eye(4)
        cur = name
        while True:
            j = parent_joint.get(cur)
            if j is None:
                out[name] = (-1, T)
                break
            if j.get("type") != "fixed":
                if j.get("name") not in index:
                    raise ValueError(f"movable joint {j.get('name')!r} is not in the model's joint order")
                out[name] = (index[j.get("name")], T)
                break
            T = _origin(j) @ T
            cur = j.find("parent").get("link")
    return out


def robot_sphere_table(spherized_urdf, urdf, joint_names, dtype=np.float64):
    """Return ``(table [NS,5], links [NS])`` for the tree solver's environment-collision channel."""
    model = parse_sphere_model(spherized_urdf)
    anchors = link_anchors(urdf, joint_names)
    rows, links = [], []
    for link, spheres in model.items():
        if link not in anchors:
            raise ValueError(f"sphere model link {link!r} is not a link of {Path(urdf).name}")
        a, T = anchors[link]
        for (x, y, z, r) in spheres:
            if not (r > 0 and np.isfinite([x, y, z, r]).all()):
                raise ValueError(f"bad sphere on {link!r}: {(x, y, z, r)}")
            c = T @ np.array([x, y, z, 1.0])
            rows.append((float(a), c[0], c[1], c[2], r)); links.append(link)
    if not rows:
        raise ValueError("the sphere model has no spheres")
    return np.asarray(rows, dtype=dtype), links


def normalize_environment(env, dtype):
    """Validate the shapes of an environment dict and return the four C-contiguous arrays (missing = empty).
    Row-level geometry checks (finite, positive radii/dims, non-zero quaternion/normal) are GRiD's
    ``make_host_environment``, done natively."""
    if env is None:
        env = {}
    unknown = set(env) - {k for k, _ in ENV_KINDS}
    if unknown:
        raise ValueError(f"unknown environment keys {sorted(unknown)}; expected spheres/capsules/cuboids/planes")
    out = {}
    for k, w in ENV_KINDS:
        a = np.ascontiguousarray(np.asarray(env.get(k, np.zeros((0, w))), dtype=dtype).reshape(-1, w) if np.size(env.get(k, [])) else np.zeros((0, w), dtype))
        out[k] = a
    return out


# ---------------------------------------------------------------- self-collision pair policy (M6b)
def anchor_parents(urdf, joint_names):
    """``{anchor: parent_anchor}`` over rigid bodies: anchor = movable-joint index (its child link plus every link
    welded to it) or -1 (the root assembly). The parent body is the anchor of the joint's parent link."""
    root = ET.parse(str(urdf)).getroot()
    anchors = link_anchors(urdf, joint_names)
    out = {}
    for j in root.findall("joint"):
        if j.get("type") != "fixed" and j.get("name") in joint_names:
            out[joint_names.index(j.get("name"))] = anchors[j.find("parent").get("link")][0]
    return out


def self_collision_pairs(urdf, joint_names, table):
    """Checked body pairs ``[NP, 2]`` (anchors, a < b) and an audit list.

    Policy = topology only (GRiD ``build_self_cc_ranges`` adjacency == MuJoCo same-weld + ``filterparent``):
      * spheres on the same rigid body (same anchor; fixed-joint links merged) are never paired;
      * directly connected bodies (parent/child in the movable-joint tree) are excluded: they overlap at the joint
        by construction;
      * every other pair of bodies that carries spheres is CHECKED, whatever it does in a nominal pose.
    No model-specific (e.g. MJCF ``<exclude>``) or pose-derived exclusions."""
    names = list(joint_names)
    par = anchor_parents(urdf, names)
    anchors = sorted({int(a) for a in np.asarray(table)[:, 0]})
    counts = {a: int((np.asarray(table)[:, 0] == a).sum()) for a in anchors}
    pairs, audit = [], []
    for i, a in enumerate(anchors):
        for b in anchors[i + 1:]:
            adjacent = par.get(a, None) == b or par.get(b, None) == a
            audit.append(dict(a=a, b=b, spheres_a=counts[a], spheres_b=counts[b], checked=not adjacent,
                              reason="parent/child bodies (joint overlap)" if adjacent else "non-adjacent bodies"))
            if not adjacent:
                pairs.append((a, b))
    return np.asarray(pairs, dtype=np.int32).reshape(-1, 2), audit
