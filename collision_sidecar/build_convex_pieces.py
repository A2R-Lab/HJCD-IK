"""Extract canonical MuJoCo collision geometry as convex pieces for the GJK narrow phase
(Checkpoint 1D, spec section 2).

For the leg-region links that need an exact narrow phase, pull the COMPILED collision geoms
(not visual meshes) from the MuJoCo model and represent each link as a small union of convex
pieces in the LINK frame:

    sphere   (center, radius)
    capsule  (p0, p1, radius)
    box      (center, half-extents, R)
    hull     (convex-hull vertices -- the same convex hull MuJoCo uses for a mesh geom)

Placed via MuJoCo's own FK at neutral (rigid attachment is config-free), then expressed in the
owning link frame. Writes generated/g1_convex_pieces.json with per-link pieces, transforms,
source geom descriptors and hashes.

    MUJOCO_GL=egl python3 collision_sidecar/build_convex_pieces.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HJCD = os.path.dirname(HERE)
PARENT = os.path.dirname(os.path.dirname(HJCD))
sys.path.insert(0, os.path.join(PARENT, "src"))
sys.path.insert(0, os.path.join(PARENT, "production"))

OUT = os.path.join(HJCD, "generated", "g1_convex_pieces.json")

# Links that participate in GJK pairs (the exact unresolved leg-region set).
GJK_LINKS = [
    "base_link",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link", "left_wrist_yaw_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link", "right_wrist_yaw_link",
    # Checkpoint 3C.1: exact convex/GJK for the shoulder_yaw<->torso pairs the torso-SDF proxy missed.
    # torso is a MULTI-piece convex link here; it remains a cluster SDF for all OTHER limb pairs.
    "torso_link", "left_shoulder_yaw_link", "right_shoulder_yaw_link",
    # Checkpoint 3C.2: knee links, for the deep cross-body leg-leg pairs the thin capsules missed
    # (hip_yaw links are already GJK links via the wrist<->thigh routing).
    "left_knee_link", "right_knee_link",
    # Checkpoint 3C.3: elbow links. Same failure mechanism as 3C.1's shoulder_yaw finding -- an arm
    # link folded across the chest sinks into the torso SDF proxy without the proxy noticing.
    # Mined from hard-mode outputs: 21 deep false negatives on right_elbow<->torso, up to 12.81 mm,
    # and mirroring those configurations reproduces 18 more on the left, so both sides are routed.
    "left_elbow_link", "right_elbow_link",
    # Native-completion checkpoint (Stage A): the broad production+diversified audit
    # (collision_sidecar/mine_all_selfcoll_fn.py -> g1_selfcoll_fn_corpus.json) found 25 deep
    # self-collision false-negative pair classes -- native calls them free while MuJoCo records up
    # to 19.6 mm of penetration -- and these HULL-representable link families had no exact convex
    # geometry (only compact primitives) that could be pierced. Every deep FN is a genuine model
    # gap (GPU == CPU). They are given exact convex hulls and their pairs routed to FP64 GJK below.
    #
    "left_wrist_pitch_link", "right_wrist_pitch_link",
    "left_wrist_roll_link", "right_wrist_roll_link",
    # Native-completion checkpoint Task A: the foot. Its MuJoCo collision geometry is 4 SPHERES, not
    # a hull. The exact GJK path now supports TYPED pieces (sphere + hull), so the foot routes to
    # exact GJK as a union of 4 spheres -- closing the 5 residual foot deep-FN pairs the previous
    # checkpoint documented. No oversized sphere, no broad hull, no tuned capsule, no margin.
    "left_ankle_roll_link", "right_ankle_roll_link",
]


def _geom_pieces(m, d, link):
    import mujoco
    from scipy.spatial import ConvexHull
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, link)
    Rl = d.xmat[bid].reshape(3, 3); pl = d.xpos[bid]
    pieces, descs = [], []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        gt = m.geom_type[g]
        # geom world pose -> link frame
        gp = Rl.T @ (d.geom_xpos[g] - pl)
        gR = Rl.T @ d.geom_xmat[g].reshape(3, 3)
        size = m.geom_size[g]
        if gt == mujoco.mjtGeom.mjGEOM_SPHERE:
            pieces.append(dict(type="sphere", center=gp.tolist(), radius=float(size[0])))
        elif gt == mujoco.mjtGeom.mjGEOM_CAPSULE:
            a = gR @ np.array([0, 0, float(size[1])])
            pieces.append(dict(type="capsule", p0=(gp - a).tolist(), p1=(gp + a).tolist(),
                               radius=float(size[0])))
        elif gt == mujoco.mjtGeom.mjGEOM_CYLINDER:
            a = gR @ np.array([0, 0, float(size[1])])   # capsule-approx (conservative outer)
            pieces.append(dict(type="capsule", p0=(gp - a).tolist(), p1=(gp + a).tolist(),
                               radius=float(size[0])))
        elif gt == mujoco.mjtGeom.mjGEOM_BOX:
            pieces.append(dict(type="box", center=gp.tolist(), half=size[:3].tolist(),
                               R=gR.tolist()))
        elif gt == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            verts = np.array(m.mesh_vert[va:va + vn]).reshape(-1, 3)
            local = (gR @ verts.T).T + gp                # mesh -> link frame
            hull = local[ConvexHull(local).vertices]     # MuJoCo collides the convex hull
            pieces.append(dict(type="hull", verts=hull.tolist()))
        else:
            continue
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        descs.append(f"{nm or 'geom'+str(g)}:{gt}")
    return pieces, descs


def build():
    import mujoco
    import route_qs_lsq as RQ
    m = RQ.load_inputs().model
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    links = {}
    geom_src = {}
    for link in GJK_LINKS:
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, link) < 0:
            continue
        pieces, descs = _geom_pieces(m, d, link)
        # bounding sphere (link frame) enclosing all pieces -- conservative broad phase
        pts = []
        for p in pieces:
            if p["type"] == "sphere":
                pts.append((p["center"], p["radius"]))
            elif p["type"] == "capsule":
                pts.append((p["p0"], p["radius"])); pts.append((p["p1"], p["radius"]))
            elif p["type"] == "box":
                c = np.array(p["center"]); pts.append((c.tolist(), float(np.linalg.norm(p["half"]))))
            else:
                for v in p["verts"]:
                    pts.append((v, 0.0))
        C = np.mean([np.array(c) for c, _r in pts], axis=0)
        rad = max(float(np.linalg.norm(np.array(c) - C) + r) for c, r in pts)
        links[link] = dict(pieces=pieces, bound_center=C.tolist(), bound_radius=rad)
        geom_src[link] = descs

    urdf = os.path.join(HJCD, "csrc", "urdf", "g1_29dof_rev_1_0.urdf")
    urdf_hash = hashlib.sha1(open(urdf, "rb").read()).hexdigest()[:16]
    blob = json.dumps(links, sort_keys=True).encode()
    geom_hash = hashlib.sha1(blob).hexdigest()[:16]
    json.dump({"links": links, "geom_src": geom_src, "urdf_hash": urdf_hash,
               "geom_hash": geom_hash}, open(OUT, "w"))
    npieces = sum(len(v["pieces"]) for v in links.values())
    nverts = sum(len(p["verts"]) for v in links.values() for p in v["pieces"] if p["type"] == "hull")
    print(f"convex pieces: {len(links)} links, {npieces} pieces, {nverts} hull verts")
    for link, v in links.items():
        types = [p["type"] for p in v["pieces"]]
        print(f"  {link:<24} {types}  bound_r={v['bound_radius']*1000:.0f}mm")
    print(f"geom_hash={geom_hash} -> {os.path.relpath(OUT, HJCD)}")


if __name__ == "__main__":
    build()
