"""foam spherized-URDF -> HJCD collision sphere model (the `<name>_spheres_array` + `<name>_sphere_to_joint`
data in csrc/robots/<name>.cuh).

foam (https://github.com/CoMMALab/foam, submodule external/foam) converts a URDF's collision geometry into
per-link spheres and writes a *spherized URDF*: each link carries
    <collision><geometry><sphere radius="R"/></geometry><origin xyz="x y z" rpy="..."/></collision>
(the sphere center is in the link frame; rpy is irrelevant for a sphere). This module parses that and emits
the two constant arrays HJCD's collision path uses, matching the hand-authored csrc/robots/panda.cuh format.

SCOPE NOTE (important): HJCD's collision path is a self-contained pRRTC port keyed by a `ppln::robots::<Robot>`
type that bundles MORE than spheres — it also needs the per-joint fixed_transforms (URDF joint origins), the
joint axes/types, dimension, and self-collision ranges (see csrc/robots/panda.cuh: `panda_fixed_transforms`,
`PANDA_SELF_CC_RANGE_COUNT`, and `ppln::collision::fk<Panda>`). foam supplies ONLY the spheres. Generating a
complete new-robot definition (fixed_transforms + axes + self-CC) from a URDF is the remaining, larger piece
and is tracked in docs/open-tasks/multi_robot_byo_urdf_plan_2026-07-07.md. This module is the sphere part.

CPU-only (stdlib XML). Tests: scripts/codegen/test_foam_spheres.py (run `pytest scripts/codegen/test_foam_spheres.py`).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import OrderedDict

ACTUATED = {"revolute", "prismatic", "continuous"}


def parse_spherized_urdf(path: str) -> "OrderedDict[str, list[tuple[float, float, float, float]]]":
    """Return an ordered {link_name: [(x, y, z, radius), ...]} for every link with collision spheres.
    Link order follows the URDF document order (stable)."""
    root = ET.parse(path).getroot()
    out: "OrderedDict[str, list]" = OrderedDict()
    for link in root.findall("link"):
        spheres = []
        for col in link.findall("collision"):
            sph = col.find("geometry/sphere")
            if sph is None:
                continue
            r = float(sph.get("radius"))
            origin = col.find("origin")
            xyz = (origin.get("xyz") if origin is not None else None) or "0 0 0"
            x, y, z = (float(v) for v in xyz.split())
            spheres.append((x, y, z, r))
        if spheres:
            out[link.get("name")] = spheres
    return out


def link_to_joint_index(urdf_path: str) -> "dict[str, int]":
    """Map each link -> the number of ACTUATED joints on the root->link path.

    This is HJCD/pRRTC's `sphere_to_joint` convention: the base/root link is 0, and the index increments by
    one across each actuated (revolute/prismatic/continuous) joint down the chain — so a sphere on the k-th
    moving link gets index k (fixed joints don't advance it). Matches panda.cuh (base=0; the kernel skips
    index 0 as the pedestal-mounted base). Works for a tree; each link's index is well-defined by its path.
    """
    root = ET.parse(urdf_path).getroot()
    child_joints: "dict[str, list[tuple[str, str]]]" = {}   # parent_link -> [(child_link, joint_type)]
    all_children = set()
    for j in root.findall("joint"):
        p = j.find("parent").get("link")
        c = j.find("child").get("link")
        child_joints.setdefault(p, []).append((c, j.get("type")))
        all_children.add(c)
    roots = [l.get("name") for l in root.findall("link") if l.get("name") not in all_children]
    idx: "dict[str, int]" = {}
    # BFS from the root link(s), carrying the cumulative actuated-joint count.
    stack = [(r, 0) for r in roots]
    while stack:
        link, count = stack.pop()
        idx[link] = count
        for child, jtype in child_joints.get(link, []):
            stack.append((child, count + (1 if jtype in ACTUATED else 0)))
    return idx


def build_sphere_model(spherized_urdf: str, kinematic_urdf: str | None = None):
    """Flatten (spheres, sphere_to_joint) in link order. `kinematic_urdf` supplies the joint tree for the
    index mapping (defaults to the spherized URDF, which carries the same joints). Returns
    (spheres: list[(x,y,z,r)], sphere_to_joint: list[int], links: list[str])."""
    by_link = parse_spherized_urdf(spherized_urdf)
    l2j = link_to_joint_index(kinematic_urdf or spherized_urdf)
    spheres, s2j, links = [], [], []
    for link, sph in by_link.items():
        if link not in l2j:
            raise KeyError(f"link {link!r} has spheres but is not in the joint tree of "
                           f"{kinematic_urdf or spherized_urdf!r}")
        for s in sph:
            spheres.append(s)
            s2j.append(l2j[link])
            links.append(link)
    return spheres, s2j, links


def emit_cuh_fragment(name: str, spheres, sphere_to_joint) -> str:
    """Emit the `<name>_spheres_array[N]` (float4 x,y,z,r) + `<name>_sphere_to_joint[]` constants, matching
    the csrc/robots/panda.cuh format. Caller wraps in the `ppln::collision` namespace + the rest of the
    robot definition (see SCOPE NOTE)."""
    n = len(spheres)
    lines = [f"    // GENERATED from a foam spherized URDF by scripts/codegen/foam_spheres.py",
             f"    #define {name.upper()}_SPHERE_COUNT {n}",
             f"    __device__ __constant__ float4 {name}_spheres_array[{n}] = {{"]
    for (x, y, z, r) in spheres:
        lines.append(f"        {{ {x:.6g}f, {y:.6g}f, {z:.6g}f, {r:.6g}f }},")
    lines.append("    };")
    lines.append(f"    __device__ __constant__ int {name}_sphere_to_joint[] = {{")
    lines.append("        " + ", ".join(str(j) for j in sphere_to_joint))
    lines.append("    };")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    sph_urdf = sys.argv[1]
    kin_urdf = sys.argv[2] if len(sys.argv) > 2 else None
    name = sys.argv[3] if len(sys.argv) > 3 else "robot"
    spheres, s2j, _ = build_sphere_model(sph_urdf, kin_urdf)
    print(emit_cuh_fragment(name, spheres, s2j))
