#!/usr/bin/env python3
"""Build a foam-format (sphere-only) collision URDF for the G1 from the CURATED proxy model.

Why not --collision-res: auto-spherizing the G1's collision MESHES produces a model that calls the
robot's own nominal all-zeros pose self-colliding (measured: 0/9 random configs free, including
q=0). Adjacent links overlap at any usable resolution. HJCD's Panda path avoids exactly this by
consuming a hand-curated pre-spherized URDF instead, and the G1 already has an equivalent curated
description -- collision_sidecar/g1_proxy_model.yaml, the geometry the g1sc self-collision sidecar
uses and which does report q=0 free.

This converts that proxy into the foam interchange format the codegen's --spherized-urdf expects:

    capsule(p0, p1, r)  ->  a chain of radius-r spheres along the segment at spacing SPACING*r
    sphere(c, r)        ->  itself

A sphere chain at spacing s has a "waist" of sqrt(r^2 - (s/2)^2) between consecutive centres, so
SPACING is the coverage/count trade-off: 0.6 gives a waist of 0.977r, i.e. under 3% pinch, which is
well inside the proxy's own fidelity.

Usage:  python scripts/dev/make_g1_spherized.py [out.urdf]
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
SRC_URDF = REPO / "csrc" / "urdf" / "g1_29dof_rev_1_0.urdf"
PROXY = REPO / "collision_sidecar" / "g1_proxy_model.yaml"
DEFAULT_OUT = REPO / "csrc" / "urdf" / "g1_proxy_spherized.urdf"

SPACING = 0.6          # sphere centre spacing, in units of the capsule radius
MIN_SPHERES = 2        # a capsule always gets at least its two end caps


def capsule_to_spheres(p0, p1, r):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.linalg.norm(p1 - p0))
    if L < 1e-9:
        return [(p0, r)]
    n = max(MIN_SPHERES, int(np.ceil(L / (SPACING * r))) + 1)
    return [(p0 + (p1 - p0) * (i / (n - 1)), r) for i in range(n)]


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    prims = yaml.safe_load(PROXY.read_text())["primitives"]

    per_link = {}
    for p in prims:
        link = p["link"]
        if p["type"] == "sphere":
            c = p.get("center") or p.get("origin") or p.get("p0")
            per_link.setdefault(link, []).append((np.asarray(c, float), float(p["radius"])))
        elif p["type"] == "capsule":
            per_link.setdefault(link, []).extend(
                capsule_to_spheres(p["p0"], p["p1"], float(p["radius"])))
        else:
            raise SystemExit(f"unhandled proxy primitive type: {p['type']}")

    tree = ET.parse(SRC_URDF)
    root = tree.getroot()

    # Replace ALL collision geometry: foam format is sphere-only, and leaving a stray mesh behind
    # would send the reader down the mesh path for that link.
    for link in root.iter("link"):
        for col in list(link.findall("collision")):
            link.remove(col)
        for c, r in per_link.get(link.get("name"), []):
            col = ET.SubElement(link, "collision")
            ET.SubElement(col, "origin", {"xyz": f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}",
                                          "rpy": "0 0 0"})
            geom = ET.SubElement(col, "geometry")
            ET.SubElement(geom, "sphere", {"radius": f"{r:.6f}"})

    out_path.write_bytes(ET.tostring(root))
    total = sum(len(v) for v in per_link.values())
    print(f"wrote {out_path}")
    print(f"  links with spheres: {len(per_link)}   total spheres: {total}")
    print()
    print(f"  {'link':<32s} {'spheres':>7s}  {'radii (min..max)':>20s}")
    for link in sorted(per_link):
        rs = [r for _, r in per_link[link]]
        print(f"  {link:<32s} {len(rs):>7d}  {min(rs):>9.4f}..{max(rs):<9.4f}")


if __name__ == "__main__":
    main()
