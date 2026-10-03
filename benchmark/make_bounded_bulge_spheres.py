#!/usr/bin/env python3
"""Conservative Panda sphere models with an explicit bulge budget, fitted to the TRUE link geometry.

A sphere model trades two errors: *bulge* (sphere union sticking out past the link, which eats clearance in
tight scenes) and *uncovered surface* (link sticking out past the spheres, which lets a "collision-free"
configuration touch an obstacle). foam's model bulges up to ~6 cm on link 5; cuRobo's leaves parts of the
true link uncovered. This tool fits a model with BOTH bounded, from the `panda_description` *visual*
meshes (the true shape, not the convex collision hulls):

  1. voxelise each link's visual mesh (pitch p) and fill it; Euclidean distance transforms give, per
     occupied voxel, the inscribed radius d_in (distance to the surface) and, per free voxel, d_out;
  2. candidate spheres sit at occupied voxels with radius d_in + b, so no sphere protrudes more than the
     bulge budget b (plus the voxel pitch) anywhere;
  3. greedy set cover of the surface voxels (occupied voxels with a free 6-neighbour) picks spheres that
     cover the most still-uncovered surface, until everything is covered (or --max-per-link is hit).

Output: a foam-format spherized URDF (the kinematic URDF's links/joints with sphere <collision>s, centres
in link frames; finger meshes are in the finger frames so the fixed 0.04 opening of csrc/urdf/panda.urdf
applies), consumable by `generate_grid.py --spherized-urdf`. `--report` evaluates any sphere URDF against
the visual meshes instead (count, max/p99 bulge, uncovered surface fraction).

  python benchmark/make_bounded_bulge_spheres.py --bulge-mm 10 --out benchmark/reference/panda_visual_b10_spherized.urdf
  python benchmark/make_bounded_bulge_spheres.py --report external/foam/assets/panda/smaller_panda_spherized.urdf
"""
from __future__ import annotations

import argparse
import heapq
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panda_model import _KINEMATIC_URDF  # noqa: E402

LINKS = [f"panda_link{i}" for i in range(1, 8)] + ["panda_hand", "panda_leftfinger", "panda_rightfinger"]


def _rpy_R(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]); Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def load_visual_meshes():
    """{link: trimesh in the LINK frame} from robot_descriptions' panda_description visual meshes."""
    import trimesh
    import yourdfpy
    from robot_descriptions import panda_description
    urdf = yourdfpy.URDF.load(panda_description.URDF_PATH, build_scene_graph=False, load_meshes=False,
                              build_collision_scene_graph=False, load_collision_meshes=False)
    out = {}
    for link in urdf.robot.links:
        if link.name not in LINKS:
            continue
        parts = []
        for vis in link.visuals:
            g = vis.geometry
            if g.mesh is not None:
                m = trimesh.load(urdf._filename_handler(g.mesh.filename), force="mesh")
                if g.mesh.scale is not None:
                    m.apply_scale(g.mesh.scale)
            elif g.box is not None:
                m = trimesh.creation.box(extents=g.box.size)
            else:
                continue
            T = np.eye(4)
            if vis.origin is not None:
                T = np.asarray(vis.origin)
            m.apply_transform(T)
            parts.append(m)
        if parts:
            out[link.name] = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
    return out


def voxel_fields(mesh, pitch):
    """Occupancy grid (filled), its origin, and the inside/outside Euclidean distance fields (metres)."""
    from scipy import ndimage
    vg = mesh.voxelized(pitch=pitch).fill()
    occ = np.asarray(vg.matrix, dtype=bool)
    occ = np.pad(occ, 2)                                             # a free margin for the outside EDT
    origin = np.asarray(vg.transform)[:3, 3] - 2 * pitch             # world of voxel index (0,0,0)
    d_in = ndimage.distance_transform_edt(occ) * pitch               # distance to nearest free voxel
    d_out = ndimage.distance_transform_edt(~occ) * pitch             # distance to nearest occupied voxel
    return occ, origin, d_in, d_out


def surface_voxels(occ):
    from scipy import ndimage
    eroded = ndimage.binary_erosion(occ, structure=ndimage.generate_binary_structure(3, 1))
    return occ & ~eroded


def fit_link(mesh, pitch, bulge, max_spheres, min_radius, stride=2):
    """Greedy bounded-bulge cover of one link. Returns [(x, y, z, r)] in the mesh frame."""
    from scipy.spatial import cKDTree
    occ, origin, d_in, d_out = voxel_fields(mesh, pitch)
    surf = np.argwhere(surface_voxels(occ))
    surf_xyz = (surf + 0.5) * pitch + origin
    lattice = np.zeros_like(occ); lattice[::stride, ::stride, ::stride] = True   # candidate centres on a coarser lattice
    cand_idx = np.argwhere(occ & lattice & (d_in >= min_radius))
    if len(cand_idx) == 0:
        cand_idx = np.argwhere(occ & lattice)
    cand_xyz = (cand_idx + 0.5) * pitch + origin
    cand_r = d_in[tuple(cand_idx.T)] + bulge                         # bulge-bounded radii
    tree = cKDTree(surf_xyz)
    covered = np.zeros(len(surf_xyz), dtype=bool)
    # lazy greedy: heap of (-gain, candidate); gains only shrink as coverage grows
    members = [None] * len(cand_xyz)
    heap = []
    for c in range(len(cand_xyz)):
        members[c] = np.asarray(tree.query_ball_point(cand_xyz[c], cand_r[c] - 0.5 * pitch), dtype=int)
        heap.append((-len(members[c]), c))
    heapq.heapify(heap)
    chosen = []
    while heap and not covered.all() and len(chosen) < max_spheres:
        neg_gain, c = heapq.heappop(heap)
        gain = int(np.count_nonzero(~covered[members[c]]))
        if gain == 0:
            continue
        if heap and gain < -heap[0][0]:                              # stale: re-insert with the true gain
            heapq.heappush(heap, (-gain, c))
            continue
        covered[members[c]] = True
        chosen.append((float(cand_xyz[c][0]), float(cand_xyz[c][1]), float(cand_xyz[c][2]), float(cand_r[c])))
    uncovered = float(np.count_nonzero(~covered)) / max(1, len(covered))
    return chosen, uncovered


def evaluate(spheres_by_link, meshes, pitch, samples=20000, seed=0):
    """Per link: max/p99 bulge of the sphere union beyond the visual mesh and the uncovered surface fraction."""
    import trimesh
    rng = np.random.default_rng(seed)
    rows = []
    for link, mesh in meshes.items():
        sph = np.asarray(spheres_by_link.get(link, []), float).reshape(-1, 4)
        occ, origin, d_in, d_out = voxel_fields(mesh, pitch)
        # bulge: sample each sphere's surface, read the outside-distance field
        bul = []
        for x, y, z, r in sph:
            v = rng.normal(size=(400, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
            pts = np.array([x, y, z]) + r * v
            ijk = np.floor((pts - origin) / pitch).astype(int)
            inside = np.all((ijk >= 0) & (ijk < np.array(occ.shape)), axis=1)
            d = np.full(len(pts), np.inf)
            d[inside] = d_out[tuple(ijk[inside].T)]
            d[~inside] = np.linalg.norm(pts[~inside] - np.clip(pts[~inside], origin, origin + np.array(occ.shape) * pitch), axis=1) + 2 * pitch
            bul.append(d)
        bul = np.concatenate(bul) if bul else np.zeros(1)
        # coverage: surface samples of the mesh vs the sphere union
        pts, _ = trimesh.sample.sample_surface(mesh, samples, seed=seed)
        if len(sph):
            dmin = np.min(np.linalg.norm(pts[:, None, :] - sph[None, :, :3], axis=2) - sph[None, :, 3], axis=1)
        else:
            dmin = np.full(len(pts), np.inf)
        rows.append((link, len(sph), float(bul.max()), float(np.percentile(bul, 99)),
                     float(np.mean(dmin > 1e-3)), float(np.percentile(np.maximum(dmin, 0), 99))))
    return rows


def spheres_from_urdf(path):
    out = {}
    for link in ET.parse(path).getroot().findall("link"):
        sp = []
        for col in link.findall("collision"):
            s = col.find("geometry/sphere")
            if s is None:
                continue
            o = col.find("origin")
            xyz = np.fromstring(o.get("xyz", "0 0 0") if o is not None else "0 0 0", sep=" ")
            sp.append((*xyz, float(s.get("radius"))))
        if sp:
            out[link.get("name")] = sp
    return out


def write_sphere_urdf(spheres_by_link, out, kinematic_urdf, comment):
    tree = ET.parse(kinematic_urdf); root = tree.getroot()
    n = 0
    for link in root.findall("link"):
        for col in link.findall("collision"):
            link.remove(col)
        for x, y, z, r in spheres_by_link.get(link.get("name"), []):
            col = ET.SubElement(link, "collision")
            ET.SubElement(col, "origin", rpy="0 0 0", xyz=f"{x:.6f} {y:.6f} {z:.6f}")
            ET.SubElement(ET.SubElement(col, "geometry"), "sphere", radius=f"{r:.6f}")
            n += 1
    root.insert(0, ET.Comment(comment))
    ET.indent(tree, space="  ")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tree.write(out, xml_declaration=True, encoding="unicode")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bulge-mm", type=float, default=10.0)
    ap.add_argument("--pitch-mm", type=float, default=2.5)
    ap.add_argument("--min-radius-mm", type=float, default=8.0, help="ignore inscribed radii below this (thin bits)")
    ap.add_argument("--max-per-link", type=int, default=400)
    ap.add_argument("--cand-stride", type=int, default=2, help="candidate-centre lattice stride in voxels")
    ap.add_argument("--out", default="")
    ap.add_argument("--report", nargs="*", default=[], help="sphere URDFs to evaluate against the visual meshes")
    ap.add_argument("--urdf", default=str(_KINEMATIC_URDF))
    args = ap.parse_args()
    pitch = args.pitch_mm * 1e-3
    meshes = load_visual_meshes()

    def print_report(title, rows):
        print(f"\n{title}")
        print(f"{'link':18s} {'spheres':>7s} {'bulge max':>10s} {'bulge p99':>10s} {'uncovered>1mm':>14s} {'gap p99':>8s}")
        for link, n, bmax, b99, unc, gap99 in rows:
            print(f"{link:18s} {n:7d} {bmax*1000:8.1f} mm {b99*1000:8.1f} mm {100*unc:12.1f} % {gap99*1000:6.1f} mm")
        print(f"{'TOTAL':18s} {sum(r[1] for r in rows):7d} {max(r[2] for r in rows)*1000:8.1f} mm")

    for path in args.report:
        print_report(f"== {path}", evaluate(spheres_from_urdf(path), meshes, pitch))
    if args.out:
        model = {}
        for link, mesh in meshes.items():
            sph, unc = fit_link(mesh, pitch, args.bulge_mm * 1e-3, args.max_per_link, args.min_radius_mm * 1e-3,
                                args.cand_stride)
            if unc > 0.005:   # thin parts (fingers, hand plates) need the full lattice and no radius floor
                sph, unc = fit_link(mesh, pitch, args.bulge_mm * 1e-3, args.max_per_link, 0.0, 1)
            model[link] = sph
            print(f"{link}: {len(sph)} spheres, surface voxels uncovered {100*unc:.2f} %")
        print_report(f"== fitted model (bulge budget {args.bulge_mm:g} mm, pitch {args.pitch_mm:g} mm)",
                     evaluate(model, meshes, pitch))
        n = write_sphere_urdf(model, args.out, args.urdf,
                              f" bounded-bulge spheres from panda_description visual meshes: bulge <= {args.bulge_mm:g} mm, "
                              f"pitch {args.pitch_mm:g} mm; generated by benchmark/make_bounded_bulge_spheres.py ")
        print(f"wrote {args.out}: {n} spheres")


if __name__ == "__main__":
    main()
