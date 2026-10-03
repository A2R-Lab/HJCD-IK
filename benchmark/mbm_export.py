#!/usr/bin/env python3
"""Export MotionBenchMaker problem folders to the `tests/mb_problems.json` schema.

MotionBenchMaker (KavrakiLab/motion_bench_maker, `problems/download.sh panda`) ships each scene as 100
MoveIt YAML pairs: `sceneNNNN.yaml` (collision objects in the robot root frame — MBM plants the robot at the
origin and shifts the scene) and `requestNNNN.yaml` (a joint-space goal and start state). The existing
`tests/mb_problems.json` sets are exactly these files: box/cylinder primitives became our `cuboid`/`cylinder`
obstacles (quaternion xyzw -> wxyz), plus a robot stand under the base.

The scenes we did not have (kitchen, table_bars) are triangle meshes, which HJCD's environment does not take.
Each mesh is split into connected components and each component becomes primitives:
  * a component that fills its oriented bounding box (volume ratio >= BOX_FILL) -> one cuboid (exact);
  * a regular prism (64-vertex, fill ratio pi/4 like MBM's bars and door handles) -> one cylinder with the
    polygon's circumradius (conservative by r(1-cos(pi/n)) ~ 0.5 % of r);
  * a rectilinear component (every face axis-aligned in the mesh frame: the hollow cupboard, the counter) ->
    the exact union of axis-aligned boxes on the grid of its vertex coordinates (cells classified by x-ray
    parity, so a hollow interior stays free — a convex hull would fill the space the robot reaches into);
  * anything else -> its oriented bounding box, over-approximated, and the exporter says so.
Goal: `goal_pose` = panda_hand FK of the request's joint goal (so it is a `panda_hand` pose, like the dataset
sets), `goal_ik` = [that joint goal]. `start` = the request start state.

  python benchmark/mbm_export.py <mbm_root>/kitchen_panda <mbm_root>/table_bars_panda --out mb_extra.json
  python benchmark/mbm_export.py <mbm_root>/cage_panda --out /tmp/cage.json --compare tests/mb_problems.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_targets import _parse_joints, _chain_to_target, _fk, _quat_from_R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PANDA_URDF = ROOT / "csrc/urdf/panda.urdf"
ROBOT_STAND = {"dims": [0.3, 0.25, 0.8], "pose": [-0.05, 0.0, -0.4, 1.0, 0.0, 0.0, 0.0]}   # as in the dataset sets
BOX_FILL = 0.985
PRISM_FILL = (np.pi / 4.0 * 0.97, np.pi / 4.0 * 1.03)
ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]


def _xyzw_to_wxyz(q):
    x, y, z, w = [float(v) for v in q]
    return [w, x, y, z]


def _quat_to_R(wxyz):
    w, x, y, z = wxyz
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _pose7(T):
    return [float(v) for v in T[:3, 3]] + [float(v) for v in _quat_from_R(T[:3, :3])]


def _T(position, wxyz):
    T = np.eye(4)
    T[:3, :3] = _quat_to_R(wxyz)
    T[:3, 3] = position
    return T


def _prism_axis(ext):
    """Index of the extent that differs from the other two (regular-prism axis), or None if ambiguous."""
    for a in range(3):
        o = [i for i in range(3) if i != a]
        if np.isclose(ext[o[0]], ext[o[1]], rtol=0.02) and not np.isclose(ext[a], ext[o[0]], rtol=0.02):
            return a
    return None


def _is_rectilinear(mesh) -> bool:
    n = mesh.face_normals
    return bool(np.all(np.isclose(np.abs(n).max(axis=1), 1.0, atol=1e-4)))


def _rectilinear_boxes(mesh):
    """Exact union-of-boxes of a watertight rectilinear (axis-aligned-face) component: the grid of its vertex
    coordinates is classified cell by cell with trimesh's point containment, then greedily merged. Returns
    [(center(3), dims(3))] in the mesh frame; the merged volume equals the mesh volume."""
    v = mesh.vertices
    xs, ys, zs = (np.unique(np.round(v[:, i], 4)) for i in range(3))     # 0.1 mm grid
    cx = 0.5 * (xs[:-1] + xs[1:]); cy = 0.5 * (ys[:-1] + ys[1:]); cz = 0.5 * (zs[:-1] + zs[1:])
    centers = np.array([[x, y, z] for x in cx for y in cy for z in cz])
    inside = mesh.contains(centers).reshape(len(cx), len(cy), len(cz))
    boxes = []
    for j in range(len(cy)):
        for k in range(len(cz)):
            i = 0
            while i < len(cx):
                if not inside[i, j, k]:
                    i += 1
                    continue
                i0 = i
                while i < len(cx) and inside[i, j, k]:
                    i += 1
                lo = np.array([xs[i0], ys[j], zs[k]]); hi = np.array([xs[i], ys[j + 1], zs[k + 1]])
                boxes.append((0.5 * (lo + hi), hi - lo))
    merged = _merge_boxes(boxes)
    vol = sum(float(np.prod(d)) for _, d in merged)
    if not np.isclose(vol, abs(mesh.volume), rtol=1e-3, atol=1e-7):
        raise RuntimeError(f"rectilinear decomposition volume {vol:.5f} != mesh volume {abs(mesh.volume):.5f}")
    return merged


def _merge_boxes(boxes):
    """Merge boxes that share identical extents+centre in all but one axis and touch along it (y then z)."""
    for axis in (1, 2):
        merged = True
        while merged:
            merged = False
            out = []
            used = [False] * len(boxes)
            for a in range(len(boxes)):
                if used[a]:
                    continue
                ca, da = boxes[a]
                for b in range(a + 1, len(boxes)):
                    if used[b]:
                        continue
                    cb, db = boxes[b]
                    other = [i for i in range(3) if i != axis]
                    if (np.allclose(ca[other], cb[other], atol=1e-6) and np.allclose(da[other], db[other], atol=1e-6)
                            and np.isclose(abs(ca[axis] - cb[axis]), 0.5 * (da[axis] + db[axis]), atol=1e-6)):
                        lo = min(ca[axis] - da[axis] / 2, cb[axis] - db[axis] / 2)
                        hi = max(ca[axis] + da[axis] / 2, cb[axis] + db[axis] / 2)
                        c = ca.copy(); d = da.copy(); c[axis] = 0.5 * (lo + hi); d[axis] = hi - lo
                        boxes[a] = (c, d); ca, da = c, d
                        used[b] = True
                        merged = True
                out.append(boxes[a])
            boxes = out
    return boxes


def mesh_to_primitives(vertices, triangles, T_mesh, name, report):
    """Yield ('cuboid'|'cylinder', dict) primitives in the world frame for one MBM mesh object."""
    import trimesh
    mesh = trimesh.Trimesh(np.asarray(vertices, float), np.asarray(triangles, int), process=True)
    prims = []
    for ci, comp in enumerate(mesh.split(only_watertight=False)):
        if len(comp.faces) < 4 or comp.area < 1e-9:
            continue                                                   # stray flat quad / degenerate
        obb = comp.bounding_box_oriented
        ext = obb.primitive.extents
        if obb.volume <= 1e-12:
            continue                                                   # degenerate (flat) component
        vol = abs(comp.volume) if comp.is_watertight else np.nan
        fill = vol / obb.volume if np.isfinite(vol) else np.nan
        T_obb = T_mesh @ obb.primitive.transform
        tag = f"{name}/{ci}"
        if np.isfinite(fill) and fill >= BOX_FILL:
            prims.append(("cuboid", {"dims": [float(e) for e in ext], "pose": _pose7(T_obb)}))
            report.append((tag, "box", fill))
        elif np.isfinite(fill) and PRISM_FILL[0] <= fill <= PRISM_FILL[1] and len(comp.vertices) % 2 == 0 \
                and _prism_axis(ext) is not None:
            axis = _prism_axis(ext)                                     # the extent unlike the other two
            R = T_obb[:3, :3]
            # circumradius: farthest vertex from the axis (conservative for the polygon)
            local = (comp.vertices - obb.primitive.transform[:3, 3]) @ obb.primitive.transform[:3, :3]
            radial = np.delete(local, axis, axis=1)
            r = float(np.linalg.norm(radial, axis=1).max())
            # cylinder frame: z along the prism axis
            z = R[:, axis]; x = R[:, (axis + 1) % 3]; y = np.cross(z, x)
            Tc = np.eye(4); Tc[:3, :3] = np.stack([x, y, z], axis=1); Tc[:3, 3] = T_obb[:3, 3]
            prims.append(("cylinder", {"radius": r, "height": float(ext[axis]), "pose": _pose7(Tc)}))
            report.append((tag, "cylinder", fill))
        elif comp.is_watertight and _is_rectilinear(comp):
            boxes = _rectilinear_boxes(comp)
            for c, d in boxes:
                Tb = T_mesh @ _T(c, [1, 0, 0, 0])
                prims.append(("cuboid", {"dims": [float(v) for v in d], "pose": _pose7(Tb)}))
            report.append((tag, f"rectilinear->{len(boxes)} boxes", fill))
        else:
            prims.append(("cuboid", {"dims": [float(e) for e in ext], "pose": _pose7(T_obb)}))
            report.append((tag, "OBB (over-approximation!)", fill))
    return prims


def export_problem(scene_path: Path, request_path: Path, chain, report, robot_stand=True) -> dict:
    scene = yaml.safe_load(open(scene_path))
    req = yaml.safe_load(open(request_path))
    cuboids, cylinders = {}, {}
    for obj in scene["world"]["collision_objects"]:
        oid = str(obj.get("id", f"obj{len(cuboids) + len(cylinders)}"))
        for prim, pose in zip(obj.get("primitives", []), obj.get("primitive_poses", [])):
            p7 = [float(v) for v in pose["position"]] + _xyzw_to_wxyz(pose["orientation"])
            dims = [float(d) for d in prim["dimensions"]]
            if prim["type"] == "box":
                cuboids[f"cube{len(cuboids)}"] = {"dims": dims, "pose": p7}
            elif prim["type"] == "cylinder":                           # MoveIt: [height, radius]
                cylinders[f"cylinder{len(cylinders)}"] = {"radius": dims[1], "height": dims[0], "pose": p7}
            else:
                raise ValueError(f"{scene_path.name}: unsupported primitive {prim['type']} on {oid}")
        for mesh, pose in zip(obj.get("meshes", []), obj.get("mesh_poses", [])):
            T_mesh = _T([float(v) for v in pose["position"]], _xyzw_to_wxyz(pose["orientation"]))
            for kind, prim in mesh_to_primitives(mesh["vertices"], mesh["triangles"], T_mesh, oid, report):
                if kind == "cuboid":
                    cuboids[f"cube{len(cuboids)}"] = prim
                else:
                    cylinders[f"cylinder{len(cylinders)}"] = prim
    if robot_stand:
        cuboids["cube_robot_stand"] = dict(ROBOT_STAND)

    goal_c = req["goal_constraints"][0]["joint_constraints"]
    goal = {c["joint_name"]: float(c["position"]) for c in goal_c}
    q_goal = [goal[j] for j in ARM_JOINTS]
    T_hand = _fk(chain, q_goal)
    st = req["start_state"]["joint_state"]
    start = dict(zip(st["name"], [float(v) for v in st["position"]]))
    return {
        "collision_buffer_ik": 0,
        "goal_ik": [q_goal],
        "goal_pose": {"frame": "panda_hand", "position_xyz": [float(v) for v in T_hand[:3, 3]],
                      "quaternion_wxyz": [float(v) for v in _quat_from_R(T_hand[:3, :3])]},
        "obstacles": {"cuboid": cuboids, "cylinder": cylinders},
        "start": [start[j] for j in ARM_JOINTS],
        "world_frame": "panda_link0",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", help="MBM problem folders (sceneNNNN.yaml + requestNNNN.yaml)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--urdf", default=str(PANDA_URDF))
    ap.add_argument("--no-robot-stand", action="store_true")
    ap.add_argument("--compare", default="", help="existing problems JSON to diff same-named sets against")
    args = ap.parse_args()

    joints = _parse_joints(args.urdf)
    chain = _chain_to_target(joints, "panda_hand_joint")
    out = {}
    for folder in args.folders:
        folder = Path(folder)
        name = folder.name
        scenes = sorted(folder.glob("scene[0-9]*.yaml"))
        report = []
        probs = []
        for sp in scenes:
            rp = folder / sp.name.replace("scene", "request")
            if not rp.exists():
                sys.exit(f"missing {rp}")
            probs.append(export_problem(sp, rp, chain, report, robot_stand=not args.no_robot_stand))
        out[name] = probs
        kinds = {}
        for tag, kind, fill in report:
            kinds.setdefault(kind.split(" ")[0], 0)
            kinds[kind.split(" ")[0]] += 1
        nc = [len(p["obstacles"]["cuboid"]) for p in probs]; ncy = [len(p["obstacles"]["cylinder"]) for p in probs]
        print(f"{name}: {len(probs)} problems; cuboids/problem {min(nc)}-{max(nc)}, cylinders {min(ncy)}-{max(ncy)}; "
              f"mesh components -> {kinds if kinds else 'no meshes'}")
        bad = sorted({tag for tag, kind, _ in report if kind.startswith("OBB")})
        if bad:
            print(f"  over-approximated components: {bad[:10]}{' ...' if len(bad) > 10 else ''}")
    if args.compare:
        ref = json.load(open(args.compare))["problems"]
        for name, probs in out.items():
            if name not in ref:
                continue
            dpos = []
            for a, b in zip(probs, ref[name]):
                dpos.append(np.linalg.norm(np.subtract(a["goal_pose"]["position_xyz"], b["goal_pose"]["position_xyz"])))
                ac, bc = a["obstacles"]["cuboid"], b["obstacles"]["cuboid"]
                dob = max((np.abs(np.subtract(ac[k]["pose"], bc[k]["pose"])).max() for k in ac if k in bc), default=0)
            print(f"  compare {name}: goal position |d| max {max(dpos) * 1000:.2f} mm; obstacle pose max |d| {dob:.2e}; "
                  f"cuboids {len(probs[0]['obstacles']['cuboid'])} vs {len(ref[name][0]['obstacles']['cuboid'])}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"problems": out}, open(args.out, "w"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
