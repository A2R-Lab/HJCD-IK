from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external"))

import math

def quat_mul_wxyz(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ]

def quat_norm_wxyz(q):
    w, x, y, z = q
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n > 0.0:
        inv = 1.0 / n
        return [w*inv, x*inv, y*inv, z*inv]
    return [1.0, 0.0, 0.0, 0.0]

def rotate_quat_90deg_local_y_wxyz(q_wxyz, sign = +1):
    s = 0.7071067811865476 # sqrt(1/2)
    d = [s, 0.0, float(sign)*s, 0.0] # wxyz
    return quat_norm_wxyz(quat_mul_wxyz(q_wxyz, d))

def _load_text(path):
    return path.expanduser().read_text(encoding="utf-8")


def num_problems(D, problem_set):
    if "problems" not in D or problem_set not in D["problems"]:
        raise KeyError(
            f"Problem set '{problem_set}' not found. Available: {list(D.get('problems', {}).keys())}"
        )
    return len(D["problems"][problem_set])


def _get_instance(D, problem_set, problem_idx):
    return D["problems"][problem_set][problem_idx]

def _parse_batches(s):
    parts = [p.strip() for p in s.replace(",", " ").split()]
    vals = [int(p) for p in parts if p]
    if not vals:
        raise argparse.ArgumentTypeError("batches list is empty")
    return vals

def _cylinders_list(inst):
    cyl_block = inst.get("obstacles", {}).get("cylinder", {})
    out: List[Dict[str, Any]] = []

    if isinstance(cyl_block, dict):
        for name, c in cyl_block.items():
            if isinstance(c, dict) and "pose" in c:
                cc = dict(c)
                cc["name"] = name
                out.append(cc)
        return out

    if isinstance(cyl_block, list):
        for i, c in enumerate(cyl_block):
            if isinstance(c, dict) and "pose" in c:
                cc = dict(c)
                cc.setdefault("name", f"cylinder{i}")
                out.append(cc)
        return out

    return []

def has_any_cylinders(inst) -> bool:
    return len(_cylinders_list(inst)) > 0

def goal_pose_wxyz(inst):
    gp = inst.get("goal_pose")
    if gp is None:
        raise KeyError("instance missing goal_pose")

    pos = gp.get("position_xyz") or gp.get("position")
    if pos is None:
        raise KeyError("goal_pose missing position_xyz/position")

    qwxyz = gp.get("quaternion_wxyz") or gp.get("quat_wxyz")
    if qwxyz is None:
        raise KeyError("goal_pose missing quaternion_wxyz/quat_wxyz")

    qw, qx, qy, qz = [float(v) for v in qwxyz]
    return [float(pos[0]), float(pos[1]), float(pos[2]), qw, qx, qy, qz]

def associate_pos_to_closest_cylinder(inst, gx, gy, gz, *, eps):
    cyls = _cylinders_list(inst)
    if not cyls:
        raise RuntimeError("No cylinders found in inst['obstacles']['cylinder'] (expected at least 1).")

    best = None
    best_key = None  # (-match_axes, dist2)

    for c in cyls:
        pose = c["pose"]
        cx, cy, cz = float(pose[0]), float(pose[1]), float(pose[2])

        dx, dy, dz = gx - cx, gy - cy, gz - cz
        dist2 = dx * dx + dy * dy + dz * dz

        match_axes = 0
        if abs(dx) <= eps:
            match_axes += 1
        if abs(dy) <= eps:
            match_axes += 1
        if abs(dz) <= eps:
            match_axes += 1

        key = (-match_axes, dist2)
        if best_key is None or key < best_key:
            best_key = key
            best = c

    assert best is not None
    return best

def build_target_cylinder_pose(inst, ref_pos_xyz, *, eps):
    gx, gy, gz = ref_pos_xyz

    # pick the cylinder closest to the GOAL position
    cyl = associate_pos_to_closest_cylinder(inst, gx, gy, gz, eps=eps)
    cpose = cyl["pose"]  # [cx,cy,cz,qw,qx,qy,qz] wxyz

    # orientation comes from GOAL POSE
    goal = goal_pose_wxyz(inst)  # [gx,gy,gz,qw,qx,qy,qz] wxyz
    q = [float(goal[3]), float(goal[4]), float(goal[5]), float(goal[6])]  # wxyz

    return [
        float(cpose[0]), float(cpose[1]), float(goal[2]),
        float(q[0]), float(q[1]), float(q[2]), float(q[3]),
    ]

def build_target_from_goal_pose(inst):
    # goal_pose_wxyz returns [x,y,z,qw,qx,qy,qz]
    return goal_pose_wxyz(inst)

def _load_filtered_targets(path):
    D = json.loads(path.expanduser().read_text("utf-8"))
    items = D["targets"] if isinstance(D, dict) and "targets" in D else D
    if not isinstance(items, list):
        raise ValueError("filtered targets JSON must be a list or {'targets': [...]}")

    targets = []
    pidxs = []

    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise ValueError(f"entry {i} must be an object")
        if "problem_idx" not in it or "target" not in it:
            raise ValueError(f"entry {i} must have keys: problem_idx, target")

        pidx = int(it["problem_idx"])
        t = list(it["target"])
        if len(t) != 7:
            raise ValueError(f"entry {i} target must have 7 floats, got {len(t)}")

        targets.append([float(v) for v in t])
        pidxs.append(pidx)

    if not targets:
        raise RuntimeError("filtered targets file contained 0 targets")
    return targets, pidxs

# GRiD codegen
def run_grid_codegen(urdf, skip, fixed_target_name=""):
    if skip:
        print("[GRiD] skipping URDF codegen...")
        return False

    try:
        from GRiD.URDFParser import URDFParser
        from GRiD.GRiDCodeGenerator import GRiDCodeGenerator
    except Exception as e:
        raise RuntimeError(
            "Failed to import GRiD URDFParser/GRiDCodeGenerator. "
            "Check that ROOT/external/GRiD is present and on sys.path."
        ) from e

    urdf = urdf if urdf.is_absolute() else (ROOT / urdf).resolve()
    print(f"[GRiD] parsing {urdf}")
    robot = URDFParser().parse(str(urdf))
    codegen = GRiDCodeGenerator(robot, False, True)

    out_dir = ROOT / "external" / "GRiD"
    out_dir.mkdir(parents=True, exist_ok=True)

    cwd = os.getcwd()
    os.chdir(out_dir)
    try:
        if fixed_target_name:
            print(f"[GRiD] generating code with fixed target: {fixed_target_name}")
            codegen.gen_all_code(
                include_homogenous_transforms=True,
                fixed_target_name=fixed_target_name,
            )
        else:
            codegen.gen_all_code(include_homogenous_transforms=True)
    finally:
        os.chdir(cwd)

    print("[GRiD] codegen done!")
    return True

def rebuild_against_current_header():
    import subprocess

    env = os.environ.copy()
    env.pop("CMAKE_ARGS", None)
    print("[build] Rebuilding HJCD-IK against current header...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."], cwd=ROOT, env=env)
    print("[build] Rebuild done!")

def write_yaml_flat(path, batch_sizes, time_ms, pos_err, ori_err):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as y:
        y.write("Batch-Size:\n")
        for v in batch_sizes:
            y.write(f"  - {v}\n")
        y.write("IK-time(ms):\n")
        for v in time_ms:
            y.write(f"  - {v:.9f}\n")
        y.write("Pos-Error:\n")
        for v in pos_err:
            y.write(f"  - {v:.17g}\n")
        y.write("Ori-Error:\n")
        for v in ori_err:
            y.write(f"  - {v:.17g}\n")


def print_batch_summary(y_batch, y_time_ms, y_pos, y_ori):
    g_time = defaultdict(list)
    g_pos = defaultdict(list)
    g_ori = defaultdict(list)

    for B, t, p, o in zip(y_batch, y_time_ms, y_pos, y_ori):
        g_time[B].append(t)
        g_pos[B].append(p)
        g_ori[B].append(o)

    print("\n==== Batch Summary (averages) ====")
    for B in sorted(g_time.keys()):
        print(f"Batch Size {B}:")
        print(f"  Time (ms): {sum(g_time[B]) / len(g_time[B]):.6f}")
        print(f"  Position Error: {sum(g_pos[B]) / len(g_pos[B]):12.6e}")
        print(f"  Orientation Error: {sum(g_ori[B]) / len(g_ori[B]):12.6e}")

def write_csv_summary(path, solver, y_batch, y_time_ms, y_pos, y_ori):
    import csv
    from collections import defaultdict

    g_time = defaultdict(list)
    g_pos  = defaultdict(list)
    g_ori  = defaultdict(list)

    for B, t, p, o in zip(y_batch, y_time_ms, y_pos, y_ori):
        g_time[B].append(float(t))
        g_pos[B].append(float(p))
        g_ori[B].append(float(o))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["solver", "Batch-Size", "time_ms", "pos_err_mm", "ori_err_rad"])

        for B in sorted(g_time.keys()):
            time_ms = sum(g_time[B]) / len(g_time[B])
            pos_mm  = (sum(g_pos[B]) / len(g_pos[B]))
            ori_rad = sum(g_ori[B]) / len(g_ori[B])

            w.writerow([solver, int(B), f"{time_ms:.9f}", f"{pos_mm:.9g}", f"{ori_rad:.9g}"])

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--skip-grid-codegen", action="store_true", help="Skip URDF parse/codegen step for GRiD.")
    ap.add_argument( "--urdf", type=str, default=str(ROOT / "include" / "test_urdf" / "panda.urdf"), help="URDF used for GRiD codegen.")
    ap.add_argument("--grid-target", type=str, default="", help="Optional GRiD fixed kinematic target name, e.g. panda_grasptarget_hand.")
    ap.add_argument("--yaml-out", type=str, default="results.yml",help="YAML output file name.")
    ap.add_argument("--batches",type=_parse_batches,default=_parse_batches("1,10,100,1000,2000"), help="Batch sizes (comma/space separated).")
    ap.add_argument("--num-solutions", type=int, default=1,help="Number of returned solutions per target.")
    ap.add_argument("--print-solutions", action="store_true",help="Print joint configs (original behavior).")

    # Non-collision
    ap.add_argument("--num-targets", type=int, default=100,help="Number of random targets (non-collision benchmark).")

    # Collision-free (RoboMetrics)
    ap.add_argument("--collision-free", action="store_true",help="Enable collision-free solutions.")
    ap.add_argument( "--problems-json", type=str, default=str(ROOT / "src" / "problems" / "panda_problems.json"),help="Path to problems JSON (e.g., mb_problems.json).")
    ap.add_argument("--problem-set", type=str, default="bookshelf_thin",help="Problem set name in JSON.")
    ap.add_argument("--problem-idx", type=int, default=-1,help="If >=0, run only this problem index; if -1 run all.")

    ap.add_argument("--assoc-eps", type=float, default=1e-4,help="Axis-coincidence epsilon for association (meters).")

    ap.add_argument("--filtered-targets", type=str, default="",help="JSON of explicit targets; xyz is used for association.")
    ap.add_argument("--max-targets", type=int, default=0,help="If >0, cap number of loaded targets (quick tests).")

    ap.add_argument("--csv-out", type=str, default="",help="If set, write a CSV summary with columns: solver,Batch-Size,time_ms,pos_err_mm,ori_err_rad")
    ap.add_argument("--solver", type=str, default="hjcdik",help="Solver name to emit in CSV (default: hjcdik).")
    
    ap.add_argument("--solutions-out", type=str, default="",help="Write joint configs for ONE chosen target (one solution per line: q1,q2,...,q7).")
    ap.add_argument("--solutions-target-idx", type=int, default=0,help="0-based target index to dump solutions for (default: 0 = first target).")
    ap.add_argument("--solutions-batch", type=int, default=-1,help="Batch size to dump solutions from. If -1, uses the last batch in --batches.")
    ap.add_argument("--solutions-count", type=int, default=50,help="Max number of solutions to write (default 50).")

    args = ap.parse_args()
    batches = list(args.batches) 
    S = int(args.num_solutions)

    dump_B = args.solutions_batch if args.solutions_batch > 0 else batches[-1]
    dump_idx = int(args.solutions_target_idx)
    dump_limit = int(args.solutions_count)
    dumped = False
    dump_solutions = []

    # GRiD codegen + rebuild
    did_codegen = run_grid_codegen(
        Path(args.urdf),
        args.skip_grid_codegen,
        args.grid_target,
    )
    if did_codegen:
        rebuild_against_current_header()

    import importlib
    hjcdik = importlib.import_module("hjcdik")
    try:
        print("[build info]", hjcdik.build_info())
    except Exception:
        pass

    problems_text = ""
    targets = []
    problem_indices = []

    if args.collision_free:
        problems_text = _load_text(Path(args.problems_json))
        D = json.loads(problems_text)
        P = num_problems(D, args.problem_set)

        if args.problem_idx is None or args.problem_idx < 0:
            problem_range = list(range(P))
        else:
            if args.problem_idx >= P:
                raise IndexError(f"--problem-idx {args.problem_idx} out of range (0..{P-1}).")
            problem_range = [args.problem_idx]

        for pidx in problem_range:
            inst = _get_instance(D, args.problem_set, pidx)

            goal = goal_pose_wxyz(inst)
            ref_xyz = (goal[0], goal[1], goal[2])

            #targets.append(build_target_cylinder_pose(inst, ref_xyz, eps=float(args.assoc_eps)))
            if has_any_cylinders(inst):
                targets.append(build_target_cylinder_pose(inst, ref_xyz, eps=float(args.assoc_eps)))
            else:
                targets.append(build_target_from_goal_pose(inst))
            problem_indices.append(pidx)

        if args.max_targets and args.max_targets > 0:
            targets = targets[: args.max_targets]
            problem_indices = problem_indices[: args.max_targets]

        if not targets:
            raise RuntimeError("No targets built (goal_pose missing or empty problem set).")

    else:
        # Non-collision: random targets sampled from model
        try:
            N = int(hjcdik.num_joints())
            print(f"[info] robot with {N} joints")
        except Exception:
            print("[info] loaded hjcdik")

        T = int(args.num_targets)
        targets = hjcdik.sample_targets(T, seed=0)
        problem_indices = [None] * T

    print(f"[info] running {len(targets)} targets, batches={batches}, num_solutions={S}, collision_free={args.collision_free}")

    # Benchmark loop
    y_batch = []
    y_time_ms = []
    y_pos = []
    y_ori = []

    for i, (target, pidx) in enumerate(zip(targets, problem_indices)):
        for B in batches:
            if args.collision_free:
                if pidx is not None:
                    eff_pidx = int(pidx)
                elif args.problem_idx is not None and args.problem_idx >= 0:
                    eff_pidx = int(args.problem_idx)
                else:
                    eff_pidx = 0 
            else:
                eff_pidx = -1 

            # Warmup
            _ = hjcdik.generate_solutions(
                target,
                batch_size=B,
                num_solutions=S,
                collision_free=args.collision_free,
                problems_json_text=problems_text,
                problem_set_name=args.problem_set,
                problem_idx=eff_pidx,
            )

            # Timed run
            t0 = time.perf_counter()
            res = hjcdik.generate_solutions(
                target,
                batch_size=B,
                num_solutions=S,
                collision_free=args.collision_free,
                problems_json_text=problems_text,
                problem_set_name=args.problem_set,
                problem_idx=eff_pidx,
            )
            dt_ms = (time.perf_counter() - t0) * 1e3

            count = int(res.get("count", S))
            pos_err = res["pos_errors"]
            ori_err = res["ori_errors"]

            for r in range(count):
                y_batch.append(B)
                y_time_ms.append(dt_ms)
                y_pos.append(float(pos_err[r]))
                y_ori.append(float(ori_err[r]))

            if args.print_solutions:
                joint_cfg = res.get("joint_config", None)
                if joint_cfg is not None:
                    for r in range(count):
                        sol = joint_cfg[r]
                        print("[joint_config] " + ", ".join([f"{v:.6f}" for v in sol]))

            if args.solutions_out and (not dumped) and (i == dump_idx) and (B == dump_B):
                joint_cfg = res.get("joint_config", None)
                if joint_cfg is not None:
                    count = int(res.get("count", len(joint_cfg)))
                    for r in range(min(count, len(joint_cfg), dump_limit)):
                        dump_solutions.append([float(v) for v in joint_cfg[r]])
                    dumped = True

        if (i % 50) == 0 or i == len(targets):
            print(f"[info] processed {i}/{len(targets)} targets")

    print_batch_summary(y_batch, y_time_ms, y_pos, y_ori)

    if args.solutions_out:
        sol_path = Path(args.solutions_out)
        if not sol_path.is_absolute():
            sol_path = (ROOT / sol_path).resolve()
        sol_path.parent.mkdir(parents=True, exist_ok=True)

        with sol_path.open("w", encoding="utf-8") as f:
            for q in dump_solutions:
                f.write(",".join(f"{v:.10f}" for v in q) + "\n")

        print(f"[OK] wrote {len(dump_solutions)} solutions to {sol_path} "
            f"(target_idx={dump_idx}, batch={dump_B})")

    if args.csv_out:
        csv_path = Path(args.csv_out)
        if not csv_path.is_absolute():
            csv_path = (ROOT / csv_path).resolve()
        write_csv_summary(csv_path, args.solver, y_batch, y_time_ms, y_pos, y_ori)
        print(f"[OK] wrote CSV summary {csv_path}")

    out_path = Path(args.yaml_out)
    if not out_path.is_absolute():
        out_path = (ROOT / out_path).resolve()
    write_yaml_flat(out_path, y_batch, y_time_ms, y_pos, y_ori)
    print(f"\n[OK] wrote {out_path} with {len(targets) * S * len(batches)} entries "
          f"({len(targets)} targets x {len(batches)} batches x {S} solutions each).")


if __name__ == "__main__":
    main()
