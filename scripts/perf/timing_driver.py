#!/usr/bin/env python
"""Neutral A/B timing driver for comparing two HJCD-IK builds (e.g. main vs a branch).

Runs against WHICHEVER hjcdik the invoking interpreter resolves — invoke it once
with endpoint A's venv and once with endpoint B's venv, alternating rounds
(A/B/A/B) to cancel boost-clock drift (sequential blocks drift ~9% on the 5090).
Auto precision is fp64 for S=1 and fp32 for S>1. The driver speaks both HJCD-IK APIs so that the
Stage 1 matched-settings comparison can run: `hjcdik.Solver` (current) and the removed
`generate_solutions` (an endpoint built before 2026-10-23). Collision settings are matched: open-world
legs use collision=None / collision_free=False; table2 uses "hard" on both. One `solve` call per
target is the timing boundary on both APIs.
Both endpoints must be compiled for panda_hand_joint with verified header hashes.

Legs:
  table1: open-world latency vs batch over shared targets
          (--targets-json: {"targets": [[x,y,z,qw,qx,qy,qz], ...]}, same file for both endpoints)
  table2: collision-free box_panda, targets = goal poses from tests/mb_problems.json
          (identical file at both endpoints)

Output: appends one CSV row per (leg, batch) to --out:
  label,leg,batch,round,n_targets,median_ms,mean_ms,min_ms,p95_ms
Timing = per-call wall time of one solve across targets in the round.
"""
import argparse, csv, json, os, statistics, sys, time
from pathlib import Path

import hjcdik
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))
from query_results import best_candidate, pose_errors, query_record, append_record

COLUMNS = ["label", "leg", "batch", "round", "n_targets", "median_ms", "mean_ms",
           "min_ms", "p95_ms", "solved", "empty", "returned", "refine_fp64",
           "correctness_only", "header_sha"]


def load_shared_targets(path):
    from hjcd_ik_bench import _load_filtered_targets
    targets, _ = _load_filtered_targets(Path(path))
    return targets


def goal_pose_targets(problems_json_path, set_name):
    D = json.load(open(problems_json_path))["problems"][set_name]
    out = []
    for inst in D:
        gp = inst.get("goal_pose")
        if not gp:
            raise ValueError("missing goal_pose would misalign scene indices")
        pos = gp.get("position_xyz") or gp.get("position")
        quat = gp.get("quaternion_wxyz") or gp.get("quat_wxyz")  # [w,x,y,z]
        out.append([pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]])
    if not out:
        raise RuntimeError(f"no goal_pose targets in set {set_name}")
    return out


NEW_API = hasattr(hjcdik, "Solver")
PRECISION = {-1: "auto", 0: "fp32", 1: "fp64"}
# Endpoint knobs for the gate (set per endpoint in gate.json "env"): a prepared runtime build to time
# instead of the packaged robot, and a clearance margin for the collision leg (hard mode only).
TIMING_ROBOT = os.environ.get("HJCDIK_TIMING_ROBOT") or None
TIMING_MARGIN = float(os.environ.get("HJCDIK_TIMING_MARGIN", "0") or 0)
TIMING_ENTRY = os.environ.get("HJCDIK_TIMING_ENTRY", "api")
TIMING_URDF = os.environ.get("HJCDIK_TIMING_URDF")
if TIMING_ENTRY not in ("api", "native"):
    raise ValueError("HJCDIK_TIMING_ENTRY must be api or native")
REFERENCE_CHAIN = None
if TIMING_URDF:
    from gen_targets import _parse_joints, _chain_to_target, _fk, _quat_from_R
    REFERENCE_CHAIN = _chain_to_target(_parse_joints(TIMING_URDF), "panda_hand_joint")


def independent_errors(q, target):
    if REFERENCE_CHAIN is None:
        return pose_errors(q, target, "panda_hand_joint")
    transform = _fk(REFERENCE_CHAIN, q)
    pos_mm = 1000 * np.linalg.norm(transform[:3, 3] - np.asarray(target[:3]))
    dot = np.clip(abs(np.dot(_quat_from_R(transform[:3, :3]), target[3:])), 0, 1)
    return float(pos_mm), float(2 * np.arccos(dot))


def unpack_native(raw):
    """Drop batch/padding outside the timed native call; preserve empty attempts."""
    n = int(raw["count"][0])
    return dict(count=n, joint_config=raw["joint_config"][0, :n],
                pos_errors=raw["pos_errors"][0, :n], ori_errors=raw["ori_errors"][0, :n])


def new_api_solver():
    return hjcdik.Solver(TIMING_ROBOT, allow_build=False) if TIMING_ROBOT else hjcdik.Solver()


def make_solve(batch, num_solutions, collision_free, pj_text, set_name, precision, targets):
    """(prepare, solve) closures with matched settings on either API.

    prepare(i) makes target i's world resident untimed (new API: set_world; old API: one untimed
    call, which fills its scene cache), so the timed boundary on both APIs is one solve with the
    world already on the device.
    """
    problems = json.loads(pj_text)["problems"][set_name] if collision_free else None
    if NEW_API:
        solver = new_api_solver()
        native_poses = [np.asarray(t,dtype=float).reshape(1,7) for t in targets]
        extra = {"collision_margin_m": TIMING_MARGIN} if (collision_free and TIMING_MARGIN > 0) else {}
        def prepare(i):
            if collision_free:
                solver.set_world(**hjcdik.worlds.from_mbm_problem(problems[i]))
        def solve(i, t):
            if TIMING_ENTRY == "native":
                return solver._native.solve_batch(native_poses[i],batch,num_solutions,
                    1 if collision_free else 0, (num_solutions==1) if precision==-1 else bool(precision),
                    False,solver._world,solver.info.device,TIMING_MARGIN if collision_free else 0.0,
                    True,1337,None,num_solutions)
            return solver.solve(t, batch_size=batch, num_solutions=num_solutions,
                                collision="hard" if collision_free else None,
                                precision=PRECISION[precision], **extra)[0]
        return prepare, solve
    def solve(i, t):
        return hjcdik.generate_solutions(t, batch_size=batch, num_solutions=num_solutions,
            collision_free=collision_free, problems_json_text=pj_text, problem_set_name=set_name,
            problem_idx=i if collision_free else 0, refine_fp64=precision, write_stats=False,
            collision_mode="hard")
    def prepare(i):
        if collision_free:
            solve(i, [0.3, 0, 0.5, 1, 0, 0, 0])
    return prepare, solve


def time_leg(targets, batch, num_solutions, collision_free, pj_text, set_name, warmup,
             precision=-1, correctness_only=False, configs_out=None, label="hjcdik"):
    from query_results import candidate_arrays
    if TIMING_URDF and collision_free:
        raise ValueError("custom reference URDF is for open-world high-DoF comparisons only")
    prepare, solve = make_solve(batch, num_solutions, collision_free, pj_text, set_name, precision, targets)
    for i, t in enumerate(targets[:warmup]):
        prepare(i)
        solve(i, t)
    times = []
    solved = empty = returned = 0
    problems = json.loads(pj_text)["problems"][set_name] if collision_free else None
    for i, t in enumerate(targets):
        prepare(i)
        t0 = None if correctness_only else time.perf_counter()
        result = solve(i, t)
        if t0 is not None:
            times.append((time.perf_counter() - t0) * 1e3)
        if NEW_API and TIMING_ENTRY == "native":
            result = unpack_native(result)
        if configs_out:
            append_record(configs_out, query_record(result, solver=label, problem_set=set_name,
                problem_idx=i, batch=batch, target=t, ee_target="panda_hand_joint",
                elapsed_ms=None if correctness_only else times[-1]))
        count, qs, pes, oes = candidate_arrays(result)
        returned += count
        best = best_candidate(result)
        if best is None:
            empty += 1
            continue
        good = False
        for j, q in enumerate(qs):
            pe, oe = independent_errors(q, t)
            if abs(pe - pes[j]) > .001 or abs(oe - oes[j]) > 1e-5:
                raise ValueError("reported errors disagree with independent FK")
            good |= pe < 5 and oe < .05
            if collision_free:
                from panda_collision import panda_config_collision_free, mb_instance_to_world_dict
                if not panda_config_collision_free(q, mb_instance_to_world_dict(problems[i]), model="hjcd"):
                    raise ValueError("hard mode returned an environment collision")
        solved += good
    return times, solved, empty, returned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="endpoint label, e.g. main-<sha> or branch-<sha>")
    ap.add_argument("--leg", choices=["table1", "table2"], required=True)
    ap.add_argument("--round", type=int, required=True, help="interleave round index (bookkeeping)")
    ap.add_argument("--targets-json", default="", help="table1: shared targets JSON")
    ap.add_argument("--problems-json", default="", help="table2: mb_problems.json path")
    ap.add_argument("--problem-set", default="box_panda")
    ap.add_argument("--batches", default="1,10,100,1000,2000")
    ap.add_argument("--num-solutions", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected-header-sha", required=True)
    ap.add_argument("--refine-fp64", type=int, choices=[-1, 0, 1], default=-1)
    ap.add_argument("--correctness-only", action="store_true", help="Do not read clocks or emit timings")
    ap.add_argument("--max-targets", type=int, default=0)
    ap.add_argument("--configs-out", help="Optional per-query JSONL, including empty results")
    args = ap.parse_args()

    # Provenance: make it impossible to silently time the wrong endpoint.
    print(f"[{args.label}] hjcdik = {hjcdik.__file__}")
    info = dict(hjcdik.build_info())
    if NEW_API and TIMING_ROBOT:   # a runtime build: its own header and artifact are the provenance
        s = new_api_solver()
        info.update(grid_header_sha256=s.info.grid_header_sha256, ee_target=s.info.ee_frame,
                    artifact=s.info.artifact, robot=s.info.robot)
    if TIMING_MARGIN:
        info["collision_margin_m"] = TIMING_MARGIN
    if info["grid_header_sha256"] != args.expected_header_sha:
        raise ValueError("wrong compiled header")
    if info.get("ee_target", "panda_hand_joint") != "panda_hand_joint":
        raise ValueError("this gate requires a panda_hand_joint build")
    print(json.dumps({"build": info, "precision": args.refine_fp64,
                      "correctness_only": args.correctness_only}))

    batches = [int(b) for b in args.batches.split(",")]
    if args.leg == "table1":
        targets = load_shared_targets(args.targets_json)
        collision_free, pj_text, set_name = False, "", ""
    else:
        targets = goal_pose_targets(args.problems_json, args.problem_set)
        collision_free = True
        pj_text = open(args.problems_json).read()
        set_name = args.problem_set

    if args.max_targets:
        targets = targets[:args.max_targets]
    if not targets:
        raise ValueError("empty workload")

    out = Path(args.out)
    new = not out.exists()
    if not new:
        with out.open(newline="") as stream:
            if next(csv.reader(stream), None) != COLUMNS:
                raise ValueError("refusing to append to a different CSV schema")
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLUMNS)
        for B in batches:
            ts, solved, empty, returned = time_leg(targets, B, args.num_solutions, collision_free,
                          pj_text, set_name, args.warmup, args.refine_fp64, args.correctness_only,
                          args.configs_out, args.label)
            ts_sorted = sorted(ts)
            metrics = ([statistics.median(ts), statistics.fmean(ts), min(ts),
                        ts_sorted[max(0, int(.95 * len(ts_sorted)) - 1)]] if ts else [None] * 4)
            w.writerow([args.label, args.leg, B, args.round, len(targets), *metrics,
                        solved, empty, returned, args.refine_fp64, args.correctness_only, info["grid_header_sha256"]])
            print(f"[{args.label} r{args.round}] {args.leg} B={B}: {solved}/{len(targets)} accurate; empty={empty}")


if __name__ == "__main__":
    main()
