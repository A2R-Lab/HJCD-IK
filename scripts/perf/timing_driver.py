#!/usr/bin/env python
"""Neutral A/B timing driver for comparing two HJCD-IK builds (e.g. main vs a branch).

Runs against WHICHEVER hjcdik the invoking interpreter resolves — invoke it once
with endpoint A's venv and once with endpoint B's venv, alternating rounds
(A/B/A/B) to cancel boost-clock drift (sequential blocks drift ~9% on the 5090).
All solver options are named: the eighth positional argument is refine_fp64,
NOT write_stats. Auto precision is fp64 for S=1 and fp32 for S>1.
Both endpoints must be compiled for panda_hand_joint with verified header hashes.

Legs:
  table1: open-world latency vs batch over shared targets
          (--targets-json: {"targets": [[x,y,z,qw,qx,qy,qz], ...]}, same file for both endpoints)
  table2: collision-free box_panda, targets = goal poses from tests/mb_problems.json
          (identical file at both endpoints)

Output: appends one CSV row per (leg, batch) to --out:
  label,leg,batch,round,n_targets,median_ms,mean_ms,min_ms,p95_ms
Timing = per-call wall time of generate_solutions across targets in the round.
"""
import argparse, csv, json, statistics, sys, time
from pathlib import Path

import hjcdik
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))
from query_results import best_candidate, pose_errors, query_record, append_record

COLUMNS = ["label", "leg", "batch", "round", "n_targets", "median_ms", "mean_ms",
           "min_ms", "p95_ms", "solved", "empty", "returned", "refine_fp64",
           "correctness_only", "header_sha"]


def load_shared_targets(path):
    D = json.load(open(path))
    return [list(map(float, t)) for t in D["targets"]]


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


def time_leg(targets, batch, num_solutions, collision_free, pj_text, set_name, warmup,
             precision=-1, correctness_only=False, configs_out=None, label="hjcdik"):
    def solve(i, t):
        return hjcdik.generate_solutions(t, batch_size=batch, num_solutions=num_solutions,
            collision_free=collision_free, problems_json_text=pj_text, problem_set_name=set_name,
            problem_idx=i if collision_free else 0, refine_fp64=precision, write_stats=False,
            collision_mode="hard")
    for i, t in enumerate(targets[:warmup]):
        solve(i, t)
    times = []
    solved = empty = returned = 0
    problems = json.loads(pj_text)["problems"][set_name] if collision_free else None
    for i, t in enumerate(targets):
        t0 = None if correctness_only else time.perf_counter()
        result = solve(i, t)
        if t0 is not None:
            times.append((time.perf_counter() - t0) * 1e3)
        if configs_out:
            append_record(configs_out, query_record(result, solver=label, problem_set=set_name,
                problem_idx=i, batch=batch, target=t, ee_target="panda_hand_joint",
                elapsed_ms=None if correctness_only else times[-1]))
        returned += result["count"]
        best = best_candidate(result)
        if best is None:
            empty += 1
            continue
        good = False
        for j, q in enumerate(result["joint_config"]):
            pe, oe = pose_errors(q, t, "panda_hand_joint")
            if abs(pe - result["pos_errors"][j]) > .001 or abs(oe - result["ori_errors"][j]) > 1e-5:
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
    info = hjcdik.build_info()
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
