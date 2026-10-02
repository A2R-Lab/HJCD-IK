#!/usr/bin/env python
"""Neutral A/B timing driver for comparing two HJCD-IK builds (e.g. main vs a branch).

Runs against WHICHEVER hjcdik the invoking interpreter resolves — invoke it once
with endpoint A's venv and once with endpoint B's venv, alternating rounds
(A/B/A/B) to cancel boost-clock drift (sequential blocks drift ~9% on the 5090).
It deliberately uses ONLY the positional API surface common to old and new builds:
generate_solutions(target, batch_size, num_solutions, collision_free,
problems_json_text, problem_set_name, problem_idx, write_stats).

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


def load_shared_targets(path):
    D = json.load(open(path))
    return [list(map(float, t)) for t in D["targets"]]


def goal_pose_targets(problems_json_path, set_name):
    D = json.load(open(problems_json_path))["problems"][set_name]
    out = []
    for inst in D:
        gp = inst.get("goal_pose")
        if not gp:
            continue
        pos = gp.get("position_xyz") or gp.get("position")
        quat = gp.get("quaternion_wxyz") or gp.get("quat_wxyz")  # [w,x,y,z]
        out.append([pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]])
    if not out:
        raise RuntimeError(f"no goal_pose targets in set {set_name}")
    return out


def time_leg(targets, batch, num_solutions, collision_free, pj_text, set_name, warmup):
    for t in targets[:warmup]:
        hjcdik.generate_solutions(t, batch, num_solutions, collision_free,
                                  pj_text, set_name, 0, False)
    times = []
    for i, t in enumerate(targets):
        t0 = time.perf_counter()
        hjcdik.generate_solutions(t, batch, num_solutions, collision_free,
                                  pj_text, set_name, i if collision_free else 0, False)
        times.append((time.perf_counter() - t0) * 1e3)
    return times


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
    args = ap.parse_args()

    # Provenance: make it impossible to silently time the wrong endpoint.
    print(f"[{args.label}] hjcdik = {hjcdik.__file__}")

    batches = [int(b) for b in args.batches.split(",")]
    if args.leg == "table1":
        targets = load_shared_targets(args.targets_json)
        collision_free, pj_text, set_name = False, "", ""
    else:
        targets = goal_pose_targets(args.problems_json, args.problem_set)
        collision_free = True
        pj_text = open(args.problems_json).read()
        set_name = args.problem_set

    out = Path(args.out)
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["label", "leg", "batch", "round", "n_targets",
                        "median_ms", "mean_ms", "min_ms", "p95_ms"])
        for B in batches:
            ts = time_leg(targets, B, args.num_solutions, collision_free,
                          pj_text, set_name, args.warmup)
            ts_sorted = sorted(ts)
            p95 = ts_sorted[max(0, int(0.95 * len(ts_sorted)) - 1)]
            w.writerow([args.label, args.leg, B, args.round, len(ts),
                        f"{statistics.median(ts):.4f}", f"{statistics.fmean(ts):.4f}",
                        f"{min(ts):.4f}", f"{p95:.4f}"])
            print(f"[{args.label} r{args.round}] {args.leg} B={B}: "
                  f"median {statistics.median(ts):.3f} ms over {len(ts)} targets")


if __name__ == "__main__":
    main()
