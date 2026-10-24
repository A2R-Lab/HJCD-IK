#!/usr/bin/env python3
"""Capture HJCD-IK baseline metrics for regression testing.

Run this on the build you want to pin (normally `main`), then commit the output to
`tests/baseline_metrics.json`. `tests/test_regression.py` imports the suite functions below and
asserts later builds do not regress against the recorded numbers (with a small slack band).

Two suites:
  sampled_unconstrained  64 sampled open-world targets, B=2000, one solution each.
  collision_free_mb      the first PROBLEMS_PER_SET goal poses of every MotionBenchMaker set in
                         tests/mb_problems.json, B=2000, one solution, collision="hard";
                         every returned configuration is also checked by the independent NumPy
                         sphere oracle (benchmark/panda_collision.py, URDF-derived "hjcd" model).

Candidate identity is not run-to-run stable (the coarse early-stop races across blocks), so the
metrics are aggregates over many targets, not per-target values.

Requires a GPU + a built `hjcdik` extension.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import hjcdik

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests" / "baseline_metrics.json"
MB_PATH = ROOT / "tests" / "mb_problems.json"
PROBLEMS_PER_SET = 8
# "Accurate" = the returned best candidate actually reaches the goal. Strict (hard) filtering may
# return a far-off collision-free candidate instead of nothing, so solved_rate alone says little.
ACCURATE_POS_MM = 1.0
ACCURATE_ORI_RAD = 1e-3

sys.path.insert(0, str(ROOT / "benchmark"))
from panda_collision import mb_instance_to_world_dict, panda_config_collision_free  # noqa: E402
from query_results import panda_hand_target


def _aggregate(num_targets, solved, pos_errs, ori_errs, **extra):
    return {
        "num_targets": num_targets,
        "solved_rate": solved / num_targets,
        "mean_pos_err": float(np.mean(pos_errs)) if pos_errs else math.inf,
        "mean_ori_err": float(np.mean(ori_errs)) if ori_errs else math.inf,
        **extra,
    }


def run_suite(num_targets=64, seed=0, batch_size=2000, num_solutions=1):
    """Open-world: solve sampled targets; return aggregate solved-rate and best-candidate errors."""
    solver = hjcdik.Solver()
    targets = solver.sample_targets(num_targets, seed=seed)
    solved, pos_errs, ori_errs = 0, [], []
    for t in targets:
        out = solver.solve(t, batch_size=batch_size, num_solutions=num_solutions, collision=None)[0]
        if out.count > 0:
            solved += 1
            pos_errs.append(float(np.min(out.pos_err_mm)))
            ori_errs.append(float(np.min(out.ori_err_rad)))
    return _aggregate(num_targets, solved, pos_errs, ori_errs, batch_size=batch_size)


def run_collision_suite(problems_per_set=PROBLEMS_PER_SET, batch_size=2000, num_solutions=1,
                        collision_mode="hard"):
    """Collision-free over tests/mb_problems.json; also counts oracle-verified collision-free returns."""
    problems = json.loads(MB_PATH.read_text())["problems"]
    solver = hjcdik.Solver()
    solved, pos_errs, ori_errs = 0, [], []
    num_targets, returned, oracle_free = 0, 0, 0
    for set_name in sorted(problems):
        for i in range(min(problems_per_set, len(problems[set_name]))):
            inst = problems[set_name][i]
            gp = inst["goal_pose"]
            target = list(gp["position_xyz"]) + list(gp["quaternion_wxyz"])
            target = panda_hand_target(target, solver.info.ee_frame)
            solver.set_world(**hjcdik.worlds.from_mbm_problem(inst))
            out = solver.solve(target, batch_size=batch_size, num_solutions=num_solutions,
                               collision=collision_mode)[0]
            num_targets += 1
            world = mb_instance_to_world_dict(inst)
            for q in np.asarray(out.q):
                returned += 1
                oracle_free += bool(panda_config_collision_free(q, world, model="hjcd"))
            if out.count > 0:
                solved += 1
                best = int(np.argmin(out.pos_err_mm))
                pe, oe = float(out.pos_err_mm[best]), float(out.ori_err_rad[best])
                if pe < ACCURATE_POS_MM and oe < ACCURATE_ORI_RAD:   # errors are averaged over accurate solves only
                    pos_errs.append(pe)
                    ori_errs.append(oe)
    return _aggregate(num_targets, solved, pos_errs, ori_errs, batch_size=batch_size,
                      problems_per_set=problems_per_set, collision_mode=collision_mode,
                      accurate_rate=len(pos_errs) / num_targets, returned=returned,
                      oracle_collision_free_rate=(oracle_free / returned) if returned else math.nan)


def main():
    data = {
        "sampled_unconstrained": run_suite(num_targets=64, seed=0),
        "collision_free_mb": run_collision_suite(),
    }
    OUT.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {OUT}:")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
