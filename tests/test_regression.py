"""Regression tests for HJCD-IK.

Asserts the solver does not regress on **solved-rate** and **position/orientation error** versus a committed
baseline. The baseline is captured from the current `main` kernel (see `scripts/bench/capture_baseline.py`) and
stored in `tests/baseline_metrics.json` — we assert against *recorded* numbers, not guessed absolute
thresholds.

Requires a CUDA GPU + a built `hjcdik` extension; skips cleanly otherwise.
"""
import json
import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
hjcdik = pytest.importorskip("hjcdik")

HERE = Path(__file__).parent
BASELINE_PATH = HERE / "baseline_metrics.json"

# Tolerance band around the baseline (relative for errors, absolute for solved-rate).
SOLVED_RATE_SLACK = 0.02      # allow 2 percentage points worse
ERROR_REL_SLACK = 0.10        # allow 10% worse mean error
# Absolute noise floors: both kernels converge to ~sub-micron / ~1e-7 rad, where the relative band is
# meaningless (fp-accumulation order differs between the hand-written X_warp baseline and stock GRiD FK).
# These floors are far tighter than any real IK tolerance, so true regressions still fail.
POS_ERR_ABS_FLOOR_MM = 0.01   # 10 microns
ORI_ERR_ABS_FLOOR_RAD = 1e-4


def _run_suite(num_targets=64, seed=0, batch_size=2000, num_solutions=1):
    """Solve a fixed set of sampled targets; return aggregate metrics."""
    targets = hjcdik.sample_targets(num_targets=num_targets, seed=seed)
    solved = 0
    pos_errs, ori_errs = [], []
    for t in targets:
        out = hjcdik.generate_solutions(t, batch_size=batch_size, num_solutions=num_solutions)
        if out["count"] > 0:
            solved += 1
            pos_errs.append(float(np.min(out["pos_errors"])))
            ori_errs.append(float(np.min(out["ori_errors"])))
    return {
        "num_targets": num_targets,
        "solved_rate": solved / num_targets,
        "mean_pos_err": float(np.mean(pos_errs)) if pos_errs else math.inf,
        "mean_ori_err": float(np.mean(ori_errs)) if ori_errs else math.inf,
    }


@pytest.mark.skipif(not BASELINE_PATH.exists(),
                    reason="baseline_metrics.json not captured yet (run scripts/bench/capture_baseline.py on main)")
def test_no_regression_vs_baseline():
    baseline = json.loads(BASELINE_PATH.read_text())["sampled_unconstrained"]
    current = _run_suite(num_targets=baseline["num_targets"], seed=0)

    assert current["solved_rate"] >= baseline["solved_rate"] - SOLVED_RATE_SLACK, (
        f"solved-rate regressed: {current['solved_rate']:.3f} < {baseline['solved_rate']:.3f}")
    pos_limit = max(baseline["mean_pos_err"] * (1 + ERROR_REL_SLACK), POS_ERR_ABS_FLOOR_MM)
    assert current["mean_pos_err"] <= pos_limit, (
        f"position error regressed: {current['mean_pos_err']:.5f} > {pos_limit:.5f} mm")
    ori_limit = max(baseline["mean_ori_err"] * (1 + ERROR_REL_SLACK), ORI_ERR_ABS_FLOOR_RAD)
    assert current["mean_ori_err"] <= ori_limit, (
        f"orientation error regressed: {current['mean_ori_err']:.6f} > {ori_limit:.6f} rad")


def test_solver_runs_and_returns_solutions():
    """Smoke test: the solver returns at least some solutions for sampled targets."""
    m = _run_suite(num_targets=8, seed=1)
    assert m["solved_rate"] > 0.0


# TODO: add a collision-free regression over tests/mb_problems.json / wall_problems.json once the problem-set
# schema and the collision_free=True path are wired into the baseline capture.
