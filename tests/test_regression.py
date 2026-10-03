"""Regression tests for HJCD-IK.

Asserts the solver does not regress on **solved-rate / accurate-rate** and **position/orientation error**
versus a committed baseline. The baseline is captured from `main` by `scripts/bench/capture_baseline.py`
and stored in `tests/baseline_metrics.json` — we assert against *recorded* numbers, not guessed absolute
thresholds. The suite functions are imported from the capture script so test and baseline run the same code.

Candidate identity is not run-to-run stable (the coarse early-stop races across blocks), so every metric
here is an aggregate over 64 targets with a slack band.

Requires a CUDA GPU + a built `hjcdik` extension; skips cleanly otherwise.
"""
import importlib.util
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
hjcdik = pytest.importorskip("hjcdik")

HERE = Path(__file__).parent
BASELINE_PATH = HERE / "baseline_metrics.json"

_spec = importlib.util.spec_from_file_location(
    "hjcd_capture_baseline", HERE.parent / "scripts" / "bench" / "capture_baseline.py")
capture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(capture)

# Tolerance band around the baseline (relative for errors, absolute for rates).
RATE_SLACK = 0.02             # allow 2 percentage points worse (~1 target in 64)
ERROR_REL_SLACK = 0.10        # allow 10% worse mean error
# Absolute noise floors: both kernels converge to ~sub-micron / ~1e-7 rad, where the relative band is
# meaningless (fp-accumulation order differs between builds). These floors are far tighter than any
# real IK tolerance, so true regressions still fail.
POS_ERR_ABS_FLOOR_MM = 0.01   # 10 microns
ORI_ERR_ABS_FLOOR_RAD = 1e-4


def _assert_no_error_regression(current, baseline):
    pos_limit = max(baseline["mean_pos_err"] * (1 + ERROR_REL_SLACK), POS_ERR_ABS_FLOOR_MM)
    assert current["mean_pos_err"] <= pos_limit, (
        f"position error regressed: {current['mean_pos_err']:.5f} > {pos_limit:.5f} mm")
    ori_limit = max(baseline["mean_ori_err"] * (1 + ERROR_REL_SLACK), ORI_ERR_ABS_FLOOR_RAD)
    assert current["mean_ori_err"] <= ori_limit, (
        f"orientation error regressed: {current['mean_ori_err']:.6f} > {ori_limit:.6f} rad")


@pytest.mark.skipif(not BASELINE_PATH.exists(),
                    reason="baseline_metrics.json not captured yet (run scripts/bench/capture_baseline.py on main)")
def test_no_regression_vs_baseline():
    baseline = json.loads(BASELINE_PATH.read_text())["sampled_unconstrained"]
    current = capture.run_suite(num_targets=baseline["num_targets"], seed=0,
                                batch_size=baseline["batch_size"])
    assert current["solved_rate"] >= baseline["solved_rate"] - RATE_SLACK, (
        f"solved-rate regressed: {current['solved_rate']:.3f} < {baseline['solved_rate']:.3f}")
    _assert_no_error_regression(current, baseline)


@pytest.mark.skipif(not BASELINE_PATH.exists(),
                    reason="baseline_metrics.json not captured yet (run scripts/bench/capture_baseline.py on main)")
@pytest.mark.skipif(not capture.MB_PATH.exists(), reason="tests/mb_problems.json missing")
def test_no_collision_free_regression_vs_baseline():
    """Collision-free (hard) over the first problems of every MotionBenchMaker set.

    Guards three things the open-world suite cannot: the accurate-rate (strict filtering may return a
    far-off collision-free candidate instead of nothing, so solved_rate alone is blind), the errors of the
    accurate solves, and that every returned configuration is collision-free by the independent sphere
    oracle (this one has no slack: a single colliding return is a contract break).
    """
    baseline = json.loads(BASELINE_PATH.read_text())["collision_free_mb"]
    current = capture.run_collision_suite(
        problems_per_set=baseline["problems_per_set"], batch_size=baseline["batch_size"],
        collision_mode=baseline["collision_mode"])
    assert current["num_targets"] == baseline["num_targets"]
    assert current["solved_rate"] >= baseline["solved_rate"] - RATE_SLACK, (
        f"collision-free solved-rate regressed: {current['solved_rate']:.3f} < {baseline['solved_rate']:.3f}")
    assert current["accurate_rate"] >= baseline["accurate_rate"] - RATE_SLACK, (
        f"collision-free accurate-rate regressed: {current['accurate_rate']:.3f} < {baseline['accurate_rate']:.3f}")
    assert current["oracle_collision_free_rate"] >= baseline["oracle_collision_free_rate"], (
        f"hard mode returned colliding configurations: oracle-free rate "
        f"{current['oracle_collision_free_rate']:.4f} < {baseline['oracle_collision_free_rate']:.4f}")
    _assert_no_error_regression(current, baseline)


def test_solver_runs_and_returns_solutions():
    """Smoke test: the solver returns at least some solutions for sampled targets."""
    m = capture.run_suite(num_targets=8, seed=1)
    assert m["solved_rate"] > 0.0
