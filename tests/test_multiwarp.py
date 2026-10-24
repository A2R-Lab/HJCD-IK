"""Multi-warp LM-refine correctness.

The LM refine packs W independent candidates per block (one per warp), selected by the env knob
`HJCD_LM_WARPS` (read per-call). Each warp's candidate is independent of W. Multiple outputs
disable LM early-stop, but coarse search still has inter-block early-stop, so comparisons below
check pose quality within tolerance rather than promising identical candidate sets. Also guards
the warp-vs-block sync trap and the partial-last-block (`gp>=B`) early-return.

Requires a CUDA GPU + built `hjcdik`; skips cleanly otherwise. Correctness-only (no timing) — safe to run
under GPU contention.
"""
import os
from contextlib import contextmanager

import pytest

np = pytest.importorskip("numpy")
hjcdik = pytest.importorskip("hjcdik")


@pytest.fixture(scope="module")
def solver():
    return hjcdik.Solver()


@contextmanager
def warps(W):
    """Set HJCD_LM_WARPS for the block, restoring the prior value (env is read per generate call)."""
    prev = os.environ.get("HJCD_LM_WARPS")
    os.environ["HJCD_LM_WARPS"] = str(W)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("HJCD_LM_WARPS", None)
        else:
            os.environ["HJCD_LM_WARPS"] = prev


def _solve_errs(solver, targets, W, num_solutions=4, batch_size=2000):
    """Per-target best (pos_mm, ori_rad) at a given W."""
    with warps(W):
        out = []
        for t in targets:
            r = solver.solve(t, batch_size=batch_size, num_solutions=num_solutions, collision=None)[0]
            out.append((float(np.min(r.pos_err_mm)) if r.count else np.inf,
                        float(np.min(r.ori_err_rad)) if r.count else np.inf,
                        int(r.count)))
    return out


def _best_pos(solver, t, **kwargs):
    r = solver.solve(t, collision=None, **kwargs)[0]
    return float(np.min(r.pos_err_mm)) if r.count else np.inf


@pytest.mark.parametrize("W", [2, 4, 8])
def test_multiwarp_matches_w1(solver, W):
    """W in {2,4,8} must match W=1 pose quality with LM early-stop disabled."""
    targets = solver.sample_targets(8, seed=0)
    ref = _solve_errs(solver, targets, W=1)
    cur = _solve_errs(solver, targets, W=W)
    for (p1, o1, c1), (p2, o2, c2) in zip(ref, cur):
        assert c1 == c2, f"count mismatch W={W}: {c1} vs {c2}"
        assert abs(p1 - p2) < 1e-3, f"pos err diff W={W}: {abs(p1-p2):.2e} mm"
        assert abs(o1 - o2) < 1e-4, f"ori err diff W={W}: {abs(o1-o2):.2e} rad"


def test_multiwarp_high_w_opt_in_smem(solver):
    """A high requested W may require opt-in shared memory or a register-pressure downshift. It must still match."""
    targets = solver.sample_targets(6, seed=2)
    ref = _solve_errs(solver, targets, W=1)
    cur = _solve_errs(solver, targets, W=16)
    for (p1, _, _), (p2, _, _) in zip(ref, cur):
        assert abs(p1 - p2) < 1e-3, f"W=16 pos err diff: {abs(p1-p2):.2e} mm"


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
def test_partial_last_block_no_crash(solver, precision):
    """The 16-repeat schedule is divisible by 8, so W=8 never covered a partial block.
    W=3 leaves idle warps in the final block for this batch; check both compute types."""
    targets = solver.sample_targets(4, seed=3)
    with warps(3):
        for t in targets:
            r = solver.solve(t, batch_size=37, num_solutions=4, precision=precision, collision=None)[0]
            assert r.count > 0
            assert float(np.min(r.pos_err_mm)) < 1.0  # sub-mm


def test_run_to_run_stability(solver):
    """The solver is NOT bit-deterministic by design: the parallel `stop_on_first` atomic race + the
    top-K score sort make run-to-run results vary at the ~um scale (measured 8-36 um across configs). This
    is a STABILITY guard (catches gross uninitialized-shared / RNG regressions), not a bit-equality check."""
    STABILITY_MM = 0.05   # > observed ~0.036mm worst-case run-to-run; << any real IK tolerance
    targets = solver.sample_targets(4, seed=7)
    for t in targets:
        vals = [_best_pos(solver, t, batch_size=2000, num_solutions=4) for _ in range(4)]
        assert max(vals) - min(vals) < STABILITY_MM, f"run-to-run spread {max(vals)-min(vals):.3e} mm too large"


def test_default_config_solves(solver):
    """With no env override (default W=1) a clean run solves sub-10um. NOTE: default is W=1 — W>1 is an
    opt-in for low-SM devices and is SLOWER on big GPUs (see multiwarp_timing_result.md)."""
    os.environ.pop("HJCD_LM_WARPS", None)
    targets = solver.sample_targets(6, seed=4)
    errs = [_best_pos(solver, t, batch_size=2000, num_solutions=4) for t in targets]
    assert max(errs) < 0.01, f"default-config worst pos err {max(errs):.3e} mm exceeds 10um"
