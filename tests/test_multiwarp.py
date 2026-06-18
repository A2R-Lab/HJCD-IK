"""Multi-warp LM-refine correctness.

The LM refine packs W independent candidates per block (one per warp), selected by the env knob
`HJCD_LM_WARPS` (read per-call). Each warp's candidate is independent of W, so with early-stop OFF
(num_solutions>=2) the per-candidate outputs must match the single-warp (W=1) baseline. Also guards the
warp-vs-block sync trap (the #1 bug class) and the partial-last-block (`gp>=B`) early-return.

Requires a CUDA GPU + built `hjcdik`; skips cleanly otherwise. Correctness-only (no timing) — safe to run
under GPU contention.
"""
import os
from contextlib import contextmanager

import pytest

np = pytest.importorskip("numpy")
hjcdik = pytest.importorskip("hjcdik")


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


def _solve_errs(targets, W, num_solutions=4, batch_size=2000):
    """Per-target best (pos_mm, ori_rad) at a given W."""
    with warps(W):
        out = []
        for t in targets:
            r = hjcdik.generate_solutions(t, batch_size=batch_size, num_solutions=num_solutions)
            pe = np.array(r["pos_errors"], dtype=float)
            oe = np.array(r["ori_errors"], dtype=float)
            out.append((float(np.min(pe)) if pe.size else np.inf,
                        float(np.min(oe)) if oe.size else np.inf,
                        int(r["count"])))
    return out


@pytest.mark.parametrize("W", [2, 4, 8])
def test_multiwarp_matches_w1(W):
    """W in {2,4,8} must match W=1 to fp-noise (num_solutions=4 => early-stop off => deterministic)."""
    targets = hjcdik.sample_targets(num_targets=8, seed=0)
    ref = _solve_errs(targets, W=1)
    cur = _solve_errs(targets, W=W)
    for (p1, o1, c1), (p2, o2, c2) in zip(ref, cur):
        assert c1 == c2, f"count mismatch W={W}: {c1} vs {c2}"
        assert abs(p1 - p2) < 1e-3, f"pos err diff W={W}: {abs(p1-p2):.2e} mm"
        assert abs(o1 - o2) < 1e-4, f"ori err diff W={W}: {abs(o1-o2):.2e} rad"


def test_multiwarp_high_w_opt_in_smem():
    """W=16 exceeds 48KB/block (fp64) -> opt-in dynamic shared via cudaFuncSetAttribute. Must still match."""
    targets = hjcdik.sample_targets(num_targets=6, seed=2)
    ref = _solve_errs(targets, W=1)
    cur = _solve_errs(targets, W=16)
    for (p1, _, _), (p2, _, _) in zip(ref, cur):
        assert abs(p1 - p2) < 1e-3, f"W=16 pos err diff: {abs(p1-p2):.2e} mm"


def test_partial_last_block_no_crash():
    """Krep not a multiple of W => the last block has idle warps that must early-return cleanly (gp>=B).
    A tiny batch makes Krep small and not 8-divisible; assert it still runs and solves."""
    targets = hjcdik.sample_targets(num_targets=4, seed=3)
    with warps(8):
        for t in targets:
            r = hjcdik.generate_solutions(t, batch_size=37, num_solutions=4)
            assert r["count"] > 0
            assert float(np.min(np.array(r["pos_errors"], dtype=float))) < 1.0  # sub-mm


def test_run_to_run_stability():
    """The solver is NOT bit-deterministic by design: the parallel `stop_on_first` atomic race + the
    top-K score sort make run-to-run results vary at the ~um scale (measured 8-36 um across configs). This
    is a STABILITY guard (catches gross uninitialized-shared / RNG regressions), not a bit-equality check."""
    STABILITY_MM = 0.05   # > observed ~0.036mm worst-case run-to-run; << any real IK tolerance
    targets = hjcdik.sample_targets(num_targets=4, seed=7)
    for t in targets:
        vals = [float(np.min(np.array(hjcdik.generate_solutions(t, batch_size=2000, num_solutions=4)["pos_errors"], dtype=float)))
                for _ in range(4)]
        assert max(vals) - min(vals) < STABILITY_MM, f"run-to-run spread {max(vals)-min(vals):.3e} mm too large"


def test_default_config_solves():
    """With no env override (default W=1) a clean run solves sub-10um. NOTE: default is W=1 — W>1 is an
    opt-in for low-SM devices and is SLOWER on big GPUs (see multiwarp_timing_result.md)."""
    os.environ.pop("HJCD_LM_WARPS", None)
    targets = hjcdik.sample_targets(num_targets=6, seed=4)
    errs = [float(np.min(np.array(hjcdik.generate_solutions(t, batch_size=2000, num_solutions=4)["pos_errors"], dtype=float)))
            for t in targets]
    assert max(errs) < 0.01, f"default-config worst pos err {max(errs):.3e} mm exceeds 10um"
