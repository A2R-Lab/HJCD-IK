"""LM-refine precision knob: fp32/fp64 accuracy + auto-select by num_solutions.

`refine_fp64` is tri-state: -1=auto (default; fp64 for num_solutions==1, fp32 for >=2), 1=force fp64,
0=force fp32. fp32 refine is 5-7x faster for num_solutions>=2 (throughput-bound) at sub-micron accuracy;
this guards that the fp32 path and the auto-select stay correct.

Requires a CUDA GPU + built `hjcdik`; skips cleanly otherwise. Correctness-only.
"""
import pytest

np = pytest.importorskip("numpy")
hjcdik = pytest.importorskip("hjcdik")

POS_FLOOR_MM = 0.01   # 10 microns: far looser than achieved (~sub-micron), tight vs any real IK tol


def _best(t, num_solutions, refine_fp64, batch_size=2000):
    r = hjcdik.generate_solutions(t, batch_size=batch_size, num_solutions=num_solutions,
                                  refine_fp64=refine_fp64)
    pe = np.array(r["pos_errors"], dtype=float)
    oe = np.array(r["ori_errors"], dtype=float)
    return (float(np.min(pe)) if pe.size else np.inf,
            float(np.min(oe)) if oe.size else np.inf,
            int(r["count"]))


def test_fp32_refine_accuracy():
    """Forced fp32 refine (refine_fp64=0) stays sub-10um for both early-stop modes."""
    targets = hjcdik.sample_targets(num_targets=8, seed=0)
    for ns in (1, 4):
        worst = max(_best(t, ns, refine_fp64=0)[0] for t in targets)
        assert worst < POS_FLOOR_MM, f"fp32 ns={ns} worst pos {worst:.3e} mm exceeds {POS_FLOOR_MM} mm"


def test_fp64_refine_accuracy():
    targets = hjcdik.sample_targets(num_targets=8, seed=0)
    for ns in (1, 4):
        worst = max(_best(t, ns, refine_fp64=1)[0] for t in targets)
        assert worst < POS_FLOOR_MM, f"fp64 ns={ns} worst pos {worst:.3e} mm exceeds {POS_FLOOR_MM} mm"


def test_auto_select_matches_explicit():
    """Default (auto, refine_fp64=-1) must produce results consistent with the precision it auto-picks
    (fp64 at ns=1, fp32 at ns>=2) within fp run-to-run noise. NOTE: fp32 and fp64 are BOTH sub-micron, so
    accuracy cannot prove which precision ran — that (the 5-7x speed) is verified by the timing sweep
    (scripts/perf/time_multiwarp_sweep.py). This guards that auto isn't broken / mis-wired."""
    band_mm, band_rad = 5e-3, 5e-4   # run-to-run fp noise band (same scale as the W-match test)
    targets = hjcdik.sample_targets(num_targets=8, seed=1)
    for t in targets:
        a1 = _best(t, 1, refine_fp64=-1)
        f1 = _best(t, 1, refine_fp64=1)
        assert a1[2] == f1[2] and abs(a1[0] - f1[0]) < band_mm and abs(a1[1] - f1[1]) < band_rad, \
            "auto ns=1 should track forced fp64"
        a4 = _best(t, 4, refine_fp64=-1)
        f4 = _best(t, 4, refine_fp64=0)
        assert a4[2] == f4[2] and abs(a4[0] - f4[0]) < band_mm and abs(a4[1] - f4[1]) < band_rad, \
            "auto ns=4 should track forced fp32"


def test_default_arg_is_auto():
    """No refine_fp64 passed == auto; must solve sub-10um."""
    targets = hjcdik.sample_targets(num_targets=6, seed=2)
    for t in targets:
        r = hjcdik.generate_solutions(t, batch_size=2000, num_solutions=4)
        assert float(np.min(np.array(r["pos_errors"], dtype=float))) < POS_FLOOR_MM
