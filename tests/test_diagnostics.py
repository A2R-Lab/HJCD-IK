"""Phase 3C: trace-derived LM diagnostics.

The per-iteration TRACE is the authoritative source. Every public counter is derived from it:

    lm_iterations      = number of rows whose EXPLICIT valid flag (column 0) is set
    lm_trials          = cumulative trial count on the last valid row
    accepted_lm_steps  = cumulative accepted count on the last valid row
    rejected_lm_steps  = lm_iterations - accepted_lm_steps
    line_searches      = cumulative line-search count on the last valid row

Row validity is column 0 and is NEVER inferred from cost, lambda or the iteration index -- a
converged row legitimately has cost == 0, and it == 0 is a real first row.

Diagnostics are opt-in. With diagnostics=False no trace buffer is allocated, no trace store executes,
and the solve path is unchanged (test_diagnostics_do_not_change_the_solution pins that).
"""
import os
from pathlib import Path

import numpy as np
import pytest

import hjcdik

REPO = Path(__file__).resolve().parents[1]
N = hjcdik.num_joints()
K = hjcdik.num_targets()
LIMITS = hjcdik.joint_limits()

VALID, IT, TRIALS, ACC_THIS, ACC_CUM, COST, MAXP, MAXO, LAM, LSEARCH = range(10)


def _sample_q(rng, margin=0.15):
    lo, hi = LIMITS[:, 0], LIMITS[:, 1]
    return rng.uniform(lo + margin * (hi - lo), hi - margin * (hi - lo))


def _quat_from_R(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = np.array([0.25*s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s])
    else:
        i = int(np.argmax([R[0,0], R[1,1], R[2,2]]))
        if i == 0:
            s = np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2
            q = np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
        elif i == 1:
            s = np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2
            q = np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
        else:
            s = np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2
            q = np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])
    return q / np.linalg.norm(q)


def _targets_at(Q):
    T = hjcdik.target_transforms(Q)
    p = T[:, :, :3, 3]
    quat = np.stack([[_quat_from_R(T[b, k, :3, :3]) for k in range(K)] for b in range(len(Q))])
    return p, quat


def _problem(rng, B, seed_scale):
    lo, hi = LIMITS[:, 0], LIMITS[:, 1]
    Q = np.stack([_sample_q(rng) for _ in range(B)])
    p, quat = _targets_at(Q)
    seeds = np.clip(Q + rng.normal(scale=seed_scale, size=Q.shape), lo, hi)
    return seeds, p, quat, Q


def _run(seeds, p, quat, **kw):
    # Counter plumbing was pinned against the fp64 convergence profile; at the fp32
    # default these particular seeds all hit the iteration cap, which would make
    # "counts differ across the batch" vacuous. fp32 counter distinctness is covered
    # by test_array_contiguity.py, which uses a mixed nearby/random fixture.
    kw.setdefault("precision", "float64")
    kw.setdefault("diagnostics", True)
    kw.setdefault("return_trace", True)
    kw.setdefault("position_tol", 1e-8)
    kw.setdefault("orientation_tol", 1e-8)
    return hjcdik.refine(seeds, p, quat, **kw)


# --- the trace is authoritative; every counter must agree with it -------------------------------

def test_counters_match_the_trace():
    rng = np.random.default_rng(4)
    seeds, p, quat, _ = _problem(rng, 8, 0.25)
    for mi in (5, 10, 25, 40):
        o = _run(seeds, p, quat, max_iters=mi)
        tr = o["trace"]
        valid = tr[:, :, VALID] != 0
        assert o["lm_iterations"].tolist() == valid.sum(axis=1).tolist()
        for b in range(len(seeds)):
            n = int(valid[b].sum())
            if n == 0:
                assert o["lm_trials"][b] == 0 and o["line_searches"][b] == 0
                continue
            last = tr[b, n - 1]
            assert o["lm_trials"][b] == int(last[TRIALS])
            assert o["accepted_lm_steps"][b] == int(last[ACC_CUM])
            assert o["line_searches"][b] == int(last[LSEARCH])
    print(f"\n  counters agree with the trace for max_iters in (5, 10, 25, 40)")


def test_accepted_plus_rejected_equals_iterations():
    rng = np.random.default_rng(11)
    seeds, p, quat, _ = _problem(rng, 12, 0.35)
    o = _run(seeds, p, quat, max_iters=30)
    tot = o["accepted_lm_steps"] + o["rejected_lm_steps"]
    np.testing.assert_array_equal(tot, o["lm_iterations"])


def test_trace_rows_are_marked_valid_explicitly():
    """Validity is column 0. It must not be inferable from cost/lambda/it -- rows past the end are
    all-zero, and a converged row can itself have cost 0."""
    rng = np.random.default_rng(3)
    seeds, p, quat, _ = _problem(rng, 4, 0.2)
    o = _run(seeds, p, quat, max_iters=30)
    tr = o["trace"]
    for b in range(4):
        n = int(o["lm_iterations"][b])
        assert np.all(tr[b, :n, VALID] == 1.0), "a completed iteration is not marked valid"
        assert np.all(tr[b, n:, VALID] == 0.0), "an unwritten row is marked valid"
        assert np.all(tr[b, :n, IT] == np.arange(n)), "iteration indices are not contiguous"


# --- the specific cases -------------------------------------------------------------------------

def test_zero_iterations_on_immediate_convergence():
    """A seed that already satisfies the tolerance does NO linearization: all counters are 0."""
    rng = np.random.default_rng(21)
    Q = np.stack([_sample_q(rng) for _ in range(4)])
    p, quat = _targets_at(Q)
    o = _run(Q, p, quat, position_tol=1e-3, orientation_tol=1e-2, max_iters=20)   # already there
    np.testing.assert_array_equal(o["lm_iterations"], np.zeros(4, dtype=int))
    np.testing.assert_array_equal(o["lm_trials"], np.zeros(4, dtype=int))
    np.testing.assert_array_equal(o["line_searches"], np.zeros(4, dtype=int))
    np.testing.assert_array_equal(o["accepted_lm_steps"], np.zeros(4, dtype=int))
    np.testing.assert_array_equal(o["rejected_lm_steps"], np.zeros(4, dtype=int))
    assert np.all(o["success"])
    assert np.all(o["trace"][:, :, VALID] == 0.0), "no trace row should have been written"


def test_exactly_one_iteration():
    rng = np.random.default_rng(31)
    seeds, p, quat, _ = _problem(rng, 6, 0.15)
    o = _run(seeds, p, quat, max_iters=1)
    np.testing.assert_array_equal(o["lm_iterations"], np.full(6, 1))
    np.testing.assert_array_equal(o["lm_trials"], np.full(6, 1))     # one damped system per iter
    assert np.all(o["line_searches"] >= 1)
    assert np.all(o["trace"][:, 0, VALID] == 1.0)
    assert np.all(o["trace"][:, 1:, VALID] == 0.0)


def test_counts_increase_with_max_iters():
    """Counts must respond to max_iters -- the symptom that opened Phase 3C was that they did not.

    A strict increase is only required while some problem is still CAPPED by max_iters. Once every
    problem converges on its own, raising the cap correctly changes nothing.
    """
    rng = np.random.default_rng(41)
    seeds, p, quat, _ = _problem(rng, 8, 0.3)
    prev = prev_mi = prev_ok = None
    saw_increase = False
    for mi in (2, 5, 10, 20, 40):
        o = _run(seeds, p, quat, max_iters=mi)
        it, ok = o["lm_iterations"], o["success"]
        assert np.all(it <= mi), "reported more iterations than max_iters"
        if prev is not None:
            assert np.all(it >= prev), "raising max_iters reduced the iteration count"
            # A problem was truly CAPPED only if it ran to the cap AND had not converged. A problem
            # can legitimately converge on exactly the last allowed iteration, which looks capped but
            # is not -- G1 K=1 does this.
            capped = (prev == prev_mi) & (~prev_ok)
            if np.any(capped):
                assert it[capped].sum() > prev[capped].sum(), (
                    f"a problem was capped at max_iters={prev_mi} and had NOT converged, but "
                    f"raising the cap to {mi} changed nothing: {prev.tolist()} -> {it.tolist()}")
                saw_increase = True
        prev, prev_mi, prev_ok = it, mi, ok
    assert saw_increase, "no problem was ever capped-and-unconverged -- the test proved nothing"


def test_different_batch_elements_get_different_counts():
    """Distinct problems must report distinct counts -- the failure mode this phase existed to fix
    was every element reporting problem 0's count."""
    rng = np.random.default_rng(4)
    seeds, p, quat, _ = _problem(rng, 16, 0.25)
    o = _run(seeds, p, quat, max_iters=30)
    it = o["lm_iterations"]
    assert len(np.unique(it)) > 1, f"all batch elements report the same count: {it.tolist()}"
    print(f"\n  distinct iteration counts across the batch: {sorted(set(it.tolist()))}")


# --- opt-in behaviour ---------------------------------------------------------------------------

def test_diagnostics_off_exposes_no_counters():
    rng = np.random.default_rng(51)
    seeds, p, quat, _ = _problem(rng, 4, 0.2)
    o = hjcdik.refine(seeds, p, quat, max_iters=20)          # default: diagnostics=False
    for key in ("lm_iterations", "lm_trials", "line_searches", "accepted_lm_steps",
                "rejected_lm_steps", "iterations", "trace"):
        assert key not in o, f"'{key}' must be absent when diagnostics are off (no trace allocated)"


def test_diagnostics_do_not_change_the_solution():
    """Turning diagnostics on must not perturb the solve by one bit."""
    rng = np.random.default_rng(61)
    seeds, p, quat, _ = _problem(rng, 8, 0.3)
    off = hjcdik.refine(seeds, p, quat, max_iters=30)
    on = hjcdik.refine(seeds, p, quat, max_iters=30, diagnostics=True)
    np.testing.assert_array_equal(off["joint_config"], on["joint_config"])
    np.testing.assert_array_equal(off["cost"], on["cost"])
    np.testing.assert_array_equal(off["position_errors"], on["position_errors"])
    np.testing.assert_array_equal(off["success"], on["success"])


def test_trace_requires_diagnostics():
    rng = np.random.default_rng(71)
    seeds, p, quat, _ = _problem(rng, 2, 0.2)
    with pytest.raises(ValueError, match="requires diagnostics"):
        hjcdik.refine(seeds, p, quat, return_trace=True)


def test_diagnostics_without_trace_still_gives_counters():
    rng = np.random.default_rng(81)
    seeds, p, quat, _ = _problem(rng, 4, 0.25)
    o = hjcdik.refine(seeds, p, quat, max_iters=20, diagnostics=True)   # return_trace=False
    assert "trace" not in o
    assert "lm_iterations" in o
    assert np.all(o["lm_iterations"] > 0)
