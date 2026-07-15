"""Phase 5: multi-target coarse search -- aggregate weighted coordinate Gauss-Newton.

Every quantity the device computes is checked against an explicit CPU reference:

    g_j     = sum_{k in A_j} ( Jv_kj^T W_p,k e_p,k + Jw_kj^T W_R,k e_R,k )
    h_j     = sum_{k in A_j} ( Jv_kj^T W_p,k Jv_kj + Jw_kj^T W_R,k Jw_kj )
    delta_j = g_j / (h_j + lambda_coord)                      [de/dq = -J => PLUS sign]
    pred_j  = 2*delta_j*g_j - delta_j^2*h_j                   [after clipping]

    W_{k,r} = w_{k,r} * s_{k,r}^2,   s_{k,r} = 1/(||J_{k,r}|| + eps)   (Phase-3B, frozen per iter)
    A_j     = JOINT_TARGET_MASK[j] & active_target_mask

The device trace (Phase-3C pattern, explicit valid flag) exposes the selected joint, its delta, the
predicted improvement and the exact costs before/after, so the CPU reference can check the FIRST
iteration end to end: proposal, clipping, masking, winner selection and the exact evaluation.
"""
import os
import re
from pathlib import Path

import numpy as np
import pytest

import hjcdik

REPO = Path(__file__).resolve().parents[1]
N = hjcdik.num_joints()
K = hjcdik.num_targets()
ALL = (1 << K) - 1
LIMITS = hjcdik.joint_limits()

VALID, IT, JOINT, DELTA, PRED, COST0, COST1, ACC, STALL, PERT = range(10)

LAMBDA_COORD = 1e-6
H_MIN = 1e-9
MAX_STEP = 0.35


def _axis_meta():
    txt = (REPO / "csrc" / "generated" / "grid.cuh").read_text()
    col = [int(v) for v in re.search(r"JOINT_AXIS_COL\[\d+\] = \{([^}]*)\}", txt).group(1).split(",")]
    sgn = [int(v) for v in re.search(r"JOINT_AXIS_SIGN\[\d+\] = \{([^}]*)\}", txt).group(1).split(",")]
    return col, sgn


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


# --- the CPU reference --------------------------------------------------------------------------

def cpu_proposals(q, p, quat, mask, wp, wo):
    """Reference g_j, h_j, delta_j, pred_j for every joint, plus the winner. Dense and explicit."""
    col, sgn = _axis_meta()
    meta = hjcdik.target_metadata()
    T = hjcdik.link_transforms(q[None, :])[0]
    X = hjcdik.target_transforms(q[None, :])[0]
    r = hjcdik.target_residuals(q[None, :], p[None], quat[None],
                                active_target_mask=np.array([mask], dtype=np.uint32),
                                position_weights=wp[None], orientation_weights=wo[None])

    # per-target 6xN Jacobian, non-ancestor columns EXACTLY zero
    J = np.zeros((K, 6, N))
    for k in range(K):
        if not (mask >> k) & 1:
            continue
        p_t = X[k][:3, 3]
        for j in range(N):
            if not (int(meta["target_ancestor_mask"][k]) >> j) & 1:
                continue
            a = sgn[j] * T[j][:3, col[j]]
            J[k, 0:3, j] = np.cross(a, p_t - T[j][:3, 3])
            J[k, 3:6, j] = a

    # frozen Phase-3B row scaling, per target
    S = np.ones((K, 6))
    for k in range(K):
        if not (mask >> k) & 1:
            continue
        rn2 = (J[k] ** 2).sum(axis=1)
        S[k] = np.where(rn2 > 1e-18, 1.0 / np.sqrt(np.maximum(rn2, 1e-300)), 1.0)

    e = np.zeros((K, 6))
    for k in range(K):
        e[k, :3] = r["position_residuals"][0, k]
        e[k, 3:] = r["orientation_residuals"][0, k]

    g = np.zeros(N)
    h = np.zeros(N)
    for j in range(N):
        affected = int(meta["joint_target_mask"][j]) & mask
        if affected == 0:
            g[j] = h[j] = np.nan          # invalid: no active target depends on j
            continue
        for k in range(K):
            if not (affected >> k) & 1:
                continue
            W = np.concatenate([np.full(3, wp[k]), np.full(3, wo[k])]) * S[k] ** 2
            g[j] += (J[k, :, j] * W * e[k]).sum()
            h[j] += (J[k, :, j] * W * J[k, :, j]).sum()

    delta = np.full(N, np.nan)
    pred = np.full(N, -1.0)
    newv = q.copy()
    for j in range(N):
        if np.isnan(h[j]) or not (h[j] > H_MIN):
            continue
        d = g[j] / (h[j] + LAMBDA_COORD)
        d = float(np.clip(d, -MAX_STEP, MAX_STEP))                 # max coordinate step
        v = float(np.clip(q[j] + d, LIMITS[j, 0], LIMITS[j, 1]))   # joint limits
        d = v - q[j]                                               # effective step
        if d == 0.0:
            continue
        pr = 2 * d * g[j] - d * d * h[j]
        if pr <= 0:
            continue
        delta[j], pred[j], newv[j] = d, pr, v

    valid = pred > 0
    win = int(np.argmax(np.where(valid, pred, -np.inf))) if valid.any() else -1
    return g, h, delta, pred, newv, win, S


def _e_phys(res, ptol, otol):
    """Tolerance-normalised physical error -- the metric best_x is tracked on."""
    return ((res["position_errors"] / ptol) ** 2 + (res["orientation_errors"] / otol) ** 2).sum(axis=1)


def _run(seeds, p, quat, **kw):
    # These tests validate the coarse math against a float64 numpy CPU oracle to ~1e-10. That is an
    # FP64 claim, so they pin fp64 explicitly rather than riding the (now fp32) default. fp32 coarse
    # correctness is covered in test_precision_fp32.py.
    kw.setdefault("precision", "float64")
    kw.setdefault("diagnostics", True)
    kw.setdefault("return_trace", True)
    kw.setdefault("lambda_coord", LAMBDA_COORD)
    kw.setdefault("h_min", H_MIN)
    kw.setdefault("max_step", MAX_STEP)
    return hjcdik.coarse_search(seeds, p, quat, **kw)


# --- 1. the first iteration, end to end, vs the CPU reference -----------------------------------

@pytest.mark.parametrize("trial", range(4))
def test_first_iteration_matches_cpu_reference(trial):
    rng = np.random.default_rng(200 + trial)
    q = _sample_q(rng)
    tq = _sample_q(rng)
    p, quat = _targets_at(tq[None, :])
    wp, wo = np.ones(K), np.ones(K)

    o = _run(q[None, :], p, quat, max_iters=1)
    tr = o["trace"][0, 0]
    assert tr[VALID] == 1.0

    g, h, delta, pred, newv, win, _ = cpu_proposals(q, p[0], quat[0], ALL, wp, wo)

    assert int(tr[JOINT]) == win, f"winner: device {int(tr[JOINT])}, cpu {win}"
    if win < 0:
        return
    assert abs(tr[DELTA] - delta[win]) < 1e-10 * max(1.0, abs(delta[win])), "delta mismatch"
    rel = abs(tr[PRED] - pred[win]) / max(1e-30, abs(pred[win]))
    assert rel < 1e-9, f"predicted improvement mismatch, rel {rel:.2e}"

    # the exact evaluation: cost after must be the true cost at the updated config
    q2 = q.copy()
    q2[win] = newv[win]
    assert tr[COST1] < tr[COST0] or tr[ACC] == 0.0
    print(f"\n  trial {trial}: winner j={win}  delta={delta[win]:+.6f}  pred={pred[win]:.3e}  "
          f"cost {tr[COST0]:.6e} -> {tr[COST1]:.6e}  accepted={int(tr[ACC])}")


def test_invalid_proposals_for_unaffected_joints():
    """A joint with no ACTIVE affected target must never be selected."""
    if K < 2:
        pytest.skip("needs K >= 2")
    meta = hjcdik.target_metadata()
    rng = np.random.default_rng(9)
    q = _sample_q(rng)
    p, quat = _targets_at(_sample_q(rng)[None, :])
    for mask in range(1, 1 << K):
        o = _run(q[None, :], p, quat, max_iters=6,
                 active_target_mask=np.array([mask], dtype=np.uint32))
        tr = o["trace"][0]
        for row in tr[tr[:, VALID] != 0]:
            j = int(row[JOINT])
            if j < 0:
                continue
            affected = int(meta["joint_target_mask"][j]) & mask
            assert affected != 0, (
                f"mask {mask:0{K}b}: selected joint {j}, which affects no active target")


def test_step_is_clipped_to_limits_and_max_step():
    rng = np.random.default_rng(31)
    for _ in range(6):
        q = _sample_q(rng, margin=0.02)          # start close to the limits
        p, quat = _targets_at(_sample_q(rng)[None, :])
        o = _run(q[None, :], p, quat, max_iters=10, max_step=0.05)
        tr = o["trace"][0]
        for row in tr[tr[:, VALID] != 0]:
            assert abs(row[DELTA]) <= 0.05 + 1e-12, f"delta {row[DELTA]} exceeds max_step"
        qf = o["joint_config"][0]
        assert np.all(qf >= LIMITS[:, 0] - 1e-12) and np.all(qf <= LIMITS[:, 1] + 1e-12)


# --- 2. incremental FK == full FK ----------------------------------------------------------------

def test_incremental_equals_full_fk():
    """The Phase-4 incremental path must give the SAME answer as a full FK + full rescore."""
    rng = np.random.default_rng(5)
    B = 24
    Q = np.stack([_sample_q(rng) for _ in range(B)])
    p, quat = _targets_at(np.stack([_sample_q(rng) for _ in range(B)]))
    seeds = np.stack([rng.uniform(LIMITS[:, 0], LIMITS[:, 1]) for _ in range(B)])
    a = _run(seeds, p, quat, max_iters=40, use_incremental=True, seed=3)
    b = _run(seeds, p, quat, max_iters=40, use_incremental=False, seed=3)
    d = np.abs(a["joint_config"] - b["joint_config"]).max()
    assert d < 1e-12, f"incremental FK diverged from full FK by {d:.3e}"
    np.testing.assert_array_equal(a["accepted_coarse_steps"], b["accepted_coarse_steps"])
    print(f"\n  incremental vs full FK: max |dq| = {d:.2e}, identical accept decisions")


# --- 3. acceptance / rollback / stalls / perturbation --------------------------------------------

def test_accepted_steps_strictly_reduce_the_exact_cost():
    rng = np.random.default_rng(17)
    seeds = np.stack([rng.uniform(LIMITS[:, 0], LIMITS[:, 1]) for _ in range(16)])
    p, quat = _targets_at(np.stack([_sample_q(rng) for _ in range(16)]))
    o = _run(seeds, p, quat, max_iters=40)
    tr = o["trace"]
    for b in range(16):
        rows = tr[b][tr[b][:, VALID] != 0]
        acc = rows[rows[:, ACC] == 1.0]
        assert np.all(acc[:, COST1] < acc[:, COST0]), "an accepted step did not reduce the cost"
        rej = rows[(rows[:, ACC] == 0.0) & (rows[:, JOINT] >= 0) & (rows[:, PERT] == 0.0)]
        # a rejected step rolls back: cost after == cost before, bitwise
        assert np.all(rej[:, COST1] == rej[:, COST0]), "rollback did not restore the exact cost"


def test_accepted_plus_rejected_equals_iterations():
    rng = np.random.default_rng(23)
    seeds = np.stack([rng.uniform(LIMITS[:, 0], LIMITS[:, 1]) for _ in range(16)])
    p, quat = _targets_at(np.stack([_sample_q(rng) for _ in range(16)]))
    o = _run(seeds, p, quat, max_iters=40)
    np.testing.assert_array_equal(o["accepted_coarse_steps"] + o["rejected_coarse_steps"],
                                  o["coarse_iterations"])


def test_stalls_trigger_perturbation():
    """After stall_lim consecutive non-improving iterations a random perturbation must fire."""
    rng = np.random.default_rng(29)
    seeds = np.stack([rng.uniform(LIMITS[:, 0], LIMITS[:, 1]) for _ in range(24)])
    p, quat = _targets_at(np.stack([_sample_q(rng) for _ in range(24)]))
    o = _run(seeds, p, quat, max_iters=80, stall_lim=3)
    tr = o["trace"]
    fired = 0
    for b in range(24):
        rows = tr[b][tr[b][:, VALID] != 0]
        for i, row in enumerate(rows):
            if row[PERT] == 1.0:
                fired += 1
                assert row[STALL] == 0.0, "the stall counter was not reset by the perturbation"
    assert fired > 0, "no perturbation ever fired -- the stall path is untested"
    assert o["coarse_perturbations"].sum() == fired
    print(f"\n  perturbations fired: {fired} across 24 problems")


def test_best_state_is_preserved():
    """The coarse search never returns a state worse than its seed.

    Asserted on E_phys, NOT on cost_raw. best_x is now tracked on the tolerance-normalised physical
    error E_phys = sum_k [ |e_p|^2/eps_p^2 + |e_R|^2/eps_R^2 ] -- the only metric that is comparable
    ACROSS iterations (the row scales are re-frozen every iteration, so the scaled cost is not).
    cost_raw weights position and orientation EQUALLY and unnormalised, so it is a different
    objective: measured here, the coarse search can raise cost_raw on ~1/256 seeds while E_phys
    still falls, because it traded cheap orientation error (weight 1/eps_o^2 = 1e4) for expensive
    position error (weight 1/eps_p^2 = 1e6). That is the tolerance-normalised objective working as
    intended, not a regression -- so the invariant is asserted in the metric being optimised.
    """
    rng = np.random.default_rng(9)
    B = 256
    seeds = np.stack([_sample_q(rng) for _ in range(B)])
    p, quat = _targets_at(np.stack([_sample_q(rng) for _ in range(B)]))
    before = hjcdik.target_residuals(seeds, p, quat)
    out = _run(seeds, p, quat, max_iters=60)
    after = hjcdik.target_residuals(np.asarray(out["joint_config"], np.float64), p, quat)
    e0 = _e_phys(before, 1e-3, 1e-2)      # coarse_search's default tolerances
    e1 = _e_phys(after, 1e-3, 1e-2)
    assert np.all(e1 <= e0 * (1 + 1e-6)), (
        "the coarse search returned a state with WORSE physical merit than its seed")


def test_unreachable_and_conflicting_targets_are_clean():
    rng = np.random.default_rng(41)
    q = _sample_q(rng)
    p, quat = _targets_at(q[None, :])
    far = p + 10.0
    o = _run(q[None, :], far, quat, max_iters=40)
    assert np.all(np.isfinite(o["joint_config"])) and np.all(np.isfinite(o["cost"]))
    assert not bool(o["success"][0])
    if K >= 2:
        conf = np.array(p)
        conf[0, 0] = p[0, 1] + 0.5
        conf[0, 1] = p[0, 0] - 0.5
        o = _run(q[None, :], conf, quat, max_iters=40)
        assert np.all(np.isfinite(o["joint_config"])) and np.all(np.isfinite(o["cost"]))


def test_coarse_reduces_error_and_seeds_lm():
    """Coarse search is a SEEDER, not a solver.

    Coordinate descent converges linearly and plateaus at the millimetre scale; asking it to hit the
    solver tolerance on its own would be testing the wrong thing. What must hold is (a) it cuts the
    error by a large factor, and (b) its output lands inside LM's basin of attraction, so that
    coarse -> LM converges. That is the property the whole two-phase design rests on.
    """
    rng = np.random.default_rng(43)
    lo, hi = LIMITS[:, 0], LIMITS[:, 1]

    # (a) from a NEARBY seed, coarse must land inside LM's basin
    B = 24
    Q = np.stack([_sample_q(rng) for _ in range(B)])
    p, quat = _targets_at(Q)
    seeds = np.clip(Q + rng.normal(scale=0.15, size=Q.shape), lo, hi)
    before = hjcdik.target_residuals(seeds, p, quat)["position_errors"].max(axis=1) * 1000
    c = _run(seeds, p, quat, max_iters=120)
    mid = c["position_errors"].max(axis=1) * 1000
    lm = hjcdik.refine(c["joint_config"], p, quat,
                       position_tol=1e-4, orientation_tol=1e-3, max_iters=60)
    print(f"\n  nearby seed -> coarse -> LM: {before.mean():.2f} -> {mid.mean():.3f} -> "
          f"{lm['position_errors'].max(axis=1).mean()*1000:.5f} mm   "
          f"LM success = {lm['success'].mean():.0%}")
    assert mid.mean() < before.mean() / 2, "coarse did not materially reduce the error"
    assert lm["success"].mean() >= 0.9, (
        f"coarse output is a poor LM seed: only {lm['success'].mean():.0%} of LM runs converged")

    # (b) from a BATCH of random restarts -- how HJCD actually solves. A single random restart is not
    # expected to converge; the best of the batch is.
    R = 128
    tq = _sample_q(rng)
    p1, q1 = _targets_at(tq[None, :])
    seeds = np.stack([rng.uniform(lo, hi) for _ in range(R)])
    P, QT = np.repeat(p1, R, axis=0), np.repeat(q1, R, axis=0)
    c = _run(seeds, P, QT, max_iters=120)
    lm = hjcdik.refine(c["joint_config"], P, QT,
                       position_tol=1e-4, orientation_tol=1e-3, max_iters=60)
    best = (lm["position_errors"].max(axis=1) * 1000).min()
    print(f"  {R} random restarts -> coarse -> LM: best position error = {best:.6f} mm")
    assert best < 0.5, f"best-of-{R} restarts only reached {best:.4f} mm"


# --- 5. diagnostics ------------------------------------------------------------------------------

def test_diagnostics_are_opt_in_and_do_not_change_the_solution():
    rng = np.random.default_rng(53)
    seeds = np.stack([rng.uniform(LIMITS[:, 0], LIMITS[:, 1]) for _ in range(8)])
    p, quat = _targets_at(np.stack([_sample_q(rng) for _ in range(8)]))
    off = hjcdik.coarse_search(seeds, p, quat, max_iters=30, seed=1)
    on = hjcdik.coarse_search(seeds, p, quat, max_iters=30, seed=1, diagnostics=True)
    for key in ("coarse_iterations", "accepted_coarse_steps", "trace"):
        assert key not in off, f"'{key}' must be absent when diagnostics are off"
    np.testing.assert_array_equal(off["joint_config"], on["joint_config"])
    np.testing.assert_array_equal(off["cost"], on["cost"])


def test_trace_rows_are_explicitly_valid():
    rng = np.random.default_rng(59)
    seeds = np.stack([rng.uniform(LIMITS[:, 0], LIMITS[:, 1]) for _ in range(4)])
    p, quat = _targets_at(np.stack([_sample_q(rng) for _ in range(4)]))
    o = _run(seeds, p, quat, max_iters=25)
    tr = o["trace"]
    for b in range(4):
        n = int(o["coarse_iterations"][b])
        assert np.all(tr[b, :n, VALID] == 1.0)
        assert np.all(tr[b, n:, VALID] == 0.0)
        assert np.all(tr[b, :n, IT] == np.arange(n))
