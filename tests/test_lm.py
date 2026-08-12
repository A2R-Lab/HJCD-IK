"""Phase 3: multi-target LM refinement.

Sign convention (approved, pinned by tests/test_residuals.py):
    e_p = p* - p            e_R = Log(R* R^T)           de/dq = -J
    (J^T W J + lambda diag(A)) dq = J^T W e             x <- x + a*dq     [PLUS sign]

The device accumulates A and b target-by-target and never forms the stacked 6K x N Jacobian.
test_A_b_vs_cpu_stacked_jacobian builds that stacked Jacobian explicitly on the CPU and compares --
that is the ground truth for the whole phase.
"""
import os
import re
from pathlib import Path

import numpy as np
import pytest

import hjcdik
from urdf_fk import UrdfFK

REPO = Path(__file__).resolve().parents[1]
URDF = Path(os.environ.get("HJCD_TEST_URDF", REPO / "csrc" / "urdf" / "panda.urdf"))

N = hjcdik.num_joints()
K = hjcdik.num_targets()
ALL = (1 << K) - 1


def _axis_meta():
    txt = (REPO / "csrc" / "generated" / "grid.cuh").read_text()
    col = [int(v) for v in re.search(r"JOINT_AXIS_COL\[\d+\] = \{([^}]*)\}", txt).group(1).split(",")]
    sgn = [int(v) for v in re.search(r"JOINT_AXIS_SIGN\[\d+\] = \{([^}]*)\}", txt).group(1).split(",")]
    return col, sgn


def cpu_stacked(q, tgt_p, tgt_q, active, wp, wo):
    """Explicit stacked 6K x N Jacobian + stacked residual -> A = J^T W J, b = J^T W e.

    Deliberately dense and dumb: it fills the FULL 6K x N matrix, zeroing non-ancestor columns, and
    forms the normal equations by matrix multiply. This is the thing the device must reproduce
    without ever materializing J.
    """
    col, sgn = _axis_meta()
    meta = hjcdik.target_metadata()
    T = hjcdik.link_transforms(q[None, :])[0]
    X = hjcdik.target_transforms(q[None, :])[0]
    r = hjcdik.target_residuals(q[None, :], tgt_p[None], tgt_q[None],
                                active_target_mask=np.array([active], dtype=np.uint32),
                                position_weights=wp[None], orientation_weights=wo[None])

    J = np.zeros((6 * K, N))
    e = np.zeros(6 * K)
    W = np.zeros(6 * K)
    for k in range(K):
        if not (active >> k) & 1:
            continue                                    # inactive: rows stay zero
        p_t = X[k][:3, 3]
        for j in range(N):
            if not (int(meta["target_ancestor_mask"][k]) >> j) & 1:
                continue                                # non-ancestor: column stays EXACTLY zero
            axis = sgn[j] * T[j][:3, col[j]]
            J[6*k:6*k+3, j] = np.cross(axis, p_t - T[j][:3, 3])   # Jv
            J[6*k+3:6*k+6, j] = axis                              # Jw
        e[6*k:6*k+3] = r["position_residuals"][0, k]
        e[6*k+3:6*k+6] = r["orientation_residuals"][0, k]
        W[6*k:6*k+3] = wp[k]
        W[6*k+3:6*k+6] = wo[k]

    A = J.T @ (W[:, None] * J)
    b = J.T @ (W * e)
    return A, b, J


@pytest.fixture(scope="module")
def oracle():
    return UrdfFK(URDF)


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(17)


LIMITS = hjcdik.joint_limits()          # exactly what the LM clamps to


def _sample_q(rng, margin=0.15):
    """A config strictly INSIDE the joint limits.

    Not cosmetic: Panda's joint 3 lives in [-3.07, -0.07] and joint 5 in [-0.0175, 3.75], so the
    obvious U(-a, a) is infeasible. Seeding LM from an out-of-limits config makes the very first
    clamp move the config, and the "target" generated from it is then unreachable -- which looks
    exactly like a broken solver.
    """
    lo, hi = LIMITS[:, 0], LIMITS[:, 1]
    span = hi - lo
    return rng.uniform(lo + margin * span, hi - margin * span)


def _perturb(rng, q, scale):
    """A seed near q, still inside the limits."""
    lo, hi = LIMITS[:, 0], LIMITS[:, 1]
    return np.clip(q + rng.normal(scale=scale, size=len(q)), lo, hi)


def _reachable(q):
    """Target poses that q reaches exactly."""
    T = hjcdik.target_transforms(q[None, :])[0]
    p = np.stack([T[k][:3, 3] for k in range(K)])
    quat = np.stack([_quat_from_R(T[k][:3, :3]) for k in range(K)])
    return p, quat


def _quat_from_R(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = np.array([0.25*s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        q = np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif R[1,1] > R[2,2]:
        s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        q = np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    else:
        s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        q = np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])
    return q / np.linalg.norm(q)


def _dev_AB(q, p, quat, active, wp, wo):
    r = hjcdik.normal_equations(q[None, :], p[None], quat[None],
                                active_target_mask=np.array([active], dtype=np.uint32),
                                position_weights=wp[None], orientation_weights=wo[None])
    return r["A"][0], r["b"][0]


# --- 1, 2, 3, 4, 5, 6: A and b vs the CPU stacked Jacobian ---------------------------------------

def _cases(rng):
    cases = []
    ident = np.ones(K)
    for k in range(K):                                    # 2. each target alone
        m = 1 << k
        cases.append((f"target{k}_only", m, ident.copy(), ident.copy()))
    cases.append(("all_targets", ALL, ident.copy(), ident.copy()))          # 3.
    if K >= 2:
        cases.append(("first_two", 0b11, ident.copy(), ident.copy()))       # 4.
        cases.append(("hands" if K == 4 else "pair", 0b0011 if K == 4 else 0b11,
                      ident.copy(), ident.copy()))
        if K == 4:
            cases.append(("feet", 0b1100, ident.copy(), ident.copy()))
    cases.append(("position_only", ALL, ident.copy(), np.zeros(K)))         # 5.
    cases.append(("orientation_only", ALL, np.zeros(K), ident.copy()))      # 5.
    cases.append(("mixed_weights", ALL,                                      # 6.
                  np.linspace(0.25, 3.0, K), np.linspace(2.0, 0.1, K)))
    return cases


@pytest.mark.parametrize("trial", range(3))
def test_A_b_vs_cpu_stacked_jacobian(rng, trial):
    q = _sample_q(rng)
    p0, quat0 = _reachable(q)
    p = p0 + rng.normal(scale=0.05, size=p0.shape)         # perturb so residuals are nonzero
    quat = quat0 + rng.normal(scale=0.02, size=quat0.shape)
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)

    worst_A = worst_b = 0.0
    for name, mask, wp, wo in _cases(rng):
        A_ref, b_ref, _ = cpu_stacked(q, p, quat, mask, wp, wo)
        A_dev, b_dev = _dev_AB(q, p, quat, mask, wp, wo)
        sA = max(1.0, np.abs(A_ref).max())
        sb = max(1.0, np.abs(b_ref).max())
        eA = np.abs(A_dev - A_ref).max() / sA
        eb = np.abs(b_dev - b_ref).max() / sb
        assert eA < 1e-12, f"[{name}] A mismatch, rel {eA:.3e}"
        assert eb < 1e-12, f"[{name}] b mismatch, rel {eb:.3e}"
        worst_A, worst_b = max(worst_A, eA), max(worst_b, eb)
    print(f"\n  trial {trial}: worst rel err over {len(_cases(rng))} cases  "
          f"A={worst_A:.2e}  b={worst_b:.2e}")


# --- 7, 8, 9: structure of A ---------------------------------------------------------------------

def test_non_ancestor_columns_are_exactly_zero(rng):
    """A row/col for a joint that cannot move ANY active target must be exactly zero."""
    if K < 2:
        pytest.skip("needs K >= 2")
    meta = hjcdik.target_metadata()
    q = _sample_q(rng)
    p, quat = _reachable(q)
    p = p + 0.05
    for k in range(K):
        mask = 1 << k
        A, b = _dev_AB(q, p, quat, mask, np.ones(K), np.ones(K))
        anc = int(meta["target_ancestor_mask"][k])
        for j in range(N):
            if (anc >> j) & 1:
                continue
            assert np.all(A[j, :] == 0.0), f"joint {j} not an ancestor of target {k}, A row nonzero"
            assert np.all(A[:, j] == 0.0), f"joint {j} not an ancestor of target {k}, A col nonzero"
            assert b[j] == 0.0, f"joint {j} not an ancestor of target {k}, b nonzero"


def test_A_is_symmetric(rng):
    q = _sample_q(rng)
    p, quat = _reachable(q)
    A, _ = _dev_AB(q, p + 0.05, quat, ALL, np.linspace(0.5, 2.0, K), np.linspace(1.5, 0.3, K))
    asym = np.abs(A - A.T).max() / max(1.0, np.abs(A).max())
    assert asym < 1e-14, f"A not symmetric: {asym:.3e}"


def test_A_is_positive_semidefinite(rng):
    q = _sample_q(rng)
    p, quat = _reachable(q)
    A, _ = _dev_AB(q, p + 0.05, quat, ALL, np.ones(K), np.ones(K))
    w = np.linalg.eigvalsh(0.5 * (A + A.T))
    assert w.min() > -1e-9 * max(1.0, w.max()), f"A has eigenvalue {w.min():.3e}"


# --- 10: LM step sign ----------------------------------------------------------------------------

def test_lm_step_reduces_cost(rng):
    """An accepted LM step must DECREASE the cost. A sign error in b would increase it."""
    q = _sample_q(rng)
    p, quat = _reachable(q)
    seed = _perturb(rng, q, 0.03)              # small error => LM territory
    c0 = hjcdik.target_residuals(seed[None], p[None], quat[None])["cost_raw"][0]
    out = hjcdik.refine(seed[None], p[None], quat[None], max_iters=1)
    c1 = out["cost"][0]
    assert c1 < c0, f"one LM iteration increased cost: {c0:.6e} -> {c1:.6e} (sign error in b?)"
    print(f"\n  one LM step: cost {c0:.3e} -> {c1:.3e}")


# --- 12: convergence from nearby seeds (LM only, no coarse search) --------------------------------

def _converge_case(rng, mask, n=24, scale=0.08, iters=60):
    ok, pe, oe = 0, [], []
    Q, P, QT = [], [], []
    for _ in range(n):
        q = _sample_q(rng)
        p, quat = _reachable(q)
        Q.append(_perturb(rng, q, scale))
        P.append(p)
        QT.append(quat)
    out = hjcdik.refine(np.array(Q), np.array(P), np.array(QT),
                        active_target_mask=np.full(n, mask, dtype=np.uint32),
                        position_tol=1e-4, orientation_tol=1e-3, max_iters=iters,
                        diagnostics=True)
    act = [k for k in range(K) if (mask >> k) & 1]
    return (out["success"].mean(),
            out["position_errors"][:, act].max(),
            out["orientation_errors"][:, act].max(),
            out["iterations"].mean())


@pytest.mark.parametrize("name,mask", [("all", ALL)])
def test_converge_from_nearby_seeds(rng, name, mask):
    sr, pe, oe, it = _converge_case(rng, mask)
    print(f"\n  [{name}] success={sr:.0%}  max_pos={pe*1000:.4f} mm  max_ori={oe:.2e} rad  "
          f"mean_iters={it:.1f}")
    assert sr >= 0.9, f"only {sr:.0%} converged"


@pytest.mark.skipif(K != 4, reason="G1 only")
@pytest.mark.parametrize("name,mask", [("both_hands", 0b0011), ("both_feet", 0b1100),
                                       ("all_four", 0b1111)])
def test_g1_multitarget_convergence(rng, name, mask):
    sr, pe, oe, it = _converge_case(rng, mask)
    print(f"\n  [{name}] success={sr:.0%}  max_pos={pe*1000:.4f} mm  max_ori={oe:.2e} rad  "
          f"mean_iters={it:.1f}")
    assert sr >= 0.9, f"[{name}] only {sr:.0%} converged"


# --- 13, 14: robustness --------------------------------------------------------------------------

def test_unreachable_targets_fail_cleanly(rng):
    """Far-away / conflicting targets must fail, not NaN."""
    q = _sample_q(rng)
    p, quat = _reachable(q)
    far = np.array(p) + np.array([10.0, 10.0, 10.0])       # metres away: unreachable
    out = hjcdik.refine(q[None], far[None], quat[None], max_iters=30)
    assert np.all(np.isfinite(out["joint_config"]))
    assert np.all(np.isfinite(out["position_errors"]))
    assert np.all(np.isfinite(out["orientation_errors"]))
    assert np.all(np.isfinite(out["cost"]))
    assert not bool(out["success"][0])

    if K >= 2:                                              # conflicting: same target, two places
        conf = np.array(p)
        conf[0] = p[1] + 0.4
        conf[1] = p[0] - 0.4
        out = hjcdik.refine(q[None], conf[None], quat[None], max_iters=30)
        assert np.all(np.isfinite(out["joint_config"]))
        assert np.all(np.isfinite(out["cost"]))


def test_quaternion_sign_invariance_through_lm(rng):
    """q and -q are the same rotation: the whole LM path must be invariant."""
    q = _sample_q(rng)
    p, quat = _reachable(q)
    seed = _perturb(rng, q, 0.05)
    a = hjcdik.refine(seed[None], p[None], quat[None], max_iters=40)
    b = hjcdik.refine(seed[None], p[None], -quat[None], max_iters=40)
    np.testing.assert_allclose(a["joint_config"], b["joint_config"], atol=1e-12)
    np.testing.assert_allclose(a["cost"], b["cost"], atol=1e-14)


def test_inactive_targets_do_not_affect_solution(rng):
    """A target that is masked off must not influence the refined config at all."""
    if K < 2:
        pytest.skip("needs K >= 2")
    q = _sample_q(rng)
    p, quat = _reachable(q)
    seed = _perturb(rng, q, 0.04)
    a = hjcdik.refine(seed[None], p[None], quat[None],
                      active_target_mask=np.array([0b1], dtype=np.uint32), max_iters=30)
    p2 = np.array(p)
    p2[1:] += 5.0                                           # wreck the inactive targets
    b = hjcdik.refine(seed[None], p2[None], quat[None],
                      active_target_mask=np.array([0b1], dtype=np.uint32), max_iters=30)
    np.testing.assert_allclose(a["joint_config"], b["joint_config"], atol=1e-13)
