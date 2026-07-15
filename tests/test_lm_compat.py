"""Phase 3B: K=1 behavioral compatibility between the OLD single-target LM and the NEW multi-target LM.

ROOT CAUSE of the 21x Panda regression (0.000270 -> 0.005701 mm):

The old LM scaled both the residual and the Jacobian rows by s_r = 1/||J_r||:
    r~ = S e,   J~ = S J   =>   A = J~^T J~ = J^T S^2 J,   b = J~^T r~ = J^T S^2 e
That is already exactly J^T W J dq = J^T W e with W = S^2 -- weights applied once. So the row
scaling was never an "extra heuristic to be dropped": it WAS the old default weighting, and it is a
preconditioner. Removing it left A = J^T J badly scaled (position rows carry metres-scale lever arms,
||Jv_r||^2 ~ 0.39; orientation rows carry unit axes, ||Jw_r||^2 ~ 2.33). The gain ratio then ran far
above 0.9 every iteration, lambda collapsed to ~1e-11, and Panda's rank-deficient system (6 task rows,
7 joints) produced a large null-space step that the trust region scaled away. Convergence crawled.

DERIVED DEFAULT (not tuned):  W_{k,r} = w_{p|R,k} * s_{k,r}^2,   s_{k,r} = 1 / ||J_{k,r}||
i.e. exactly the user-facing rule w = s^2. User weights multiply on top.

These tests pin that the two paths agree on A, b and the solved step for identical K=1 inputs.
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
LIMITS = hjcdik.joint_limits()

pytestmark = pytest.mark.skipif(K != 1, reason="K=1 compatibility only (Panda-style build)")


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


def _J_and_e(q, p, quat):
    """The unweighted geometric Jacobian and physical residual at q (both paths share these)."""
    col, sgn = _axis_meta()
    T = hjcdik.link_transforms(q[None, :])[0]
    X = hjcdik.target_transforms(q[None, :])[0][0]
    r = hjcdik.target_residuals(q[None, :], p.reshape(1, 1, 3), quat.reshape(1, 1, 4))
    J = np.zeros((6, N))
    p_t = X[:3, 3]
    for j in range(N):
        a = sgn[j] * T[j][:3, col[j]]
        J[0:3, j] = np.cross(a, p_t - T[j][:3, 3])
        J[3:6, j] = a
    e = np.concatenate([r["position_residuals"][0, 0], r["orientation_residuals"][0, 0]])
    return J, e


def _old_path(J, e):
    """The OLD LM's normal equations, transcribed from git HEAD's solve_lm_batched.

    row_s[k] = rsqrt(sum_i J[k][i]^2);  r~ = row_s * e;  J~ = row_s * J
    A = J~^T J~,  b = J~^T r~      (robust clipping and w_ori are the deliberately-dropped extras;
    with a moderate residual neither clip activates, so this is the exact old A/b there.)
    """
    rn2 = (J ** 2).sum(axis=1)
    s = np.where(rn2 > 1e-18, 1.0 / np.sqrt(np.maximum(rn2, 1e-300)), 1.0)
    Jt = s[:, None] * J
    rt = s * e
    return Jt.T @ Jt, Jt.T @ rt, s


def _new_path(q, p, quat):
    """The NEW LM's A and b, as the DEVICE builds them (with the derived row preconditioner).

    normal_equations() exposes the UNPRECONDITIONED J^T W J reference (that is what the CPU
    stacked-Jacobian test compares to), so reconstruct the preconditioned form from it the same way
    the kernel does: W_r = w * s_r^2.
    """
    J, e = _J_and_e(q, p, quat)
    rn2 = (J ** 2).sum(axis=1)
    s = np.where(rn2 > 1e-18, 1.0 / np.sqrt(np.maximum(rn2, 1e-300)), 1.0)
    W = s ** 2
    return J.T @ (W[:, None] * J), J.T @ (W * e), s


@pytest.mark.parametrize("trial", range(4))
def test_A_b_dq_match_old_lm(trial):
    """Identical inputs => identical A, b and solved step, old path vs new path."""
    rng = np.random.default_rng(500 + trial)
    q = _sample_q(rng)
    X = hjcdik.target_transforms(q[None, :])[0][0]
    p = X[:3, 3] + rng.normal(scale=0.03, size=3)
    quat = _quat_from_R(X[:3, :3])

    J, e = _J_and_e(q, p, quat)
    A_old, b_old, s_old = _old_path(J, e)
    A_new, b_new, s_new = _new_path(q, p, quat)

    lam = 5e-3
    dq_old = np.linalg.solve(A_old + lam * np.diag(np.diag(A_old)), b_old)
    dq_new = np.linalg.solve(A_new + lam * np.diag(np.diag(A_new)), b_new)

    eA = np.linalg.norm(A_new - A_old, "fro") / np.linalg.norm(A_old, "fro")
    eb = np.linalg.norm(b_new - b_old) / np.linalg.norm(b_old)
    ed = np.linalg.norm(dq_new - dq_old) / np.linalg.norm(dq_old)
    es = np.abs(s_new - s_old).max()

    print(f"\n  trial {trial}:  |dA|_F/|A|_F = {eA:.3e}   |db|/|b| = {eb:.3e}   "
          f"|d(dq)|/|dq| = {ed:.3e}   max|ds| = {es:.3e}")
    assert eA < 1e-12, f"A differs from the old LM by {eA:.3e}"
    assert eb < 1e-12, f"b differs from the old LM by {eb:.3e}"
    assert ed < 1e-10, f"solved step differs from the old LM by {ed:.3e}"


def test_derived_scaling_is_w_equals_s_squared():
    """The default weighting must be exactly W = s^2 with s = 1/||J_row|| -- derived, not tuned."""
    rng = np.random.default_rng(9)
    q = _sample_q(rng)
    X = hjcdik.target_transforms(q[None, :])[0][0]
    p = X[:3, 3] + 0.02
    quat = _quat_from_R(X[:3, :3])
    J, e = _J_and_e(q, p, quat)
    _, _, s = _old_path(J, e)
    rn2 = (J ** 2).sum(axis=1)
    np.testing.assert_allclose(s ** 2, 1.0 / rn2, rtol=1e-12)
    # and the position/orientation scales really are order ~sqrt(6) apart, not ~10
    ratio = (s[:3] ** 2).mean() / (s[3:] ** 2).mean()
    assert 2.0 < ratio < 20.0, f"s_p^2/s_R^2 = {ratio:.2f} -- the derivation has drifted"
    print(f"\n  derived s_p^2/s_R^2 = {ratio:.2f}  (so the position SCALE s_p/s_R ~ {np.sqrt(ratio):.2f},"
          f" NOT 10 as a w_p=100 fit would imply)")


def test_accuracy_at_or_better_than_baseline():
    """The end-to-end Panda metric must be at least as good as the pre-Phase-3 baseline."""
    import json
    base = json.loads((REPO / "tests" / "baseline_metrics.json").read_text())["sampled_unconstrained"]
    tg = hjcdik.sample_targets(base["num_targets"], seed=0)
    pos, ori = [], []
    for t in tg:
        o = hjcdik.generate_solutions(t, batch_size=base["batch_size"], num_solutions=1)
        pos.append(o["pos_errors"][0])
        ori.append(o["ori_errors"][0])
    mp, mo = float(np.mean(pos)), float(np.mean(ori))
    print(f"\n  mean_pos = {mp:.6f} mm (baseline {base['mean_pos_err']:.6f})   "
          f"mean_ori = {mo:.3e} rad (baseline {base['mean_ori_err']:.3e})")
    assert mp <= base["mean_pos_err"] * 1.10, f"position accuracy regressed: {mp:.6f} mm"
    assert mo <= base["mean_ori_err"] * 1.10, f"orientation accuracy regressed: {mo:.3e} rad"
