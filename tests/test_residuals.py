"""Phase 2: the multi-target residual layer, tested independently of any optimizer.

CONVENTIONS PINNED HERE (Phase 3's Jacobian must match):
  quaternions        WXYZ, unit. Normalized on the host; q and -q are the same input.
  position residual  e_p = p* - p                                (world, metres)
  orientation resid. e_R = Log(R* R^T) == rotvec(q* (x) q^-1)    (WORLD frame, radians)
  cost               c_k = w_p |e_p|^2 + w_R |e_R|^2   -- weights applied ONCE, here only.

e_R is the SPATIAL error, not the body-frame Log(R^T R*): it is what pairs with a world-frame
angular Jacobian Jw = axis_world. test_residual_sign_matches_jacobian_convention is the check that
keeps Phase 3 honest about that.

Robot-agnostic via HJCD_TEST_URDF. K=1 (Panda) and K=4 (G1) both run.
"""
import os
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


def _quat_from_R(R):
    """Row-major 3x3 -> unit wxyz."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                      (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s,
                      (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s,
                      (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def _R_from_axis_angle(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * Kx + (1 - np.cos(ang)) * (Kx @ Kx)


@pytest.fixture(scope="module")
def oracle():
    fk = UrdfFK(URDF)
    assert len(fk.joint_order()) == N
    return fk


@pytest.fixture(scope="module")
def q0():
    return np.random.default_rng(5).uniform(-0.7, 0.7, size=(1, N))


@pytest.fixture(scope="module")
def poses(q0):
    """The CURRENT target poses at q0 -- so a target set to these is exactly reached."""
    T = hjcdik.target_transforms(q0)[0]                    # (K, 4, 4)
    p = np.stack([T[k][:3, 3] for k in range(K)])
    qt = np.stack([_quat_from_R(T[k][:3, :3]) for k in range(K)])
    return p, qt


def _call(q, p, qt, **kw):
    return hjcdik.target_residuals(q, p[None] if p.ndim == 2 else p,
                                   qt[None] if qt.ndim == 2 else qt, **kw)


# --- orientation residual: the six required properties ------------------------------------------

def test_ori_identical_orientation_is_zero(q0, poses):
    p, qt = poses
    r = _call(q0, p, qt)
    assert np.abs(r["orientation_residuals"]).max() < 1e-9
    assert np.abs(r["position_residuals"]).max() < 1e-9
    assert np.abs(r["orientation_errors"]).max() < 1e-9


def test_ori_quaternion_sign_flip_is_identical(q0, poses):
    p, qt = poses
    rng = np.random.default_rng(2)
    qt2 = np.stack([_quat_from_R(_R_from_axis_angle(rng.normal(size=3), 0.3) @
                                 hjcdik.target_transforms(q0)[0][k][:3, :3]) for k in range(K)])
    a = _call(q0, p, qt2)
    b = _call(q0, p, -qt2)                                  # exact double-cover negation
    np.testing.assert_allclose(a["orientation_residuals"], b["orientation_residuals"], atol=1e-14)
    np.testing.assert_allclose(a["target_costs"], b["target_costs"], atol=1e-14)


@pytest.mark.parametrize("ang", [1e-6, 1e-3, 0.1, 1.0, 2.5])
def test_ori_known_rotation_magnitude(q0, poses, ang):
    """|e_R| must equal the applied rotation angle."""
    p, qt = poses
    T = hjcdik.target_transforms(q0)[0]
    axis = np.array([0.3, -0.5, 0.8])
    qt2 = np.stack([_quat_from_R(_R_from_axis_angle(axis, ang) @ T[k][:3, :3]) for k in range(K)])
    r = _call(q0, p, qt2)
    for k in range(K):
        assert abs(r["orientation_errors"][0, k] - ang) < 1e-7, (
            f"target {k}: |e_R| = {r['orientation_errors'][0,k]:.9f}, expected {ang}")


def test_ori_near_pi_is_finite(q0, poses):
    """Rotations approaching pi must stay finite (no 1/sin(theta/2) blowup)."""
    p, qt = poses
    T = hjcdik.target_transforms(q0)[0]
    for ang in (np.pi - 1e-3, np.pi - 1e-7, np.pi):
        qt2 = np.stack([_quat_from_R(_R_from_axis_angle([0, 0, 1], ang) @ T[k][:3, :3])
                        for k in range(K)])
        r = _call(q0, p, qt2)
        e = r["orientation_residuals"]
        assert np.all(np.isfinite(e)), f"non-finite residual at angle {ang}"
        assert np.all(r["orientation_errors"] <= np.pi + 1e-6)
        assert np.abs(r["orientation_errors"] - ang).max() < 1e-5


def test_quaternion_normalization_is_defined(q0, poses):
    """Non-unit target quaternions are normalized on the host; residual is scale-invariant."""
    p, qt = poses
    T = hjcdik.target_transforms(q0)[0]
    qt2 = np.stack([_quat_from_R(_R_from_axis_angle([1, 0, 0], 0.4) @ T[k][:3, :3])
                    for k in range(K)])
    a = _call(q0, p, qt2)
    b = _call(q0, p, qt2 * 7.3)                            # same rotation, non-unit
    np.testing.assert_allclose(a["orientation_residuals"], b["orientation_residuals"], atol=1e-12)
    with pytest.raises(ValueError, match="zero quaternion"):
        _call(q0, p, np.zeros_like(qt2))


def test_residual_sign_matches_jacobian_convention(q0, poses, oracle):
    """THE convention test: d(e)/dq must be -J, with J the world-frame geometric Jacobian.

    Phase 3 solves (J^T W J) dq = J^T W e with J built from axis_world. That is only a descent
    direction if the residual really is e = target - current, i.e. de/dq = -J. Verify by finite
    difference against the analytic column, for every ancestor joint of every target.
    """
    import re
    txt = (REPO / "csrc" / "generated" / "grid.cuh").read_text()
    col = [int(v) for v in re.search(r"JOINT_AXIS_COL\[\d+\] = \{([^}]*)\}", txt).group(1).split(",")]
    sgn = [int(v) for v in re.search(r"JOINT_AXIS_SIGN\[\d+\] = \{([^}]*)\}", txt).group(1).split(",")]
    meta = hjcdik.target_metadata()

    p, qt = poses
    q = q0[0]
    eps = 1e-6

    T0 = hjcdik.link_transforms(q[None, :])[0]
    X0 = hjcdik.target_transforms(q[None, :])[0]

    Q = np.repeat(q[None, :], 2 * N, axis=0)
    for j in range(N):
        Q[2 * j, j] += eps
        Q[2 * j + 1, j] -= eps
    P = np.repeat(p[None], 2 * N, axis=0)
    QT = np.repeat(qt[None], 2 * N, axis=0)
    r = hjcdik.target_residuals(Q, P, QT)

    worst_p = worst_o = 0.0
    for k in range(K):
        p_tgt = X0[k][:3, 3]
        for j in range(N):
            if not (int(meta["joint_target_mask"][j]) >> k) & 1:
                continue
            de_p = (r["position_residuals"][2 * j, k] -
                    r["position_residuals"][2 * j + 1, k]) / (2 * eps)
            de_o = (r["orientation_residuals"][2 * j, k] -
                    r["orientation_residuals"][2 * j + 1, k]) / (2 * eps)
            axis = sgn[j] * T0[j][:3, col[j]]
            Jv = np.cross(axis, p_tgt - T0[j][:3, 3])      # Phase 3's Jacobian columns
            Jw = axis
            worst_p = max(worst_p, np.abs(de_p - (-Jv)).max())
            worst_o = max(worst_o, np.abs(de_o - (-Jw)).max())
    assert worst_p < 1e-5, f"d(e_p)/dq != -Jv  (off by {worst_p:.2e})"
    assert worst_o < 1e-5, f"d(e_R)/dq != -Jw  (off by {worst_o:.2e}) -- world/body frame mismatch"
    print(f"\n  de_p/dq vs -Jv: {worst_p:.2e}    de_R/dq vs -Jw: {worst_o:.2e}")


# --- masks, weights, costs ----------------------------------------------------------------------

def test_inactive_targets_are_exactly_zero(q0, poses):
    if K == 1:
        pytest.skip("single-target robot: no inactive target possible")
    p, qt = poses
    off = np.array(p)
    off[:, 0] += 1.0                                        # make every target badly wrong
    r = _call(q0, off, qt, active_target_mask=np.array([[True] + [False] * (K - 1)]))
    for k in range(1, K):
        assert r["position_residuals"][0, k].tolist() == [0.0, 0.0, 0.0]
        assert r["orientation_residuals"][0, k].tolist() == [0.0, 0.0, 0.0]
        assert r["position_errors"][0, k] == 0.0
        assert r["orientation_errors"][0, k] == 0.0
        assert r["target_costs"][0, k] == 0.0
        assert bool(r["target_success"][0, k]) is False
    assert r["cost_raw"][0] == pytest.approx(r["target_costs"][0, 0])


def test_one_active_and_all_active(q0, poses):
    p, qt = poses
    r_all = _call(q0, p, qt)
    assert int(r_all["active_target_mask"][0]) == ALL
    assert bool(r_all["success"][0]) is True
    r_one = _call(q0, p, qt, active_target_mask=np.array([[True] + [False] * (K - 1)]))
    assert int(r_one["active_target_mask"][0]) == 1


def test_different_masks_across_batch(q0, poses):
    if K < 2:
        pytest.skip("needs K >= 2")
    p, qt = poses
    B = 3
    Q = np.repeat(q0, B, axis=0)
    P = np.repeat(p[None], B, axis=0)
    QT = np.repeat(qt[None], B, axis=0)
    m = np.zeros((B, K), dtype=bool)
    m[0, 0] = True
    m[1, 1] = True
    m[2, :] = True
    r = hjcdik.target_residuals(Q, P, QT, active_target_mask=m)
    assert [int(x) for x in r["active_target_mask"]] == [1, 2, ALL]
    assert all(bool(s) for s in r["success"])              # all reached at q0


def test_position_only_and_orientation_only(q0, poses):
    """A zero-weight channel is 'don't care': it must not block success even when badly wrong."""
    p, qt = poses
    T = hjcdik.target_transforms(q0)[0]
    bad_q = np.stack([_quat_from_R(_R_from_axis_angle([0, 0, 1], 1.0) @ T[k][:3, :3])
                      for k in range(K)])
    bad_p = np.array(p) + 0.5

    r = _call(q0, p, bad_q, orientation_weights=0.0)        # position-only
    assert bool(r["success"][0]) is True
    assert np.all(r["orientation_errors"][0] > 0.9)         # residual still REPORTED
    assert r["target_costs"][0].sum() == pytest.approx(0.0, abs=1e-12)

    r = _call(q0, bad_p, qt, position_weights=0.0)          # orientation-only
    assert bool(r["success"][0]) is True
    assert np.all(r["position_errors"][0] > 0.4)


def test_cost_equation_and_normalization(q0, poses):
    p, qt = poses
    bad_p = np.array(p) + 0.1
    T = hjcdik.target_transforms(q0)[0]
    bad_q = np.stack([_quat_from_R(_R_from_axis_angle([0, 1, 0], 0.2) @ T[k][:3, :3])
                      for k in range(K)])
    wp = np.arange(1, K + 1, dtype=float)
    wo = np.arange(1, K + 1, dtype=float) * 0.5
    r = _call(q0, bad_p, bad_q, position_weights=wp, orientation_weights=wo)

    pn, on = r["position_errors"][0], r["orientation_errors"][0]
    expect = wp * pn ** 2 + wo * on ** 2                     # weights applied EXACTLY once
    np.testing.assert_allclose(r["target_costs"][0], expect, rtol=1e-12)
    np.testing.assert_allclose(r["cost_raw"][0], expect.sum(), rtol=1e-12)
    np.testing.assert_allclose(r["cost_normalized"][0], expect.sum() / ((wp + wo).sum() + 1e-12),
                               rtol=1e-9)


def test_weight_broadcasting_scalar_K_BK(q0, poses):
    p, qt = poses
    bad_p = np.array(p) + 0.1
    a = _call(q0, bad_p, qt, position_weights=2.0)
    b = _call(q0, bad_p, qt, position_weights=np.full(K, 2.0))
    c = _call(q0, bad_p, qt, position_weights=np.full((1, K), 2.0))
    np.testing.assert_allclose(a["target_costs"], b["target_costs"], rtol=1e-14)
    np.testing.assert_allclose(a["target_costs"], c["target_costs"], rtol=1e-14)


# --- rejection ----------------------------------------------------------------------------------

def test_empty_mask_rejected(q0, poses):
    p, qt = poses
    with pytest.raises(ValueError, match="at least one target"):
        _call(q0, p, qt, active_target_mask=np.zeros((1, K), dtype=bool))


def test_invalid_mask_bits_rejected(q0, poses):
    p, qt = poses
    with pytest.raises(ValueError, match="bits above"):
        _call(q0, p, qt, active_target_mask=np.array([1 << K], dtype=np.uint32))


@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf])
def test_invalid_weights_rejected(q0, poses, bad):
    p, qt = poses
    with pytest.raises(ValueError):
        _call(q0, p, qt, position_weights=bad)


def test_active_target_with_zero_weights_rejected(q0, poses):
    p, qt = poses
    with pytest.raises(ValueError, match="both position_weights and orientation_weights"):
        _call(q0, p, qt, position_weights=0.0, orientation_weights=0.0)


def test_bad_shapes_rejected(q0, poses):
    p, qt = poses
    with pytest.raises(ValueError):
        hjcdik.target_residuals(q0, p[None][:, :, :2], qt[None])
    with pytest.raises(ValueError):
        hjcdik.target_residuals(q0, p[None], qt[None][..., :3])


# --- K=1 (Panda) compatibility ------------------------------------------------------------------

def test_single_target_matches_solver_error_convention(q0, poses):
    """For K=1 the residual layer must reproduce the solver's own pos/ori error definition."""
    if K != 1:
        pytest.skip("Panda-only")
    p, qt = poses
    off = np.array(p)
    off[0] += np.array([0.01, -0.02, 0.005])
    r = _call(q0, off, qt)
    X = hjcdik.target_transforms(q0)[0][0]
    np.testing.assert_allclose(r["position_residuals"][0, 0], off[0] - X[:3, 3], atol=1e-12)
    assert r["position_errors"][0, 0] == pytest.approx(np.linalg.norm(off[0] - X[:3, 3]), rel=1e-12)
