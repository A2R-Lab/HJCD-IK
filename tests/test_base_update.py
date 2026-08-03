"""M2: the host-side floating-base reference (hjcdik/base_update.py).

Two tiers:
  * CPU-only (the bulk). The base update's mathematics does not involve the robot: J_b
    depends on the joints ONLY through the contact point x_i (eq. 3), so synthetic contact
    points validate it exactly as rigorously as real G1 FK does -- and on a box with no CUDA.
  * GPU (marked, skipped with a reason). Confirms the contact points HJCD actually produces
    compose with the base as eq. (1) claims, i.e. that the reference is wired to the real
    robot and not just self-consistent.

Run:  python -m pytest tests/test_base_update.py -v
"""
import pathlib

import numpy as np
import pytest


def _load_reference():
    """Import hjcdik.base_update WITHOUT requiring the CUDA extension.

    `hjcdik/__init__.py` eagerly does `from ._hjcdik import ...`, so the ordinary package
    import needs a built .so. The reference itself is pure numpy and depends on nothing from
    the robot (J_b sees the joints only through x_i), so on a CPU-only box we load the module
    file directly rather than skip. That is the point of keeping the base mathematics free of
    the kinematics: it stays testable where the kernel is not.
    """
    try:
        from hjcdik import base_update
        return base_update
    except ImportError:
        import importlib.util
        path = pathlib.Path(__file__).resolve().parents[1] / "hjcdik" / "base_update.py"
        spec = importlib.util.spec_from_file_location("_hjcdik_base_update_ref", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


_bu = _load_reference()
BaseUpdateConfig = _bu.BaseUpdateConfig
apply_base_update = _bu.apply_base_update
base_cost = _bu.base_cost
base_jacobian = _bu.base_jacobian
base_normal_equations = _bu.base_normal_equations
base_update_step = _bu.base_update_step
clip_base_step = _bu.clip_base_step
contact_points_world = _bu.contact_points_world
mat_to_quat = _bu.mat_to_quat
quat_mul = _bu.quat_mul
quat_normalize = _bu.quat_normalize
quat_to_mat = _bu.quat_to_mat
skew = _bu.skew
so3_exp = _bu.so3_exp
solve_base_update = _bu.solve_base_update

FD_TOL = 1e-8       # analytic-vs-finite-difference, central differences at h=1e-6 in fp64


# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------

def _rand_state(rng, K=4, spread=0.6):
    """A random (contacts, base) state. Contacts are spread around the base so the lever
    arms (x_i - p_b) are non-degenerate -- degeneracy is tested explicitly elsewhere."""
    c_base = rng.uniform(-spread, spread, (K, 3))
    p_b = rng.uniform(-0.5, 0.5, 3)
    q_b = quat_normalize(rng.normal(size=4))
    return c_base, p_b, q_b


def _fd_base_jacobian(c_base, p_b, q_b, h=1e-6):
    """Central finite differences of x(p_b, q_b) in the 6D tangent of eq. (2).

    This is the independent oracle: it perturbs the STATE through the same exp-map update
    the solver uses and re-runs the composition, touching none of base_jacobian's algebra.
    """
    K = np.asarray(c_base).reshape(-1, 3).shape[0]
    R = quat_to_mat(q_b)
    J = np.zeros((3 * K, 6))
    for d in range(6):
        dp, dphi = np.zeros(3), np.zeros(3)
        (dp if d < 3 else dphi)[d % 3] = h
        qp = mat_to_quat(so3_exp(dphi) @ R)
        qm = mat_to_quat(so3_exp(-dphi) @ R)
        xp = contact_points_world(c_base, p_b + dp, qp)
        xm = contact_points_world(c_base, p_b - dp, qm)
        J[:, d] = ((xp - xm) / (2 * h)).reshape(-1)
    return J


# ----------------------------------------------------------------------------------------
# SO(3) / quaternion primitives
# ----------------------------------------------------------------------------------------

def test_skew_is_the_cross_product():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.normal(size=3), rng.normal(size=3)
        np.testing.assert_allclose(skew(a) @ b, np.cross(a, b), rtol=0, atol=1e-15)


def test_skew_is_antisymmetric():
    rng = np.random.default_rng(1)
    for _ in range(10):
        S = skew(rng.normal(size=3))
        np.testing.assert_allclose(S.T, -S, rtol=0, atol=1e-15)


def test_so3_exp_is_a_rotation_and_matches_the_axis_angle():
    rng = np.random.default_rng(2)
    for _ in range(50):
        w = rng.normal(size=3) * rng.uniform(0, 2.0)
        R = so3_exp(w)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)      # orthogonal
        assert abs(np.linalg.det(R) - 1.0) < 1e-12                       # proper
        th = np.linalg.norm(w)
        if th > 1e-6:                                                    # axis is preserved
            np.testing.assert_allclose(R @ (w / th), w / th, atol=1e-12)
            # trace(R) = 1 + 2 cos(theta)
            assert abs(np.trace(R) - (1 + 2 * np.cos(th))) < 1e-12


def test_so3_exp_small_angle_branch_is_continuous():
    """The 1e-12 branch must not introduce a jump -- repeated tiny updates run through it."""
    for eps in (1e-13, 1e-12, 1e-11, 1e-9):
        w = np.array([eps, 0.0, 0.0])
        np.testing.assert_allclose(so3_exp(w), np.eye(3) + skew(w), atol=1e-18)


def test_quat_mat_round_trip_including_180_degrees():
    """trace -> -1 is where a naive w = sqrt(1+trace)/2 loses all precision."""
    rng = np.random.default_rng(3)
    for _ in range(50):
        q = quat_normalize(rng.normal(size=4))
        np.testing.assert_allclose(quat_to_mat(mat_to_quat(quat_to_mat(q))),
                                   quat_to_mat(q), atol=1e-12)
    for axis in np.eye(3):                      # exact 180-degree rotations
        R = so3_exp(axis * np.pi)
        np.testing.assert_allclose(quat_to_mat(mat_to_quat(R)), R, atol=1e-9)


def test_quaternion_sign_equivalence():
    """q and -q are the SAME rotation. Nothing downstream may distinguish them."""
    rng = np.random.default_rng(4)
    c_base, p_b, q_b = _rand_state(rng)
    np.testing.assert_allclose(quat_to_mat(q_b), quat_to_mat(-q_b), atol=1e-15)
    np.testing.assert_allclose(contact_points_world(c_base, p_b, q_b),
                               contact_points_world(c_base, p_b, -q_b), atol=1e-15)
    tgt = contact_points_world(c_base, p_b, q_b) + 0.05
    a = base_update_step(c_base, tgt, p_b, q_b)
    b = base_update_step(c_base, tgt, p_b, -q_b)
    np.testing.assert_allclose(a["p_b"], b["p_b"], atol=1e-12)
    np.testing.assert_allclose(a["dxi"], b["dxi"], atol=1e-12)
    # the returned quaternions may differ in sign, but must be the same rotation
    np.testing.assert_allclose(quat_to_mat(a["q_b"]), quat_to_mat(b["q_b"]), atol=1e-12)


def test_quat_mul_matches_matrix_composition():
    rng = np.random.default_rng(5)
    for _ in range(20):
        a, b = quat_normalize(rng.normal(size=4)), quat_normalize(rng.normal(size=4))
        np.testing.assert_allclose(quat_to_mat(quat_mul(a, b)),
                                   quat_to_mat(a) @ quat_to_mat(b), atol=1e-12)


def test_degenerate_quaternion_is_rejected():
    with pytest.raises(ValueError, match="degenerate"):
        quat_normalize([0.0, 0.0, 0.0, 0.0])


# ----------------------------------------------------------------------------------------
# The base Jacobian vs finite differences  (the milestone's headline check)
# ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("K", [1, 2, 3, 4])
def test_base_jacobian_matches_finite_differences(K):
    """Eq. (3) against central differences, for 1..4 contacts."""
    rng = np.random.default_rng(100 + K)
    worst = 0.0
    for _ in range(25):
        c_base, p_b, q_b = _rand_state(rng, K=K)
        x = contact_points_world(c_base, p_b, q_b)
        err = np.abs(base_jacobian(x, p_b) - _fd_base_jacobian(c_base, p_b, q_b)).max()
        worst = max(worst, float(err))
    assert worst < FD_TOL, f"K={K}: max |analytic - FD| = {worst:.3e}"


def test_base_jacobian_translation_block_is_identity():
    """dx_i/dp_b = I for every contact -- moving the base moves every contact with it."""
    rng = np.random.default_rng(6)
    c_base, p_b, q_b = _rand_state(rng)
    J = base_jacobian(contact_points_world(c_base, p_b, q_b), p_b)
    for i in range(4):
        np.testing.assert_allclose(J[3 * i:3 * i + 3, 0:3], np.eye(3), atol=1e-15)


def test_base_jacobian_rotation_block_is_the_lever_arm():
    """The rotation column is -[x_i - p_b]x: a contact AT the base origin has none."""
    rng = np.random.default_rng(7)
    p_b, q_b = rng.uniform(-1, 1, 3), quat_normalize(rng.normal(size=4))
    # x = R_b * 0 + p_b = p_b, i.e. the contact sits exactly ON the base origin
    c0 = np.zeros((1, 3))
    x = contact_points_world(c0, p_b, q_b)
    np.testing.assert_allclose(x[0], p_b, atol=1e-12)
    J = base_jacobian(x, p_b)
    np.testing.assert_allclose(J[0:3, 3:6], np.zeros((3, 3)), atol=1e-12)


def test_base_jacobian_is_independent_of_how_contacts_were_produced():
    """J_b depends on the joints ONLY through x_i. This is why the reference never calls FK,
    and why a fixed-base codegen suffices for a floating-base solve."""
    rng = np.random.default_rng(8)
    p_b, q_b = rng.uniform(-1, 1, 3), quat_normalize(rng.normal(size=4))
    cA, cB = rng.uniform(-1, 1, (4, 3)), rng.uniform(-1, 1, (4, 3))     # unrelated "postures"
    xA, xB = (contact_points_world(c, p_b, q_b) for c in (cA, cB))
    for c, x in ((cA, xA), (cB, xB)):
        assert np.abs(base_jacobian(x, p_b) - _fd_base_jacobian(c, p_b, q_b)).max() < FD_TOL


# ----------------------------------------------------------------------------------------
# Residual shapes: translation-only, rotation-only, mixed
# ----------------------------------------------------------------------------------------

def test_translation_only_residual_is_solved_in_one_step():
    """A pure base translation is exactly in the model's span => one step, ~machine zero."""
    rng = np.random.default_rng(9)
    c_base, p_b, q_b = _rand_state(rng)
    shift = np.array([0.03, -0.02, 0.01])
    targets = contact_points_world(c_base, p_b + shift, q_b)
    out = base_update_step(c_base, targets, p_b, q_b,
                           BaseUpdateConfig(damping=1e-12))
    assert out["accepted"]
    np.testing.assert_allclose(out["p_b"], p_b + shift, atol=1e-6)
    assert out["cost_after"] < 1e-12 * max(out["cost_before"], 1e-30) + 1e-18


def test_rotation_only_residual_converges_without_quaternion_drift():
    rng = np.random.default_rng(10)
    c_base, p_b, q_b = _rand_state(rng)
    dphi = np.array([0.0, 0.0, 0.06])
    targets = contact_points_world(c_base, p_b, mat_to_quat(so3_exp(dphi) @ quat_to_mat(q_b)))
    cfg = BaseUpdateConfig(damping=1e-9)
    p, q = p_b, q_b
    c0 = base_cost(contact_points_world(c_base, p, q), targets)
    for _ in range(12):
        out = base_update_step(c_base, targets, p, q, cfg)
        p, q = out["p_b"], out["q_b"]
        assert abs(np.linalg.norm(q) - 1.0) < 1e-12          # no drift, every step
    assert base_cost(contact_points_world(c_base, p, q), targets) < 1e-10 * c0


def test_mixed_translation_and_rotation_residual_converges():
    rng = np.random.default_rng(11)
    c_base, p_b, q_b = _rand_state(rng)
    dphi = np.array([0.02, -0.03, 0.05])
    targets = contact_points_world(c_base, p_b + np.array([0.02, 0.01, -0.03]),
                                   mat_to_quat(so3_exp(dphi) @ quat_to_mat(q_b)))
    cfg = BaseUpdateConfig(damping=1e-9)
    p, q = p_b, q_b
    c0 = base_cost(contact_points_world(c_base, p, q), targets)
    for _ in range(20):
        out = base_update_step(c_base, targets, p, q, cfg)
        p, q = out["p_b"], out["q_b"]
    assert base_cost(contact_points_world(c_base, p, q), targets) < 1e-9 * c0


def test_zero_residual_is_a_no_op():
    """Already on target: the step must be ~0 and must not be 'accepted' into a worse pose."""
    rng = np.random.default_rng(12)
    c_base, p_b, q_b = _rand_state(rng)
    targets = contact_points_world(c_base, p_b, q_b)
    out = base_update_step(c_base, targets, p_b, q_b)
    assert out["cost_before"] == pytest.approx(0.0, abs=1e-24)
    np.testing.assert_allclose(out["dxi"], np.zeros(6), atol=1e-12)
    np.testing.assert_allclose(out["p_b"], p_b, atol=1e-12)
    assert out["cost_after"] <= out["cost_before"] + 1e-18


# ----------------------------------------------------------------------------------------
# Active masks, 1..4 contacts
# ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("mask", [0b0001, 0b0011, 0b0111, 0b1111, 0b1010, 0b1000])
def test_active_mask_reduces_cost_on_active_targets(mask):
    rng = np.random.default_rng(13)
    c_base, p_b, q_b = _rand_state(rng)
    targets = contact_points_world(c_base, p_b + np.array([0.02, -0.01, 0.015]), q_b)
    out = base_update_step(c_base, targets, p_b, q_b,
                           BaseUpdateConfig(damping=1e-6), active_mask=mask)
    assert out["accepted"]
    assert out["cost_after"] < out["cost_before"]


def test_inactive_targets_do_not_influence_the_update():
    """An inactive target is 'don't care': moving it must not change the step at all."""
    rng = np.random.default_rng(14)
    c_base, p_b, q_b = _rand_state(rng)
    targets = contact_points_world(c_base, p_b, q_b) + 0.02
    a = base_update_step(c_base, targets, p_b, q_b, active_mask=0b0011)
    moved = targets.copy()
    moved[2] += 5.0                        # target 2 is inactive -- and now wildly wrong
    moved[3] -= 3.0
    b = base_update_step(c_base, moved, p_b, q_b, active_mask=0b0011)
    np.testing.assert_allclose(a["dxi"], b["dxi"], rtol=0, atol=1e-14)
    np.testing.assert_allclose(a["p_b"], b["p_b"], rtol=0, atol=1e-14)


def test_single_contact_leaves_rotation_about_the_lever_arm_unconstrained():
    """K=1 cannot pin the rotation about the base->contact axis. H is rank-deficient and
    damping must carry it -- there is no 'inactive joint' pin for the base (the base moves
    every target), so this is the case that would NaN without lambda."""
    rng = np.random.default_rng(15)
    c_base, p_b, q_b = _rand_state(rng, K=1)
    x = contact_points_world(c_base, p_b, q_b)
    J = base_jacobian(x, p_b)
    H_undamped, _ = base_normal_equations(J, (x * 0).reshape(-1), damping=0.0)
    assert np.linalg.matrix_rank(H_undamped, tol=1e-9) < 6      # genuinely singular
    _, ok = solve_base_update(H_undamped, np.ones(6))
    assert not ok                                                # Cholesky refuses it
    targets = x + np.array([[0.01, 0.02, -0.015]])
    out = base_update_step(c_base, targets, p_b, q_b, BaseUpdateConfig(damping=1e-6))
    assert not out["numerical_failure"] and out["accepted"]
    assert np.all(np.isfinite(out["dxi"]))
    assert out["cost_after"] < out["cost_before"]


def test_empty_mask_produces_no_motion():
    rng = np.random.default_rng(16)
    c_base, p_b, q_b = _rand_state(rng)
    targets = contact_points_world(c_base, p_b, q_b) + 0.1
    out = base_update_step(c_base, targets, p_b, q_b,
                           BaseUpdateConfig(damping=1e-6), active_mask=0b0000)
    np.testing.assert_allclose(out["p_b"], p_b, atol=1e-12)


def test_mask_above_K_is_rejected():
    rng = np.random.default_rng(17)
    c_base, p_b, q_b = _rand_state(rng)
    with pytest.raises(ValueError, match="above K"):
        base_update_step(c_base, contact_points_world(c_base, p_b, q_b), p_b, q_b,
                         active_mask=0b10000)


# ----------------------------------------------------------------------------------------
# Singular / ill-conditioned arrangements
# ----------------------------------------------------------------------------------------

def test_all_contacts_at_the_base_origin_is_singular_and_survives():
    """Every lever arm zero => the whole rotation block vanishes; H is rank 3."""
    p_b, q_b = np.array([0.1, -0.2, 0.3]), quat_normalize([1.0, 0.0, 0.0, 0.0])
    c_base = np.zeros((4, 3))                        # all contacts AT the base origin
    x = contact_points_world(c_base, p_b, q_b)
    H, _ = base_normal_equations(base_jacobian(x, p_b), np.zeros(12), damping=0.0)
    assert np.linalg.matrix_rank(H, tol=1e-9) == 3
    targets = x + np.array([0.01, 0.0, 0.0])
    out = base_update_step(c_base, targets, p_b, q_b, BaseUpdateConfig(damping=1e-6))
    assert not out["numerical_failure"] and np.all(np.isfinite(out["dxi"]))
    assert out["cost_after"] <= out["cost_before"]


def test_collinear_contacts_lose_rank_about_their_axis():
    """Contacts collinear THROUGH the base leave rotation about that line unconstrained."""
    p_b, q_b = np.zeros(3), quat_normalize([1.0, 0.0, 0.0, 0.0])
    c_base = np.array([[0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0], [0.4, 0, 0]])
    x = contact_points_world(c_base, p_b, q_b)
    H, _ = base_normal_equations(base_jacobian(x, p_b), np.zeros(12), damping=0.0)
    assert np.linalg.matrix_rank(H, tol=1e-9) == 5          # exactly one lost DOF
    out = base_update_step(c_base, x + 0.01, p_b, q_b, BaseUpdateConfig(damping=1e-6))
    assert not out["numerical_failure"] and np.all(np.isfinite(out["dxi"]))


def test_damping_escalates_and_never_returns_nan():
    """A non-SPD H must escalate lambda rather than NaN or invalidate the candidate."""
    p_b, q_b = np.zeros(3), quat_normalize([1.0, 0.0, 0.0, 0.0])
    c_base = np.zeros((2, 3))                                  # rank-3 H
    x = contact_points_world(c_base, p_b, q_b)
    out = base_update_step(c_base, x + 0.02, p_b, q_b,
                           BaseUpdateConfig(damping=0.0, damping_escalation=10.0))
    assert np.all(np.isfinite(out["dxi"])) and np.all(np.isfinite(out["p_b"]))
    assert out["failures"] >= 1 and out["damping"] > 0.0       # it escalated, then solved


def test_nan_input_cannot_silently_produce_a_step():
    p_b, q_b = np.zeros(3), quat_normalize([1.0, 0.0, 0.0, 0.0])
    c_base = np.array([[0.3, 0.1, 0.0], [0.2, -0.1, 0.1], [0.1, 0.2, 0.0], [0.0, 0.1, 0.2]])
    targets = contact_points_world(c_base, p_b, q_b)
    targets[1, 0] = np.nan
    out = base_update_step(c_base, targets, p_b, q_b, BaseUpdateConfig(damping=1e-6))
    assert out["numerical_failure"] or not out["accepted"]
    assert np.all(np.isfinite(out["p_b"])) and np.all(np.isfinite(out["q_b"]))


# ----------------------------------------------------------------------------------------
# Clipping, bounds, acceptance
# ----------------------------------------------------------------------------------------

def test_large_residual_is_clipped_and_preserves_direction():
    rng = np.random.default_rng(18)
    c_base, p_b, q_b = _rand_state(rng)
    targets = contact_points_world(c_base, p_b + np.array([2.0, 0.0, 0.0]), q_b)  # 2 m away
    cfg = BaseUpdateConfig(damping=1e-9, max_translation_step=0.05, max_rotation_step=0.10)
    out = base_update_step(c_base, targets, p_b, q_b, cfg)
    assert out["clipped"]
    assert np.linalg.norm(out["dxi"][0:3]) <= 0.05 + 1e-12
    assert np.linalg.norm(out["dxi"][3:6]) <= 0.10 + 1e-12
    assert np.linalg.norm(out["p_b"] - p_b) <= 0.05 + 1e-12
    assert out["accepted"] and out["cost_after"] < out["cost_before"]   # still progress


def test_clip_scales_the_block_rather_than_its_components():
    """Clipping components independently would rotate the step; scaling preserves it."""
    dxi = np.array([0.3, 0.4, 0.0, 0.6, 0.8, 0.0])
    out = clip_base_step(dxi, 0.05, 0.10)
    np.testing.assert_allclose(np.linalg.norm(out[0:3]), 0.05, atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(out[3:6]), 0.10, atol=1e-15)
    np.testing.assert_allclose(out[0:3] / np.linalg.norm(out[0:3]),
                               dxi[0:3] / np.linalg.norm(dxi[0:3]), atol=1e-15)


def test_a_step_within_the_clip_is_untouched():
    dxi = np.array([0.01, 0.0, 0.0, 0.02, 0.0, 0.0])
    np.testing.assert_allclose(clip_base_step(dxi, 0.05, 0.10), dxi, rtol=0, atol=0)


def test_base_position_bounds_are_enforced():
    rng = np.random.default_rng(19)
    c_base, p_b, q_b = _rand_state(rng)
    p_b = np.array([0.0, 0.0, 0.0])
    targets = contact_points_world(c_base, p_b + np.array([1.0, 1.0, 1.0]), q_b)
    lo, hi = np.array([-0.02, -0.02, -0.02]), np.array([0.02, 0.02, 0.02])
    cfg = BaseUpdateConfig(damping=1e-9, position_lower=lo, position_upper=hi)
    p, q = p_b, q_b
    for _ in range(10):
        out = base_update_step(c_base, targets, p, q, cfg)
        p, q = out["p_b"], out["q_b"]
        assert np.all(p >= lo - 1e-12) and np.all(p <= hi + 1e-12)


def test_inverted_bounds_are_rejected():
    with pytest.raises(ValueError, match="inverted"):
        apply_base_update(np.zeros(3), [1, 0, 0, 0], np.ones(6),
                          position_lower=[1, 1, 1], position_upper=[0, 0, 0])


def test_update_is_rejected_when_it_increases_cost():
    """An enormous step_scale overshoots; accept_only_on_improvement must catch it."""
    rng = np.random.default_rng(20)
    c_base, p_b, q_b = _rand_state(rng)
    targets = contact_points_world(c_base, p_b, q_b) + 0.01
    cfg = BaseUpdateConfig(damping=1e-12, step_scale=60.0,
                           max_translation_step=1e9, max_rotation_step=1e9)
    out = base_update_step(c_base, targets, p_b, q_b, cfg)
    assert not out["accepted"]
    np.testing.assert_allclose(out["p_b"], p_b, atol=1e-15)          # rolled back
    assert out["cost_after"] == out["cost_before"]


def test_acceptance_can_be_disabled():
    rng = np.random.default_rng(21)
    c_base, p_b, q_b = _rand_state(rng)
    targets = contact_points_world(c_base, p_b, q_b) + 0.01
    cfg = BaseUpdateConfig(damping=1e-12, step_scale=60.0, max_translation_step=1e9,
                           max_rotation_step=1e9, accept_only_on_improvement=False)
    assert base_update_step(c_base, targets, p_b, q_b, cfg)["accepted"]


def test_repeated_updates_preserve_unit_norm_and_reduce_cost():
    """200 steps: quaternion drift compounds, so renormalizing every step is load-bearing."""
    rng = np.random.default_rng(22)
    c_base, p_b, q_b = _rand_state(rng)
    dphi = np.array([0.05, -0.04, 0.03])
    targets = contact_points_world(c_base, p_b + np.array([0.04, -0.02, 0.03]),
                                   mat_to_quat(so3_exp(dphi) @ quat_to_mat(q_b)))
    cfg = BaseUpdateConfig(damping=1e-8)
    p, q = p_b, q_b
    prev = base_cost(contact_points_world(c_base, p, q), targets)
    c0 = prev
    for _ in range(200):
        out = base_update_step(c_base, targets, p, q, cfg)
        p, q = out["p_b"], out["q_b"]
        assert abs(np.linalg.norm(q) - 1.0) < 1e-12
        now = base_cost(contact_points_world(c_base, p, q), targets)
        assert now <= prev + 1e-18                     # monotone: never uphill
        prev = now
    assert prev < 1e-10 * c0


def test_damping_shrinks_the_step():
    """More damping => a shorter, more conservative step. The knob must actually do that."""
    rng = np.random.default_rng(23)
    c_base, p_b, q_b = _rand_state(rng)
    targets = contact_points_world(c_base, p_b + np.array([0.05, 0, 0]), q_b)
    norms = [np.linalg.norm(base_update_step(
        c_base, targets, p_b, q_b,
        BaseUpdateConfig(damping=d, max_translation_step=1e9,
                         max_rotation_step=1e9))["dxi"]) for d in (1e-9, 1e-1, 1e1, 1e3)]
    assert all(a > b for a, b in zip(norms, norms[1:])), norms


def test_weights_are_honoured():
    """Up-weighting a target must make the solution serve THAT target better.

    Requires an OVERDETERMINED, inconsistent system: W only decides which target absorbs
    error that no rigid base motion can remove. At K>=3 the 9-12 equations exceed the base's
    6 DOF (rank 6), so the leftover has to go somewhere and W chooses where.
    """
    rng = np.random.default_rng(24)
    c_base, p_b, q_b = _rand_state(rng)
    x = contact_points_world(c_base, p_b, q_b)
    # per-target random displacements: 12 equations, rank(J_b) = 6, and no rigid base motion
    # satisfies all four at once -- see test_base_jacobian_rank_structure.
    targets = x + rng.normal(0.0, 0.02, (4, 3))

    def err(out, k):
        moved = contact_points_world(c_base, out["p_b"], out["q_b"])[k]
        return float(np.linalg.norm(targets[k] - moved))

    cfg = BaseUpdateConfig(damping=1e-9, max_translation_step=1e9, max_rotation_step=1e9)
    equal = base_update_step(c_base, targets, p_b, q_b, cfg, weights=[1, 1, 1, 1])
    heavy0 = base_update_step(c_base, targets, p_b, q_b, cfg, weights=[1000, 1, 1, 1])
    assert err(heavy0, 0) < err(equal, 0)              # target 0 is served better ...
    assert err(heavy0, 1) > err(equal, 1)              # ... at the others' expense
    with pytest.raises(ValueError, match="non-negative"):
        base_update_step(c_base, targets, p_b, q_b, weights=[-1, 1, 1, 1])


@pytest.mark.parametrize("K,expected_rank", [(1, 3), (2, 5), (3, 6), (4, 6)])
def test_base_jacobian_rank_structure(K, expected_rank):
    """How many of the base's 6 DOF the contacts actually constrain. This is WHY damping is
    not optional, and it is not intuitive:

      K=1 -> rank 3: position pins translation; nothing pins rotation.
      K=2 -> rank 5: SIX equations but only rank five -- rotation about the LINE through the
             two contacts moves neither of them. So a 'both hands' or 'both feet' mask is
             rank-deficient for the base and cannot be solved without lambda. (The kernel
             records an analogous joint-side trap at hjcd_kernel.cu:1649-1657 -- a different
             mechanism, same moral: the two-target masks are the degenerate ones.)
      K>=3 non-collinear -> rank 6: fully constrained, and overdetermined, which is the only
             regime where the target weights can trade anything off.
    """
    rng = np.random.default_rng(300 + K)
    for _ in range(50):
        c_base, p_b, q_b = _rand_state(rng, K=K)
        J = base_jacobian(contact_points_world(c_base, p_b, q_b), p_b)
        assert np.linalg.matrix_rank(J, tol=1e-9) == expected_rank


def test_two_contacts_leave_rotation_about_their_own_line_free():
    """The exact null direction at K=2, named: the axis through both contacts."""
    p_b, q_b = np.zeros(3), quat_normalize([1.0, 0.0, 0.0, 0.0])
    c_base = np.array([[0.2, 0.3, 0.1], [0.5, 0.3, 0.1]])       # separated along +x
    x = contact_points_world(c_base, p_b, q_b)
    J = base_jacobian(x, p_b)
    assert np.linalg.matrix_rank(J, tol=1e-9) == 5
    # rotating about the line through both contacts moves neither: build that twist and
    # check it lies in the nullspace. axis = x2 - x1 (through the contacts).
    axis = x[1] - x[0]
    axis /= np.linalg.norm(axis)
    # a rotation about a line through point x0 with direction `axis` is
    # dphi = axis, dp = -axis x (x0 - p_b) ... i.e. the twist fixing that line
    dxi = np.concatenate([np.cross(x[0] - p_b, axis), axis])
    np.testing.assert_allclose(J @ dxi, np.zeros(6), atol=1e-12)


# ----------------------------------------------------------------------------------------
# GPU: the reference is wired to the real robot
# ----------------------------------------------------------------------------------------

def _hjcdik_g1_or_skip():
    hj = pytest.importorskip("hjcdik", reason="needs the built extension (CUDA)")
    if not hasattr(hj, "num_targets") or hj.num_joints() != 29 or hj.num_targets() != 4:
        pytest.skip(f"needs a G1 build (num_joints=29, num_targets=4); this build reports "
                    f"num_joints={hj.num_joints()}")
    return hj


@pytest.mark.gpu_proof
def test_real_g1_fk_composes_with_the_base_as_the_reference_claims():
    """Eq. (1) against the real HJCD G1 FK: the base really is a rigid transform on top of
    the FIXED-base kinematics, which is the premise the whole architecture rests on."""
    hj = _hjcdik_g1_or_skip()
    rng = np.random.default_rng(0)
    lim = np.asarray(hj.joint_limits())
    worst = 0.0
    for _ in range(8):
        q_j = rng.uniform(lim[:, 0], lim[:, 1])
        p_b = rng.uniform(-0.5, 0.5, 3)
        q_b = quat_normalize(rng.normal(size=4))
        c_base = np.asarray(hj.target_transforms(q_j[None, :])[0])[:, :3, 3]   # [K,3]
        x_ref = contact_points_world(c_base, p_b, q_b)
        err = np.abs(base_jacobian(x_ref, p_b) -
                     _fd_base_jacobian(c_base, p_b, q_b)).max()
        worst = max(worst, float(err))
    assert worst < FD_TOL, f"max |analytic - FD| on real G1 contacts = {worst:.3e}"


@pytest.mark.gpu_proof
def test_base_update_reduces_error_on_a_real_g1_four_contact_task():
    """A reachable 4-contact task displaced by a known base offset must be recovered."""
    hj = _hjcdik_g1_or_skip()
    rng = np.random.default_rng(1)
    lim = np.asarray(hj.joint_limits())
    q_j = rng.uniform(lim[:, 0], lim[:, 1])
    c_base = np.asarray(hj.target_transforms(q_j[None, :])[0])[:, :3, 3]
    p_true, q_true = np.array([0.05, -0.03, 0.02]), mat_to_quat(so3_exp([0.0, 0.0, 0.04]))
    targets = contact_points_world(c_base, p_true, q_true)          # reachable BY CONSTRUCTION

    p, q = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
    c0 = base_cost(contact_points_world(c_base, p, q), targets)
    for _ in range(30):
        out = base_update_step(c_base, targets, p, q, BaseUpdateConfig(damping=1e-9))
        p, q = out["p_b"], out["q_b"]
    assert base_cost(contact_points_world(c_base, p, q), targets) < 1e-10 * c0
    np.testing.assert_allclose(p, p_true, atol=1e-4)
    np.testing.assert_allclose(quat_to_mat(q), quat_to_mat(q_true), atol=1e-4)
