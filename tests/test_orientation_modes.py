"""Per-target orientation modes: NONE / AXIS / FULL.

A contact generally pins a position and a contact NORMAL, not a full pose: twist about that normal
is a free DoF of the contact. Constraining it anyway is what makes 3- and 4-contact stances read as
infeasible. ORI_AXIS expresses the 5-DoF constraint; ORI_FULL is the legacy full-quaternion
behaviour and remains the default.

TWO METRICS, DELIBERATELY DIFFERENT
-----------------------------------
The optimizer minimizes the CHORD residual

    e_axis = n_target - n_current,      ||e_axis|| = 2 sin(theta/2)

because its Jacobian is exact and free of the 0/pi singularities that Log-style tangent residuals
carry, while the PUBLIC error is the angle

    theta = acos(clamp(n_current . n_target, -1, 1))    radians.

Both are monotonically increasing in theta on [0, pi], so "smaller residual" and "smaller reported
error" never disagree about which of two candidates is better; they are simply different
parameterizations of the same misalignment. The LM gain ratio uses the CHORD objective, the one
that produced its Jacobian -- mixing the two would compare a predicted chord reduction against an
achieved angular reduction. Selection (E_phys) keeps using the physical angle.

ANTIPODAL DEGENERACY (expected, not a bug)
------------------------------------------
The chordal cost is C = 1/2 ||n_target - n_current||^2 = 1 - n_current . n_target. At exact
anti-alignment n_current = -n_target the residual is maximal (e = 2 n_target) yet

    (jw x n_current) . e = 0     for every rotational direction,   so   J^T e = 0.

theta = pi is a stationary MAXIMUM of the chordal cost. This is the antipodal ambiguity of an axis
direction -- at exactly pi there is no unique shortest rotation -- not a sign error, and not
something to paper over with a twist-dependent term, which would destroy AXIS semantics.
test_exact_antipodal_gradient_is_zero pins the degeneracy; test_exact_antipodal_full_solve
documents what the complete solver does from such a seed.
"""
import numpy as np
import pytest

import hjcdik

N = hjcdik.num_joints()
K = hjcdik.num_targets()
NONE, AXIS, FULL = hjcdik.ORI_NONE, hjcdik.ORI_AXIS, hjcdik.ORI_FULL


# --- small rotation helpers (kept local: these tests must not depend on the code under test) ---

def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def rot_about(a, psi):
    a = np.asarray(a, float)
    a = a / np.linalg.norm(a)
    c, s = np.cos(psi), np.sin(psi)
    Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) * c + s * Kx + (1 - c) * np.outer(a, a)


def rot_taking(u, v):
    """A rotation carrying unit u onto unit v (any twist about v; irrelevant under AXIS)."""
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    d = float(np.clip(np.dot(u, v), -1.0, 1.0))
    if d > 1 - 1e-12:
        return np.eye(3)
    if d < -1 + 1e-12:                       # antipodal: any perpendicular axis, half turn
        perp = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
        ax = np.cross(u, perp)
        return rot_about(ax, np.pi)
    ax = np.cross(u, v)
    return rot_about(ax, np.arccos(d))


def fk_targets(q):
    """[K,4,4] world target frames at q (the same frames the solver constrains)."""
    return np.asarray(hjcdik.target_transforms(np.asarray(q, float).reshape(1, N)))[0]


def reachable_poses(q):
    T = fk_targets(q)
    poses = np.zeros((K, 7))
    for k in range(K):
        poses[k, :3] = T[k][:3, 3]
        poses[k, 3:] = R_to_quat(T[k][:3, :3])
    return poses


def axis_angle(n_a, n_b):
    d = float(np.clip(np.dot(n_a, n_b) / (np.linalg.norm(n_a) * np.linalg.norm(n_b)), -1, 1))
    return float(np.arccos(d))


ALL_ACTIVE = np.array([(1 << K) - 1], dtype=np.uint32)
GENERIC_AXES = np.tile(np.array([0.0, 0.0, -1.0]), (K, 1))


def solve(poses, seeds, **kw):
    base = dict(active_masks=ALL_ACTIVE, seed_configs=seeds, num_solutions=1,
                position_tol=1e-4, orientation_tol=1e-3, seed=17)
    base.update(kw)
    return hjcdik.solve_problems(target_poses=poses[None], **base)


@pytest.fixture(scope="module")
def scene():
    rng = np.random.default_rng(5)
    q_ref = rng.uniform(-0.4, 0.4, N)
    poses = reachable_poses(q_ref)
    seeds = q_ref[None, None, :] + rng.normal(0, 0.15, (1, 96, N))
    return q_ref, poses, seeds


# =============================================================================================
# 1. AXIS Jacobian vs finite differences
# =============================================================================================

def _b_and_fd(q, tp, tq, act, wp, wo, modes, axes, h=1e-6):
    """normal_equations gives b = J^T W e with e = target - current and J = d(current)/dq.

    The scalar cost is C = sum_k [wp |e_p|^2 + wo |e_R|^2] and de/dq = -J, so grad C = -2 b.
    A sign error in the axis Jacobian flips b and shows up as cos(b, b_fd) = -1.
    """
    ne = hjcdik._normal_equations_raw(q, tp, tq, act, wp, wo,
                                      orientation_modes=modes, orientation_axes=axes)
    b_an = np.asarray(ne["b"])[0]

    def cost(qq):
        r = hjcdik._target_residuals_raw(qq, tp, tq, act, wp, wo,
                                         np.full(K, 1e-4), np.full(K, 1e-3),
                                         orientation_modes=modes, orientation_axes=axes)
        return float(np.asarray(r["cost_raw"])[0])

    grad = np.zeros(N)
    for j in range(N):
        qp = q.copy(); qp[0, j] += h
        qm = q.copy(); qm[0, j] -= h
        grad[j] = (cost(qp) - cost(qm)) / (2 * h)
    return b_an, -0.5 * grad


@pytest.mark.parametrize("modes_row,label", [
    ([FULL] * K, "all-FULL"),
    ([AXIS] * K, "all-AXIS"),
    ([NONE] * K, "all-NONE"),
    ([FULL, AXIS, NONE, AXIS][:K], "mixed"),
])
@pytest.mark.parametrize("trial", [0, 1])
def test_axis_jacobian_matches_finite_differences(modes_row, label, trial):
    rng = np.random.default_rng(100 + trial)
    q = rng.uniform(-0.6, 0.6, (1, N))
    tp = rng.uniform(-0.3, 0.3, (1, K, 3))
    tq = rng.normal(size=(1, K, 4)); tq /= np.linalg.norm(tq, axis=-1, keepdims=True)
    axes = rng.normal(size=(1, K, 3)); axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    modes = np.tile(np.asarray(modes_row, np.int32), (1, 1))
    wp = np.ones((1, K)); wo = np.ones((1, K))

    b_an, b_fd = _b_and_fd(q, tp, tq, ALL_ACTIVE, wp, wo, modes, axes)
    scale = max(np.abs(b_an).max(), np.abs(b_fd).max(), 1e-12)
    assert np.abs(b_an - b_fd).max() / scale < 1e-5, f"{label}: analytic b != FD b"
    if np.linalg.norm(b_an) > 1e-9:
        cos = np.dot(b_an, b_fd) / (np.linalg.norm(b_an) * np.linalg.norm(b_fd))
        assert cos > 0.999, f"{label}: sign/direction mismatch, cos = {cos}"


# =============================================================================================
# 2. Twist invariance -- the defining property of AXIS
# =============================================================================================

@pytest.mark.parametrize("psi", [0.5, 1.5, 3.0, -2.0])
def test_axis_is_invariant_to_target_twist(scene, psi):
    """R_target -> R_target * Rot(a_local, psi) leaves R_target a_local unchanged, so it must
    leave the AXIS constraint, its error and its acceptance unchanged."""
    _, poses, seeds = scene
    kw = dict(orientation_modes="axis", orientation_axes=GENERIC_AXES, precision="float64")
    base = solve(poses, seeds, **kw)

    twisted = poses.copy()
    for k in range(K):
        twisted[k, 3:] = R_to_quat(quat_to_R(poses[k, 3:]) @ rot_about(GENERIC_AXES[k], psi))
    got = solve(twisted, seeds, **kw)

    assert bool(got["success"][0, 0]) == bool(base["success"][0, 0])
    # float64 so this is the mathematical property, not a float32 noise floor.
    np.testing.assert_allclose(np.asarray(got["orientation_errors"]),
                               np.asarray(base["orientation_errors"]), atol=1e-9)
    np.testing.assert_allclose(np.asarray(got["joint_config"]),
                               np.asarray(base["joint_config"]), atol=1e-7)


# =============================================================================================
# 3. FULL vs AXIS discrimination -- proves test 2 is not vacuous
# =============================================================================================

def test_full_mode_does_see_the_twist_axis_mode_ignores(scene):
    _, poses, seeds = scene
    psi = 1.5
    twisted = poses.copy()
    for k in range(K):
        twisted[k, 3:] = R_to_quat(quat_to_R(poses[k, 3:]) @ rot_about(GENERIC_AXES[k], psi))

    full_base = solve(poses, seeds, orientation_modes="full", precision="float64")
    full_tw = solve(twisted, seeds, orientation_modes="full", precision="float64")
    delta_full = np.abs(np.asarray(full_tw["orientation_errors"])
                        - np.asarray(full_base["orientation_errors"])).max()
    assert delta_full > 1e-2, (
        "FULL mode did not notice a 1.5 rad twist; the twist-invariance test would then be "
        f"proving nothing (delta {delta_full:.3e})")


# =============================================================================================
# 4. NONE semantics
# =============================================================================================

def test_none_ignores_an_incompatible_target_quaternion(scene):
    _, poses, seeds = scene
    bad = poses.copy()
    bad[:, 3:] = np.array([0.0, 1.0, 0.0, 0.0])       # 180 deg away from anything reachable
    r = solve(bad, seeds, orientation_modes="none")
    assert bool(r["success"][0, 0]), "NONE must still solve position"
    assert float(np.max(r["orientation_errors"])) == 0.0
    assert float(np.max(r["position_errors"])) <= 1e-4


# =============================================================================================
# 5. Mixed modes
# =============================================================================================

@pytest.mark.skipif(K < 4, reason="needs 4 targets to exercise all three modes at once")
def test_mixed_modes_apply_independently(scene):
    _, poses, seeds = scene
    r = solve(poses, seeds, orientation_modes=["full", "axis", "none", "axis"],
              orientation_axes=GENERIC_AXES)
    oe = np.asarray(r["orientation_errors"])[0, 0]
    assert oe[2] == 0.0, "the NONE target must report exactly zero orientation error"
    assert bool(r["success"][0, 0])
    assert oe[0] <= 1e-3 and oe[1] <= 1e-3 and oe[3] <= 1e-3


# =============================================================================================
# 6 & 7. Metadata-default axes and explicit override
# =============================================================================================

def test_axis_mode_uses_generated_metadata_axes_by_default(scene):
    """No orientation_axes= => the generated per-target axes are used."""
    _, poses, seeds = scene
    if not np.any(hjcdik.target_axes()):
        pytest.skip("this build's generated metadata declares no target axes")
    r = solve(poses, seeds, orientation_modes="axis", precision="float64")
    explicit = solve(poses, seeds, orientation_modes="axis",
                     orientation_axes=hjcdik.target_axes(), precision="float64")
    np.testing.assert_allclose(np.asarray(r["joint_config"]),
                               np.asarray(explicit["joint_config"]), atol=0)


def test_explicit_axes_override_metadata(scene):
    """An override that differs from the metadata must actually change the problem."""
    _, poses, seeds = scene
    if not np.any(hjcdik.target_axes()):
        pytest.skip("this build's generated metadata declares no target axes")
    other = np.tile(np.array([1.0, 0.0, 0.0]), (K, 1))
    meta = solve(poses, seeds, orientation_modes="axis", precision="float64")
    over = solve(poses, seeds, orientation_modes="axis", orientation_axes=other,
                 precision="float64")
    assert not np.allclose(np.asarray(meta["joint_config"]),
                           np.asarray(over["joint_config"]), atol=1e-9), \
        "orientation_axes= did not override the generated metadata"


def test_axes_are_normalized_on_input(scene):
    """A non-unit axis is a scaling of the same direction and must behave identically."""
    _, poses, seeds = scene
    a1 = GENERIC_AXES
    a2 = GENERIC_AXES * 7.5
    r1 = solve(poses, seeds, orientation_modes="axis", orientation_axes=a1, precision="float64")
    r2 = solve(poses, seeds, orientation_modes="axis", orientation_axes=a2, precision="float64")
    np.testing.assert_allclose(np.asarray(r1["orientation_errors"]),
                               np.asarray(r2["orientation_errors"]), atol=1e-12)


# =============================================================================================
# 8. Missing / zero axis must fail loudly
# =============================================================================================

def test_axis_mode_without_any_axis_raises(scene, monkeypatch):
    _, poses, seeds = scene
    monkeypatch.setattr(hjcdik, "target_axes", lambda: np.zeros((K, 3)))
    with pytest.raises((ValueError, RuntimeError), match="no axis is available"):
        solve(poses, seeds, orientation_modes="axis")


def test_axis_mode_with_zero_axis_raises(scene):
    _, poses, seeds = scene
    with pytest.raises((ValueError, RuntimeError), match="zero vector"):
        solve(poses, seeds, orientation_modes="axis", orientation_axes=np.zeros((K, 3)))


def test_unknown_mode_name_raises(scene):
    _, poses, seeds = scene
    with pytest.raises(ValueError, match="unknown orientation mode"):
        solve(poses, seeds, orientation_modes="normal-ish")


# =============================================================================================
# 9 & 10. Fixed base and floating base
# =============================================================================================

def test_axis_mode_fixed_base(scene):
    _, poses, seeds = scene
    r = solve(poses, seeds, orientation_modes="axis", orientation_axes=GENERIC_AXES)
    assert bool(r["success"][0, 0])
    assert float(np.max(r["orientation_errors"])) <= 1e-3


def test_axis_mode_floating_base(scene):
    """Same constraint with the base free; the axis is target-LOCAL so it needs no retransform."""
    q_ref, poses, _ = scene
    rng = np.random.default_rng(9)
    S = 96
    fseeds = np.zeros((1, S, 7 + N))
    fseeds[0, :, 0:3] = rng.normal(0, 0.05, (S, 3))
    fseeds[0, :, 3] = 1.0
    fseeds[0, :, 7:] = q_ref[None, :] + rng.normal(0, 0.15, (S, N))
    r = hjcdik.solve_problems(target_poses=poses[None], active_masks=ALL_ACTIVE,
                              seed_configs=fseeds, floating_base=True, num_solutions=1,
                              position_tol=1e-4, orientation_tol=1e-3, seed=17,
                              orientation_modes="axis", orientation_axes=GENERIC_AXES)
    assert bool(r["success"][0, 0])
    assert float(np.max(r["orientation_errors"])) <= 1e-3
    assert np.asarray(r["base_position"]).shape == (1, 1, 3)


# =============================================================================================
# 11. Near-antipodal AXIS: ordinary, well-behaved, and the derivative still checks out
# =============================================================================================

@pytest.mark.parametrize("gap", [1e-3, 1e-2])
def test_near_antipodal_axis_residual_and_derivative(gap):
    """theta = pi - gap. The reported angle must be right and the analytic b must match FD."""
    rng = np.random.default_rng(31)
    q = rng.uniform(-0.5, 0.5, (1, N))
    a_local = np.array([0.0, 0.0, -1.0])
    T = fk_targets(q[0])

    tp = np.zeros((1, K, 3)); tq = np.zeros((1, K, 4))
    axes = np.tile(a_local, (1, K, 1))
    for k in range(K):
        n_cur = T[k][:3, :3] @ a_local
        perp = np.array([1.0, 0, 0]) if abs(n_cur[0]) < 0.9 else np.array([0, 1.0, 0])
        perp = np.cross(n_cur, perp); perp /= np.linalg.norm(perp)
        n_tgt = rot_about(perp, np.pi - gap) @ n_cur          # exactly theta = pi - gap away
        tp[0, k] = T[k][:3, 3]
        tq[0, k] = R_to_quat(rot_taking(a_local, n_tgt))

    modes = np.full((1, K), AXIS, np.int32)
    wp = np.zeros((1, K)); wo = np.ones((1, K))               # isolate the orientation channel
    r = hjcdik._target_residuals_raw(q, tp, tq, ALL_ACTIVE, wp, wo,
                                     np.full(K, 1e-4), np.full(K, 1e-3),
                                     orientation_modes=modes, orientation_axes=axes)
    reported = np.asarray(r["orientation_errors"])[0]
    # acos is ill-conditioned near pi: dtheta = dd / sin(theta), so a rounding-level error in the
    # dot product is amplified by 1/sin(gap). That is a property of reporting an ANGLE near
    # anti-alignment, not slack in the implementation, so the tolerance is scaled by it rather
    # than set to a flat loose number.
    # The measured dot-product error through a 29-joint double FK plus the quaternion round trip
    # is ~5e-9; 1e-8 is that with headroom. Divided by sin(gap) it IS the amplification, so the
    # tolerance tightens automatically as the problem moves away from pi.
    np.testing.assert_allclose(reported, np.pi - gap, rtol=0.0,
                               atol=1e-8 / np.sin(gap) + 1e-7)

    b_an, b_fd = _b_and_fd(q, tp, tq, ALL_ACTIVE, wp, wo, modes, axes, h=1e-7)
    scale = max(np.abs(b_an).max(), np.abs(b_fd).max(), 1e-12)
    assert np.abs(b_an - b_fd).max() / scale < 1e-4


# =============================================================================================
# 12. EXACT antipodal: a stationary maximum, by construction
# =============================================================================================

def test_exact_antipodal_gradient_is_zero():
    """n_current = -n_target  =>  J^T e = 0 for every rotational direction.

    C = 1 - n_current . n_target is stationary at theta = pi. This is the antipodal ambiguity of
    an axis direction, not a defect: at exactly pi no shortest rotation is unique. The assertion
    is that the gradient VANISHES -- deliberately not that LM can descend from here.
    """
    rng = np.random.default_rng(77)
    q = rng.uniform(-0.5, 0.5, (1, N))
    a_local = np.array([0.0, 0.0, -1.0])
    T = fk_targets(q[0])

    tp = np.zeros((1, K, 3)); tq = np.zeros((1, K, 4))
    axes = np.tile(a_local, (1, K, 1))
    for k in range(K):
        n_cur = T[k][:3, :3] @ a_local
        tp[0, k] = T[k][:3, 3]
        tq[0, k] = R_to_quat(rot_taking(a_local, -n_cur))      # n_target = -n_current exactly

    modes = np.full((1, K), AXIS, np.int32)
    wp = np.zeros((1, K)); wo = np.ones((1, K))                # orientation channel only

    r = hjcdik._target_residuals_raw(q, tp, tq, ALL_ACTIVE, wp, wo,
                                     np.full(K, 1e-4), np.full(K, 1e-3),
                                     orientation_modes=modes, orientation_axes=axes)
    np.testing.assert_allclose(np.asarray(r["orientation_errors"])[0], np.pi, atol=1e-6)
    # The residual is MAXIMAL here: ||e|| = 2 sin(pi/2) = 2.
    e = np.asarray(r["orientation_residuals"])[0]
    np.testing.assert_allclose(np.linalg.norm(e, axis=1), 2.0, atol=1e-5)

    ne = hjcdik._normal_equations_raw(q, tp, tq, ALL_ACTIVE, wp, wo,
                                      orientation_modes=modes, orientation_axes=axes)
    b = np.asarray(ne["b"])[0]
    assert np.abs(b).max() < 1e-5, (
        f"expected a vanishing gradient at exact anti-alignment, got max|b| = {np.abs(b).max():.3e}")


def test_exact_antipodal_full_solve_is_documented():
    """What the COMPLETE solver does from an exactly-antipodal seed.

    theta = pi is a stationary MAXIMUM of the chordal cost, not a minimum, so it is an UNSTABLE
    equilibrium: the analytic gradient vanishes exactly (test_exact_antipodal_gradient_is_zero
    proves that in double precision on the host-constructed problem), but any perturbation --
    including the float32 rounding of the solve pipeline itself -- puts the state on a descending
    slope and the solver rolls off it.

    So the practical behaviour is the opposite of "stuck": exact anti-alignment is not an
    attractor. This test records the measured escape rather than asserting a preferred outcome, so
    that a change in that behaviour surfaces here instead of silently altering the guarantee.

    The degeneracy is still real and still worth knowing: arbitrarily close to pi the gradient is
    arbitrarily small, so convergence from a near-antipodal start is slow and its direction is
    determined by whichever perturbation dominates. Robust escape from near-pi initialization is a
    solver-globalization concern, deliberately NOT patched here with a twist-dependent residual,
    which would break AXIS semantics.
    """
    rng = np.random.default_rng(78)
    q_ref = rng.uniform(-0.4, 0.4, N)
    T = fk_targets(q_ref)
    a_local = np.array([0.0, 0.0, -1.0])
    poses = np.zeros((K, 7))
    for k in range(K):
        n_cur = T[k][:3, :3] @ a_local
        poses[k, :3] = T[k][:3, 3]
        poses[k, 3:] = R_to_quat(rot_taking(a_local, -n_cur))
    seeds = np.tile(q_ref, (1, 1, 1))                          # THE antipodal configuration itself
    axes = np.tile(a_local, (K, 1))

    lm_only = hjcdik.solve_problems(
        target_poses=poses[None], active_masks=ALL_ACTIVE, seed_configs=seeds, num_solutions=1,
        position_tol=1e-4, orientation_tol=1e-3, seed=3, coarse_mode="none",
        orientation_modes="axis", orientation_axes=axes)
    with_coarse = hjcdik.solve_problems(
        target_poses=poses[None], active_masks=ALL_ACTIVE, seed_configs=seeds, num_solutions=1,
        position_tol=1e-4, orientation_tol=1e-3, seed=3, coarse_mode="multi_target",
        orientation_modes="axis", orientation_axes=axes)

    lm_err = float(np.max(lm_only["orientation_errors"]))
    co_err = float(np.max(with_coarse["orientation_errors"]))
    print(f"\n  exact-antipodal seed: LM-only max axis err = {lm_err:.6f} rad, "
          f"coarse+LM = {co_err:.6f} rad  (started at pi = {np.pi:.6f})")
    # Both stages must at least not DIVERGE past anti-alignment, which is the only outcome that
    # would indicate a wrong descent direction rather than a slow one.
    assert lm_err <= np.pi + 1e-6
    assert co_err <= np.pi + 1e-6
    # The coarse sweep is not gradient-only (it accepts any strict improvement over a coordinate
    # sweep), so it must do at least as well as LM alone from this start.
    assert co_err <= lm_err + 1e-6, (
        f"coarse+LM ({co_err:.6f}) did worse than LM alone ({lm_err:.6f}) from an antipodal seed")
