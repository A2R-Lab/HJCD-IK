"""M4: the GPU alternating base update (refinement only).

Every test here is GPU + G1 (K=4). The base MATHEMATICS is covered CPU-only in
tests/test_base_update.py against the host reference; this file covers the things only the kernel
can get wrong: coupled best-state snapshotting, exact rollback, candidate isolation, and the
fixed-base bit-identity guarantee.
"""
import numpy as np
import pytest

hjcdik = pytest.importorskip("hjcdik", reason="needs the built CUDA extension")

from hjcdik.base_update import (contact_points_world, mat_to_quat, quat_to_mat,  # noqa: E402
                                so3_exp, quat_normalize)

N, K = (hjcdik.num_joints(), hjcdik.num_targets()) if hasattr(hjcdik, "num_targets") else (0, 0)
pytestmark = pytest.mark.skipif(K != 4, reason=f"needs a G1 build (K=4), this build has K={K}")

IDENT = np.array([1.0, 0.0, 0.0, 0.0])


def _reachable(rng, p_b=None, q_b=None):
    """A 4-contact task that IS solvable, expressed at a given world base."""
    lim = np.asarray(hjcdik.joint_limits())
    q_true = rng.uniform(lim[:, 0], lim[:, 1])
    T = np.asarray(hjcdik.target_transforms(q_true[None, :])[0])          # [K,4,4] base frame
    p_b = np.zeros(3) if p_b is None else np.asarray(p_b, float)
    q_b = IDENT.copy() if q_b is None else np.asarray(q_b, float)
    Rb = quat_to_mat(q_b)
    pos = contact_points_world(T[:, :3, 3], p_b, q_b)
    quat = np.array([mat_to_quat(Rb @ T[k, :3, :3]) for k in range(K)])
    return q_true, T, np.concatenate([pos, quat], axis=1)[None, :, :]


def _seeds(rng, S, p_b, q_b, jitter=0.0, q_center=None):
    lim = np.asarray(hjcdik.joint_limits())
    j = (rng.uniform(lim[:, 0], lim[:, 1], (1, S, N)) if q_center is None
         else np.clip(q_center + rng.normal(0, jitter, (1, S, N)), lim[:, 0], lim[:, 1]))
    base = np.tile(np.concatenate([np.asarray(p_b, float), np.asarray(q_b, float)]), (1, S, 1))
    return np.concatenate([base, j], axis=2)


def _solve(poses, seeds, floating, base_update=None, **kw):
    args = dict(active_masks=np.array([0b1111], dtype=np.uint32), num_solutions=1,
                precision="float64", lm_iters=60, coarse_iters=0, coarse_mode="none", seed=0)
    args.update(kw)
    if base_update is not None:
        args["base_update"] = base_update
    return hjcdik.solve_problems(target_poses=poses, seed_configs=seeds,
                                 floating_base=floating, **args)


def _base_of(out, p=0, m=0):
    """The base belonging to solution (p, m), from the PUBLIC [P,M,...] surface.

    M5 gathers b = p*S + selected_seed_ids[p,m] inside solve_problems, so this is now just an
    index. It is kept as a helper only so the tests read the same way they did when they were
    proving that rule; `_base_position_candidates` is the raw candidate-major array for the
    tests that genuinely reason per candidate.
    """
    assert int(out["selected_seed_ids"][p, m]) >= 0, "no valid candidate for this solution"
    return (np.asarray(out["base_position"], float)[p, m],
            np.asarray(out["base_quaternion"], float)[p, m])


def _raw_bases(out):
    """The candidate-major [B,3] bases -- the private representation M5 preserves."""
    return np.asarray(out["_base_position_candidates"], float)


# ------------------------------------------------------------------------------------------
# Fixed-base regression: the feature must be invisible when off
# ------------------------------------------------------------------------------------------

def test_fixed_base_is_bit_identical_to_a_floating_solve_with_an_identity_base():
    """A floating solve at the identity base must reproduce the fixed-base solve EXACTLY.
    world_target_to_base's identity path is a verbatim copy, so there is no excuse for drift."""
    rng = np.random.default_rng(0)
    _, _, poses = _reachable(rng)
    S = 16
    fb = _seeds(rng, S, np.zeros(3), IDENT)
    fixed = _solve(poses, fb[:, :, 7:], False)
    floating = _solve(poses, fb, True)
    np.testing.assert_array_equal(fixed["joint_config"], floating["joint_config"])
    np.testing.assert_array_equal(fixed["position_errors"], floating["position_errors"])


def test_base_bounds_are_enforced_not_merely_accepted():
    """M5 wires base_bounds. The bar is ENFORCEMENT: through M4 this kwarg raised rather than lie
    about doing nothing, so 'it no longer raises' is not evidence it works. A target 2 m away pulls
    the base hard against the box on every step; every candidate must stay inside it.
    """
    rng = np.random.default_rng(1)
    q_true, T, poses = _reachable(rng, np.array([2.0, 0.0, 0.0]), IDENT)   # far => steady pull +x
    seeds = _seeds(rng, 8, np.zeros(3), IDENT, jitter=0.1, q_center=q_true)
    lower, upper = np.array([-0.02, -0.03, -0.04]), np.array([0.06, 0.03, 0.04])

    free = _solve(poses, seeds, True, base_update={"enabled": True}, lm_iters=60)
    bounded = _solve(poses, seeds, True, base_update={"enabled": True}, lm_iters=60,
                     base_bounds=(lower, upper))

    # the bound has to actually bite, or this proves nothing
    assert (_raw_bases(free) > upper + 1e-9).any(), "unbounded run stayed inside the box anyway"
    b = _raw_bases(bounded)                                  # EVERY candidate, not just selected
    assert np.all(b >= lower - 1e-9) and np.all(b <= upper + 1e-9), (
        f"base escaped its bounds: min={b.min(axis=0)}, max={b.max(axis=0)}, "
        f"lower={lower}, upper={upper}")


@pytest.mark.parametrize("bounds,msg", [
    (((0.1, 0, 0), (-0.1, 0, 0)), "lower must be <= upper"),
    (((0, 0), (1, 1)), "two 3-vectors"),
    (((0, 0, 0), (np.inf, 1, 1)), "must be finite"),
    ((np.zeros(3),), "pair of 3-vectors"),
])
def test_bad_base_bounds_are_rejected_before_any_cuda_work(bounds, msg):
    rng = np.random.default_rng(1)
    q_true, T, poses = _reachable(rng)
    seeds = _seeds(rng, 2, np.zeros(3), IDENT, jitter=0.0, q_center=q_true)
    with pytest.raises(ValueError, match=msg):
        _solve(poses, seeds, True, base_update={"enabled": True}, base_bounds=bounds)


# ------------------------------------------------------------------------------------------
# The coupled best state
# ------------------------------------------------------------------------------------------

def test_returned_joints_and_base_are_the_pair_that_scored_the_reported_error():
    """THE coupling test. best_x tracks the best JOINTS; with a mutable base those joints are
    only meaningful against the base they were scored at. If the kernel restored best joints
    against a stale base, recomputing the contact error from the RETURNED (joints, base) would
    disagree with the RETURNED position_errors."""
    rng = np.random.default_rng(2)
    p_true, q_true_b = np.array([0.06, -0.04, 0.03]), mat_to_quat(so3_exp([0.0, 0.0, 0.15]))
    q_true, T, poses = _reachable(rng, p_true, q_true_b)
    S = 32
    # start the base away from the truth so the update must actually move it
    seeds = _seeds(rng, S, np.zeros(3), IDENT, jitter=0.25, q_center=q_true)
    out = _solve(poses, seeds, True, base_update={"enabled": True})

    q_out = np.asarray(out["joint_config"][0, 0], float)
    p_out, b_out = _base_of(out)
    assert abs(np.linalg.norm(b_out) - 1.0) < 1e-9, "returned base quaternion is not unit"

    # recompute the contact error from the RETURNED pair, independently of the kernel
    c_base = np.asarray(hjcdik.target_transforms(q_out[None, :])[0])[:, :3, 3]
    x = contact_points_world(c_base, p_out, b_out)
    recomputed = np.linalg.norm(poses[0, :, :3] - x, axis=1)
    np.testing.assert_allclose(recomputed, out["position_errors"][0, 0], atol=1e-7)


def test_multi_seed_candidate_state_isolation():
    """Each candidate owns its base. Seeds at wildly different bases must not contaminate one
    another -- the base is candidate-level (gp) while the targets are problem-level (pid), which
    is exactly the indexing that would be wrong if they were confused."""
    rng = np.random.default_rng(3)
    q_true, T, poses = _reachable(rng, np.array([0.05, 0.0, 0.0]), IDENT)
    S = 8
    lim = np.asarray(hjcdik.joint_limits())
    j = np.tile(q_true, (1, S, 1))
    bases = np.zeros((1, S, 7))
    for s in range(S):                       # every seed a DIFFERENT base
        bases[0, s, :3] = [0.05 * s, -0.02 * s, 0.01 * s]
        bases[0, s, 3:] = mat_to_quat(so3_exp([0.0, 0.0, 0.05 * s]))
    seeds = np.concatenate([bases, j], axis=2)
    out = _solve(poses, seeds, True, num_solutions=S, base_update={"enabled": False})
    # seed 1 sits at the true base with the true joints => it must be the (near) exact solution,
    # and the others must NOT be dragged to it.
    errs = out["position_errors"][0]                      # [M,K]
    assert errs.min() < 1e-6, f"the exactly-correct candidate was lost: {errs.min():.3e}"
    assert errs.max() > 1e-3, "all candidates collapsed to one state: the base is being shared"


@pytest.mark.parametrize("enabled", [False, True])
def test_a_seed_that_is_already_solved_keeps_its_base(enabled):
    """A candidate seeded AT the answer must be returned unchanged -- the easiest input there is.

    best_x is seeded from s_x in the LM prologue but best_base_* was not, while the epilogue
    restores s_base_* from best_base_* unconditionally. The prologue sets s_break when the seed is
    already converged, so the iteration loop never runs, `improved` never fires, best_base_* is
    never written, and the restore reads UNINITIALIZED shared memory -- returning a zero/garbage
    base and destroying a perfect solution. Parametrized on `enabled` because this is NOT a
    base-update bug: it reproduces with the update off, so it must be fixed for both.
    """
    rng = np.random.default_rng(4)
    p_true = np.array([0.08, -0.05, 0.04])
    q_true, T, poses = _reachable(rng, p_true, IDENT)
    seeds = _seeds(rng, 1, p_true, IDENT, jitter=0.0, q_center=q_true)   # EXACTLY the optimum
    out = _solve(poses, seeds, True, base_update={"enabled": enabled}, lm_iters=20)
    assert out["position_errors"][0, 0].max() < 1e-6, (
        f"a seed at the exact optimum came back with error "
        f"{out['position_errors'][0,0].max():.3e} -- its base was discarded")
    np.testing.assert_allclose(_base_of(out)[0], p_true, atol=1e-9)


@pytest.mark.parametrize("S", [1, 2, 4])
def test_solve_does_not_mutate_the_callers_seed_configs(S):
    """solve_problems takes seeds as INPUT; writing the refined base back into the caller's array
    is a silent, destructive side effect.

    This is shape-dependent and so is exactly the kind of bug that hides: base_p was built with
    ascontiguousarray(flat[:, 0:3]), and a [B,3] column slice is non-contiguous (=> copies) for
    B > 1 but IS "C-contiguous" for B == 1 (a leading dim of 1 makes its stride irrelevant), so at
    B == 1 it returned a VIEW aliasing seed_configs and the IN/OUT base D2H landed in the caller's
    array. S is parametrized because S=1 is the aliasing case and S>1 is the control.
    """
    rng = np.random.default_rng(4)
    q_true, T, poses = _reachable(rng, np.array([0.08, -0.05, 0.04]), IDENT)
    seeds = _seeds(rng, S, np.zeros(3), IDENT, jitter=0.0, q_center=q_true)
    seeds_before, poses_before = seeds.copy(), poses.copy()
    _solve(poses, seeds, True, base_update={"enabled": True}, lm_iters=10)
    np.testing.assert_array_equal(
        seeds, seeds_before,
        err_msg=f"S={S}: solve_problems mutated the caller's seed_configs")
    np.testing.assert_array_equal(poses, poses_before,
                                  err_msg=f"S={S}: solve_problems mutated the caller's target_poses")


def test_repeated_solves_reusing_one_seed_array_are_bit_identical():
    """The user-visible symptom of the aliasing above: the kernel is a pure function of its inputs
    (no clock, no atomics, no RNG once seeds are fixed), so calling it twice with the SAME array
    must reproduce bit for bit. It did not -- each call silently resumed from the previous call's
    refined base, so a fixed config drifted 7.97e-03 -> 9.77e-04 -> 2.62e-05 across repeats and
    looked exactly like nondeterminism. Reuses ONE array by design; do not rebuild it in the loop.
    """
    rng = np.random.default_rng(4)
    q_true, T, poses = _reachable(rng, np.array([0.08, -0.05, 0.04]), IDENT)
    seeds = _seeds(rng, 1, np.zeros(3), IDENT, jitter=0.0, q_center=q_true)
    ref = _solve(poses, seeds, True, base_update={"enabled": True}, lm_iters=10)
    for rep in range(5):
        out = _solve(poses, seeds, True, base_update={"enabled": True}, lm_iters=10)
        np.testing.assert_array_equal(
            out["joint_config"], ref["joint_config"],
            err_msg=f"repeat {rep}: joints not reproducible")
        np.testing.assert_array_equal(
            np.asarray(out["base_position"], float), np.asarray(ref["base_position"], float),
            err_msg=f"repeat {rep}: base_position not reproducible")


# ------------------------------------------------------------------------------------------
# The update itself
# ------------------------------------------------------------------------------------------

def test_base_update_solves_a_task_a_fixed_base_cannot():
    """The whole point: a candidate seeded at the WRONG base must fix it itself.

    Asserts the SOLVE, not the base value. The base is NOT identifiable from contact positions:
    with 29 joints free there are many (base, joints) pairs that put the 4 contacts in the same
    place, and the solver is entitled to any of them. Measured here -- this task is solved to
    2.9e-05 m with a base 0.044 m away from the p_true it was built from -- so asserting
    recovery of p_true would be asserting something false. What IS guaranteed: the contacts get
    hit, the base does real work, and the returned (base, joints) pair is self-consistent (that
    last one is test_returned_joints_and_base_are_the_pair_that_scored_the_reported_error).

    Consequence for M6: validate a base by the contact error it produces, never by comparing it
    to a ground-truth base. A sampled-base-vs-native-base comparison must score both on error.
    """
    rng = np.random.default_rng(4)
    p_true = np.array([0.08, -0.05, 0.04])
    q_true, T, poses = _reachable(rng, p_true, IDENT)
    # Seeds must be DIVERSE in the joints: every seed carries base=0, so a batch that is
    # effectively one repeated seed lands every candidate in the same local minimum. Alternating
    # block descent is not globally convergent (the design's risk #1), and HJCD's answer to that
    # is the batch -- so exercise it the way it is meant to be used.
    seeds = _seeds(rng, 32, np.zeros(3), IDENT, jitter=0.4, q_center=q_true)
    off = _solve(poses, seeds, True, base_update={"enabled": False}, lm_iters=200)
    on = _solve(poses, seeds, True, base_update={"enabled": True}, lm_iters=200)

    assert on["position_errors"][0, 0].max() < off["position_errors"][0, 0].max()
    assert on["position_errors"][0, 0].max() < 1e-3, (
        f"the base update did not solve the task: {on['position_errors'][0,0].max():.3e}")
    # off leaves every base exactly at its seed; on must actually move the one it selected
    np.testing.assert_array_equal(np.asarray(off["base_position"], float),
                                  np.zeros_like(np.asarray(off["base_position"], float)))
    assert np.linalg.norm(_base_of(on)[0]) > 1e-3, "the selected base never moved off its seed"


# ------------------------------------------------------------------------------------------
# M5: the public API surface
# ------------------------------------------------------------------------------------------

def test_public_shapes_are_selection_major_and_pair_with_joint_config():
    """base_position/base_quaternion are gathered to the SAME [P,M,...] selection as joint_config,
    so a caller never open-codes b = p*S + seed_id (and never drops the p*S stride)."""
    rng = np.random.default_rng(30)
    p_trues = [[0.06, -0.03, 0.02], [-0.05, 0.04, -0.03], [0.09, 0.07, 0.05]]
    S, M = 8, 3
    poses, seeds, _, masks = _multi(rng, p_trues, S)
    out = _solve(poses, seeds, True, base_update={"enabled": True},
                 active_masks=masks, num_solutions=M, lm_iters=120)
    P = len(p_trues)
    assert out["joint_config"].shape == (P, M, N)
    assert np.asarray(out["base_position"]).shape == (P, M, 3)
    assert np.asarray(out["base_quaternion"]).shape == (P, M, 4)
    # the private candidate-major representation is preserved for per-candidate reasoning
    assert _raw_bases(out).shape == (P * S, 3)
    assert np.asarray(out["_base_quaternion_candidates"]).shape == (P * S, 4)
    q = np.asarray(out["base_quaternion"], float)
    np.testing.assert_allclose(np.linalg.norm(q, axis=-1), 1.0, atol=1e-9)
    # and the gather agrees with doing it by hand
    for p in range(P):
        for m in range(M):
            sid = int(out["selected_seed_ids"][p, m])
            if sid < 0:
                continue
            np.testing.assert_array_equal(np.asarray(out["base_position"])[p, m],
                                          _raw_bases(out)[p * S + sid])


def test_fixed_base_calls_are_untouched_by_M5():
    """A fixed-base caller must see exactly what it saw before the floating base existed: no base
    keys at all, and the floating-base kwargs rejected rather than quietly ignored."""
    rng = np.random.default_rng(31)
    q_true, T, poses = _reachable(rng)
    seeds = _seeds(rng, 8, np.zeros(3), IDENT, jitter=0.2, q_center=q_true)
    out = _solve(poses, seeds[:, :, 7:], False)
    for k in ("base_position", "base_quaternion",
              "_base_position_candidates", "_base_quaternion_candidates"):
        assert k not in out, f"fixed-base solve leaked '{k}'"
    with pytest.raises(ValueError, match="base_update requires floating_base=True"):
        _solve(poses, seeds[:, :, 7:], False, base_update={"enabled": True})
    with pytest.raises(ValueError, match="base_bounds requires floating_base=True"):
        _solve(poses, seeds[:, :, 7:], False, base_bounds=((-1, -1, -1), (1, 1, 1)))


def test_base_updates_are_off_unless_asked_for():
    """floating_base alone CARRIES a base; it must not silently start optimizing it. Both the
    no-config and the explicit-default calls must leave every base exactly at its seed."""
    rng = np.random.default_rng(32)
    q_true, T, poses = _reachable(rng, np.array([0.08, -0.05, 0.04]), IDENT)
    seeds = _seeds(rng, 4, np.zeros(3), IDENT, jitter=0.1, q_center=q_true)
    for cfg in (None, {}, {"enabled": False}):
        out = _solve(poses, seeds, True, base_update=cfg, lm_iters=60)
        np.testing.assert_array_equal(
            _raw_bases(out), np.zeros((4, 3)),
            err_msg=f"base_update={cfg!r} moved the base without being enabled")
    on = _solve(poses, seeds, True, base_update={"enabled": True}, lm_iters=60)
    assert np.abs(_raw_bases(on)).max() > 1e-6, "the enabled control did not bite"


@pytest.mark.parametrize("bad,msg", [
    ({"enabled": True, "interval": 0}, r"'interval'.*>= 1"),
    ({"enabled": True, "interval": -3}, r"'interval'.*>= 1"),
    ({"enabled": True, "damping": -1e-3}, r"'damping'.*>= 0"),
    ({"enabled": True, "damping": float("nan")}, r"'damping'.*>= 0"),
    ({"enabled": True, "step_scale": 0.0}, r"'step_scale'.*> 0"),
    ({"enabled": True, "max_translation_step": 0.0}, r"'max_translation_step'.*> 0"),
    ({"enabled": True, "max_rotation_step": -1.0}, r"'max_rotation_step'.*> 0"),
    ({"enabled": True, "typo_key": 1.0}, r"unknown base_update keys \['typo_key'\]"),
])
def test_bad_base_update_config_is_rejected_before_any_cuda_work(bad, msg):
    """Named, eager validation. A bad base config that reaches the kernel does not raise -- it
    silently solves a different problem, which is the entire failure mode of this subsystem."""
    rng = np.random.default_rng(33)
    q_true, T, poses = _reachable(rng)
    seeds = _seeds(rng, 2, np.zeros(3), IDENT, jitter=0.0, q_center=q_true)
    with pytest.raises((ValueError, TypeError), match=msg):
        _solve(poses, seeds, True, base_update=bad)


def test_clipping_can_be_disabled_explicitly_with_none():
    """M7 must be able to ablate the clips. None means unclipped -- distinct from 0, which would
    be an unreadable way to spell 'no limit'."""
    rng = np.random.default_rng(34)
    q_true, T, poses = _reachable(rng, np.array([2.0, 0.0, 0.0]), IDENT)   # 2 m => huge ask
    seeds = _seeds(rng, 4, np.zeros(3), IDENT, jitter=0.01, q_center=q_true)
    clipped = _solve(poses, seeds, True, lm_iters=1,
                     base_update={"enabled": True, "max_translation_step": 0.05})
    free = _solve(poses, seeds, True, lm_iters=1,
                  base_update={"enabled": True, "max_translation_step": None})
    assert np.linalg.norm(_raw_bases(clipped)[0]) <= 0.05 + 1e-9
    assert np.linalg.norm(_raw_bases(free)[0]) > 0.05, "None did not disable the clip"


def test_seed_quaternion_must_be_a_real_rotation():
    rng = np.random.default_rng(35)
    q_true, T, poses = _reachable(rng)
    seeds = _seeds(rng, 2, np.zeros(3), IDENT, jitter=0.0, q_center=q_true)
    bad = seeds.copy()
    bad[0, 0, 3:7] = 0.0                       # zero quaternion: not a rotation
    with pytest.raises(ValueError, match="quaternion"):
        _solve(poses, bad, True, base_update={"enabled": True})


def test_floating_seed_shape_names_what_it_wants():
    rng = np.random.default_rng(36)
    q_true, T, poses = _reachable(rng)
    joints_only = _seeds(rng, 2, np.zeros(3), IDENT, jitter=0.0, q_center=q_true)[:, :, 7:]
    with pytest.raises(ValueError, match=rf"\[P, S, {N + 7}\].*qw,qx,qy,qz"):
        _solve(poses, joints_only, True, base_update={"enabled": True})


# ------------------------------------------------------------------------------------------
# The oracle: host reference and kernel must solve the SAME system
# ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("scale_p,scale_R", [(1.0, 1.0), (0.05, 0.10), (2.0, 0.5)])
def test_gpu_base_proposal_matches_the_host_reference(scale_p, scale_R):
    """THE oracle test. hjcdik/base_update.py is only a reference if it solves what the kernel
    solves; they diverged once (host lambda*I vs kernel lambda*diag(H)) and nothing caught it,
    because nothing compared them. Both now form H_lambda = H + lambda*D with
    D = diag(s_p^-2 I3, s_R^-2 I3), so one step must agree.

    Read at lm_iters=1: the base update runs AFTER the joint step (hjcd_kernel.cu, the
    `bcfg.enabled` block), so the joints it saw are the ones that come back in joint_config --
    which is what the host is handed here. The seed base is 0.1 m off with the joints at q_true,
    so the step is a large clear improvement and BOTH sides accept it; that keeps this a test of
    the PROPOSAL rather than of the two acceptance rules (the kernel accepts on E_phys, the host
    on its own position cost). Tolerance is 1e-8, not exact: the device composes the rotation as
    a quaternion and the host as a matrix -- same mathematics, different float path (M3 measured
    the same 2.2e-09 gap).
    """
    from hjcdik.base_update import BaseUpdateConfig, base_update_step

    rng = np.random.default_rng(21)
    p_seed = np.zeros(3)
    q_true, T, poses = _reachable(rng, np.array([0.08, -0.05, 0.04]), IDENT)
    seeds = _seeds(rng, 1, p_seed, IDENT, jitter=0.0, q_center=q_true)
    bu = {"enabled": True, "interval": 1,
          "damping": 1e-3, "step_scale": 1.0,
          "scale_p": scale_p, "scale_R": scale_R,
          "max_translation_step": 0.05, "max_rotation_step": 0.10}
    out = _solve(poses, seeds, True, base_update=bu, lm_iters=1)

    q_seen = np.asarray(out["joint_config"][0, 0], float)      # the joints the base step saw
    p_gpu, q_gpu = _base_of(out)
    assert np.linalg.norm(p_gpu - p_seed) > 1e-6, "the kernel took no step; nothing to compare"

    c_base = np.asarray(hjcdik.target_transforms(q_seen[None, :])[0])[:, :3, 3]
    ref = base_update_step(
        c_base, poses[0, :, :3], p_seed, IDENT,
        BaseUpdateConfig(damping=1e-3, step_scale=1.0, scale_p=scale_p, scale_R=scale_R,
                         max_translation_step=0.05, max_rotation_step=0.10),
        active_mask=0b1111)
    assert ref["accepted"], "the host reference rejected a step the kernel accepted"

    np.testing.assert_allclose(
        p_gpu, ref["p_b"], atol=1e-8,
        err_msg=f"base_position disagrees with the host reference at "
                f"(s_p={scale_p}, s_R={scale_R}) -- the oracle and the kernel are solving "
                f"different systems again")
    # q and -q are the same rotation; compare on the shorter arc
    dq = min(np.linalg.norm(q_gpu - ref["q_b"]), np.linalg.norm(q_gpu + ref["q_b"]))
    assert dq < 1e-8, f"base_quaternion disagrees with the host reference by {dq:.3e}"


def test_damping_scales_must_be_positive():
    """s_p/s_R enter as s^-2 and are what make lambda*D positive definite -- which is what lets
    the kernel drop the zero-diagonal pin. A zero scale would put a singular H_lambda back."""
    rng = np.random.default_rng(22)
    q_true, T, poses = _reachable(rng)
    seeds = _seeds(rng, 2, np.zeros(3), IDENT, jitter=0.0, q_center=q_true)
    for bad in ({"scale_p": 0.0}, {"scale_p": -1.0},
                {"scale_R": 0.0}, {"scale_R": float("inf")}):
        with pytest.raises(ValueError, match=r"must be > 0 and finite"):
            _solve(poses, seeds, True, base_update={"enabled": True, **bad})


# ------------------------------------------------------------------------------------------
# The flattening rule: b = p*S + selected_seed_ids[p, m]
# ------------------------------------------------------------------------------------------

def _multi(rng, p_trues, S, jitter=0.4):
    """P independent tasks, each at its OWN world base, each with its own seeds."""
    poses, seeds, truths = [], [], []
    for p_b in p_trues:
        q_true, _, ps = _reachable(rng, np.asarray(p_b, float), IDENT)
        poses.append(ps[0])
        seeds.append(_seeds(rng, S, np.zeros(3), IDENT, jitter=jitter, q_center=q_true)[0])
        truths.append(q_true)
    P = len(p_trues)
    return (np.stack(poses), np.stack(seeds), truths,
            np.full(P, 0b1111, dtype=np.uint32))


def test_selected_state_is_consistent_for_every_p_and_m():
    """THE flattening test. base_position/base_quaternion are per CANDIDATE [B,...] while
    joint_config is per SELECTED solution [P,M,N]; they are paired ONLY through
    b = p*S + selected_seed_ids[p, m]. At P == 1 a wrong rule (e.g. b = seed_id, dropping the
    problem stride) is indistinguishable from the right one -- it only breaks at P > 1, and every
    other test in this file runs P = 1. Recomputes the contact error for EVERY (p, m) from the
    returned triple and requires it to match the reported one.
    """
    rng = np.random.default_rng(11)
    p_trues = [[0.06, -0.03, 0.02], [-0.05, 0.04, -0.03], [0.09, 0.07, 0.05]]
    S, M = 8, 3
    poses, seeds, _, masks = _multi(rng, p_trues, S)
    out = _solve(poses, seeds, True, base_update={"enabled": True},
                 active_masks=masks, num_solutions=M, lm_iters=120)
    assert out["seeds_per_problem"] == S
    checked = 0
    for p in range(len(p_trues)):
        for m in range(M):
            if int(out["selected_seed_ids"][p, m]) < 0:
                continue
            q_out = np.asarray(out["joint_config"][p, m], float)
            p_out, b_out = _base_of(out, p, m)
            assert abs(np.linalg.norm(b_out) - 1.0) < 1e-9, f"({p},{m}): base quat not unit"
            c_base = np.asarray(hjcdik.target_transforms(q_out[None, :])[0])[:, :3, 3]
            x = contact_points_world(c_base, p_out, b_out)
            recomputed = np.linalg.norm(poses[p, :, :3] - x, axis=1)
            np.testing.assert_allclose(
                recomputed, out["position_errors"][p, m], atol=1e-7,
                err_msg=f"solution ({p},{m}): the returned (base, joints) do not reproduce the "
                        f"reported error -- the b = p*S + seed_id gather is wrong")
            checked += 1
    assert checked >= len(p_trues), f"only {checked} solutions were valid; the test proved little"


def test_two_problems_selecting_the_same_local_seed_id_get_their_own_base():
    """seed_id is LOCAL to a problem, so two problems can both select seed 0 -- and then only the
    p*S stride separates them. If it were dropped, problem 1 would silently read problem 0's base.
    Forced here: seed 0 of each problem is planted at that problem's own solution, so it wins.
    """
    rng = np.random.default_rng(12)
    p_trues = [np.array([0.07, -0.02, 0.03]), np.array([-0.06, 0.05, -0.04])]
    S = 4
    poses, seeds, truths = [], [], []
    for p_b in p_trues:
        q_true, _, ps = _reachable(rng, p_b, IDENT)
        poses.append(ps[0])
        sd = _seeds(rng, S, np.zeros(3), IDENT, jitter=0.5, q_center=q_true)[0]
        sd[0, :3] = p_b                 # seed 0 == the exact answer for THIS problem
        sd[0, 3:7] = IDENT
        sd[0, 7:] = q_true
        seeds.append(sd)
        truths.append(p_b)
    poses, seeds = np.stack(poses), np.stack(seeds)
    out = _solve(poses, seeds, True, base_update={"enabled": True},
                 active_masks=np.full(2, 0b1111, dtype=np.uint32), num_solutions=1, lm_iters=60)

    for p in range(2):
        assert int(out["selected_seed_ids"][p, 0]) == 0, (
            f"problem {p} did not select its planted seed; the test setup no longer bites")
        assert out["position_errors"][p, 0].max() < 1e-6
        np.testing.assert_allclose(
            _base_of(out, p, 0)[0], truths[p], atol=1e-9,
            err_msg=f"problem {p} got the wrong base: both problems selected local seed 0, so a "
                    f"gather missing the p*S stride hands problem 1 problem 0's base")


def test_no_solution_slot_can_borrow_another_problems_candidate():
    """A slot filled with a seed_id outside [0, S) would gather b = p*S + sid from a NEIGHBOURING
    problem's block -- a real base belonging to the wrong task, so the output would look entirely
    plausible. Two things keep that from happening and both are asserted here: the API refuses
    num_solutions > S (so no slot is ever left to be filled by accident), and every id a solve
    does return is local to its own problem and distinct.

    Run on a task NO candidate solves, because that is when a selector is most likely to reach
    for something outside its block.
    """
    rng = np.random.default_rng(13)
    p_trues = [[0.06, -0.03, 0.02], [-0.05, 0.04, -0.03]]
    S = 4
    poses, seeds, _, masks = _multi(rng, p_trues, S)
    poses[:, :, :3] += 5.0                       # unreachable: nothing converges

    with pytest.raises(ValueError, match=r"num_solutions must be in \[1, S=4\]"):
        _solve(poses, seeds, True, base_update={"enabled": True},
               active_masks=masks, num_solutions=S + 1)

    out = _solve(poses, seeds, True, base_update={"enabled": True},
                 active_masks=masks, num_solutions=S, lm_iters=60)
    assert not out["success"].any(), "the task was supposed to be unreachable"
    sids = np.asarray(out["selected_seed_ids"])
    for p in range(len(p_trues)):
        row = [int(s) for s in sids[p]]
        assert all(0 <= s < S for s in row), (
            f"problem {p} selected {row}, outside its own [0, {S}) block")
        assert len(set(row)) == len(row), f"problem {p} selected a candidate twice: {row}"
    """Acceptance is on E_phys with exact rollback, so enabling the update can never make a
    candidate worse than carrying its seed base. Checked across many random tasks."""
    for t in range(6):
        rng = np.random.default_rng(100 + t)
        p_true = rng.uniform(-0.08, 0.08, 3)
        q_true, T, poses = _reachable(rng, p_true, IDENT)
        seeds = _seeds(rng, 8, np.zeros(3), IDENT, jitter=0.05, q_center=q_true)
        off = _solve(poses, seeds, True, base_update={"enabled": False})
        on = _solve(poses, seeds, True, base_update={"enabled": True})
        assert on["cost_physical"][0, 0] <= off["cost_physical"][0, 0] + 1e-9, (
            f"trial {t}: base update made it worse "
            f"({on['cost_physical'][0,0]:.6e} > {off['cost_physical'][0,0]:.6e})")


def test_no_nan_on_a_degenerate_all_contacts_at_base_task():
    """All lever arms ~zero => the whole rotation block of H vanishes. Scaled damping cannot
    rescue a zero diagonal (lambda*diag adds nothing), so this is precisely the case the
    zero-diagonal pin exists for. It must produce a finite result, never NaN."""
    rng = np.random.default_rng(5)
    q_true, T, poses = _reachable(rng)
    poses[0, :, :3] = 0.0                      # every target AT the base origin: unreachable+degenerate
    seeds = _seeds(rng, 8, np.zeros(3), IDENT)
    out = _solve(poses, seeds, True, base_update={"enabled": True})
    assert np.all(np.isfinite(out["joint_config"]))
    assert np.all(np.isfinite(np.asarray(out["base_position"], float)))
    assert np.all(np.isfinite(np.asarray(out["base_quaternion"], float)))
    assert np.all(np.isfinite(out["position_errors"]))


def test_translation_clipping_bounds_the_per_step_motion():
    """A far-away target would ask for a huge step; the clip must bound it. With interval=1 and
    lm_iters=1 exactly ONE base step is taken, so the total motion is one clipped step."""
    rng = np.random.default_rng(6)
    q_true, T, poses = _reachable(rng, np.array([2.0, 0.0, 0.0]), IDENT)   # 2 m away
    seeds = _seeds(rng, 4, np.zeros(3), IDENT, jitter=0.01, q_center=q_true)
    out = _solve(poses, seeds, True, lm_iters=1,
                 base_update={"enabled": True, "max_translation_step": 0.05})
    moved = np.linalg.norm(_base_of(out)[0])
    assert moved <= 0.05 + 1e-9, f"one step moved {moved:.4f} m > the 0.05 m clip"


def test_quaternion_stays_unit_over_many_steps():
    rng = np.random.default_rng(7)
    q_true, T, poses = _reachable(rng, np.array([0.05, 0.02, 0.0]),
                                  mat_to_quat(so3_exp([0.0, 0.0, 0.2])))
    seeds = _seeds(rng, 8, np.zeros(3), IDENT, jitter=0.05, q_center=q_true)
    out = _solve(poses, seeds, True, lm_iters=60, base_update={"enabled": True})
    for b in range(np.asarray(out["base_quaternion"]).shape[0]):
        n = np.linalg.norm(np.asarray(out["base_quaternion"], float)[b])
        assert abs(n - 1.0) < 1e-9, f"quaternion drifted: |q| = {n:.12f}"
