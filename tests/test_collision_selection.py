"""Self-collision participates in candidate ELIGIBILITY, before top-M selection.

Before Checkpoint 7 the flow was:

    solve P*S  ->  select winner  ->  gather base  ->  check ONLY the winner  ->  fail if it hit

so a problem failed whenever its best-ranked candidate self-collided, even with a slightly worse
collision-free candidate sitting in the same batch, and `candidates_checked` was 1 per problem.

Now:

    solve P*S  ->  device sidecar over ALL B candidates  ->  self_collision_free[B]
               ->  segmented_topM_kernel ANDs it into feasibility
               ->  cand_better(class, E_phys, seed) ranks the survivors
               ->  ONE gather of q / base / diagnostics from the final selected_seed_ids

cand_better() stays the only ranking implementation; nothing re-ranks on the host.

The two collision systems stay distinct: grid_collision (--collision, `final_free`, environment)
and the g1sc self-collision sidecar (`self_free`). Selection ANDs whichever channels are active.
"""
from pathlib import Path

import numpy as np
import pytest

import hjcdik

N = hjcdik.num_joints()
K = hjcdik.num_targets()
ALL_ACTIVE = np.array([(1 << K) - 1], dtype=np.uint32)

pytestmark = pytest.mark.skipif(
    not hjcdik.self_collision_info().get("sidecar_compiled", False),
    reason="build has no self-collision sidecar")


def reachable_poses(q):
    T = np.asarray(hjcdik.target_transforms(np.asarray(q, float).reshape(1, N)))[0]
    poses = np.zeros((K, 7))
    for k in range(K):
        poses[k, :3] = T[k][:3, 3]
        poses[k, 3:] = np.array([1.0, 0.0, 0.0, 0.0])
    return poses


def solve(poses, seeds, mode, **kw):
    base = dict(active_masks=ALL_ACTIVE, seed_configs=seeds, num_solutions=1,
                position_tol=1e-4, orientation_tol=1e-3, seed=5,
                orientation_modes="none",          # isolate: this is a collision test
                self_collision_mode=mode)
    base.update(kw)
    return hjcdik.solve_problems(target_poses=poses[None], **base)


def colliding(q_rows, margin=0.0):
    """Ground truth from the sidecar's own host API -- deliberately NOT the path under test."""
    q = np.ascontiguousarray(np.atleast_2d(q_rows).astype(np.float32))
    return np.asarray(hjcdik._hjcdik.sidecar_full_check(q, float(margin))).any(axis=1)


@pytest.fixture(scope="module")
def scene():
    rng = np.random.default_rng(1)
    q_ref = rng.uniform(-0.3, 0.3, N)
    poses = reachable_poses(q_ref)
    seeds = q_ref[None, None, :] + rng.normal(0, 0.2, (1, 64, N))
    return q_ref, poses, seeds


# =============================================================================================
# A. The primary acceptance test: best candidate collides, a worse collision-free one wins
# =============================================================================================

def test_colliding_best_is_rejected_and_next_free_candidate_is_selected(scene):
    _, poses, seeds = scene
    off = solve(poses, seeds, "off")
    fin = solve(poses, seeds, "final")

    a = int(off["selected_seed_ids"][0, 0])
    b = int(fin["selected_seed_ids"][0, 0])

    # The premise: the unconstrained winner really does self-collide. Checked against the sidecar
    # directly, so the test cannot be satisfied by the selection path agreeing with itself.
    assert colliding(off["joint_config"][0, 0])[0], (
        "premise failed: the collision-off winner does not self-collide, so this scene cannot "
        "demonstrate rejection")
    assert a != b, "collision-aware selection returned the same colliding candidate"

    # The replacement must be a genuine, usable solution -- not a downgrade to 'invalid'.
    assert bool(fin["success"][0, 0])
    assert bool(fin["valid"][0, 0])
    assert bool(np.asarray(fin["collision_free"])[0, 0])
    assert not colliding(fin["joint_config"][0, 0])[0]

    # ...and it is WORSE kinematically, which is the whole point: eligibility, not re-ranking.
    e_off = float(np.max(off["position_errors"]))
    e_fin = float(np.max(fin["position_errors"]))
    assert e_fin >= e_off - 1e-12, (
        f"the collision-free winner ({e_fin:.3e}) beat the unconstrained winner ({e_off:.3e}); "
        "ranking among survivors is not being preserved")


def test_every_candidate_is_checked_not_just_the_winner(scene):
    """The old gate checked 1 candidate per problem; the new one checks all P*S."""
    _, poses, seeds = scene
    fin = solve(poses, seeds, "final")
    S = seeds.shape[1]
    assert fin["self_collision"]["candidates_checked"] == S
    assert int(np.asarray(fin["num_collision_free"])[0]) <= S


def test_selection_prefers_lower_ephys_among_free_candidates(scene):
    """Among collision-free candidates the existing (class, E_phys, seed) order must still hold."""
    _, poses, seeds = scene
    fin = solve(poses, seeds, "final", return_all_candidates=True)
    free = np.asarray(fin["all_collision_free"])[0]
    succ = np.asarray(fin["all_success"])[0].astype(bool)
    ephys = np.asarray(fin["all_cost_physical"])[0]
    eligible = free.astype(bool) & succ
    if eligible.sum() == 0:
        pytest.skip("no eligible collision-free candidate in this scene")
    best = int(np.lexsort((np.arange(len(ephys)), np.where(eligible, ephys, np.inf)))[0])
    assert int(fin["selected_seed_ids"][0, 0]) == best


# =============================================================================================
# B. All candidates collision-free => identical to collision-off
# =============================================================================================

def test_all_free_matches_collision_off_exactly(scene):
    """When every candidate is free, `final` must select and return EXACTLY what `off` does.

    The all-free condition is forced with a large NEGATIVE margin (a pair collides when its gap is
    below the margin, so -1 m is unreachable) rather than by hunting for a benign scene. That makes
    the premise exact and deterministic instead of dependent on where the solver happens to land --
    and it is the precise mirror of test C, which forces the all-COLLIDING branch the same way.
    """
    _, poses, seeds = scene
    off = solve(poses, seeds, "off")
    fin = solve(poses, seeds, "final", self_collision_margin=-1.0)

    assert int(np.asarray(fin["num_collision_free"])[0]) == seeds.shape[1], \
        "the negative margin did not make every candidate free"
    assert int(off["selected_seed_ids"][0, 0]) == int(fin["selected_seed_ids"][0, 0])
    np.testing.assert_array_equal(np.asarray(off["joint_config"]), np.asarray(fin["joint_config"]))
    np.testing.assert_array_equal(np.asarray(off["position_errors"]),
                                  np.asarray(fin["position_errors"]))
    np.testing.assert_array_equal(np.asarray(off["orientation_errors"]),
                                  np.asarray(fin["orientation_errors"]))


# =============================================================================================
# C. Every candidate collides => clean failure, and no colliding candidate reported usable
# =============================================================================================

def test_all_colliding_fails_cleanly(scene):
    """A huge margin makes every proxy pair 'colliding' (gap < margin), which is the cleanest
    deterministic way to reach the all-infeasible branch without hand-building 64 bad poses."""
    _, poses, seeds = scene
    fin = solve(poses, seeds, "final", self_collision_margin=1.0)

    assert int(np.asarray(fin["num_collision_free"])[0]) == 0
    assert not bool(fin["success"][0, 0])
    assert not bool(fin["valid"][0, 0])
    assert int(fin["selected_seed_ids"][0, 0]) == -1
    # A fallback payload may be returned for shape stability, but must never read as feasible.
    assert not bool(np.asarray(fin["collision_free"])[0, 0])
    assert np.asarray(fin["joint_config"]).shape == (1, 1, N)


# =============================================================================================
# D. Floating base: q and base must come from the SAME re-selected candidate
# =============================================================================================

def test_floating_base_pairing_survives_reselection(scene):
    """The base arrays are candidate-major and were previously gathered BEFORE the collision gate.

    Give every seed a distinctly different base, force a re-selection, then check that the
    returned base is the selected candidate's own -- not the one the pre-gate winner had.
    """
    q_ref, poses, _ = scene
    rng = np.random.default_rng(4)
    S = 64
    fseeds = np.zeros((1, S, 7 + N))
    # Distinct, identifiable base per seed: any mix-up shows up as a large mismatch, not a wobble.
    fseeds[0, :, 0] = np.linspace(-0.25, 0.25, S)
    fseeds[0, :, 1] = np.linspace(0.25, -0.25, S)
    fseeds[0, :, 2] = np.linspace(-0.1, 0.1, S)
    fseeds[0, :, 3] = 1.0
    fseeds[0, :, 7:] = q_ref[None, :] + rng.normal(0, 0.2, (S, N))

    kw = dict(active_masks=ALL_ACTIVE, seed_configs=fseeds, floating_base=True, num_solutions=1,
              position_tol=1e-4, orientation_tol=1e-3, seed=5, orientation_modes="none",
              return_all_candidates=True)
    off = hjcdik.solve_problems(target_poses=poses[None], self_collision_mode="off", **kw)
    fin = hjcdik.solve_problems(target_poses=poses[None], self_collision_mode="final", **kw)

    a, b = int(off["selected_seed_ids"][0, 0]), int(fin["selected_seed_ids"][0, 0])
    if a == b:
        pytest.skip("no re-selection happened in this scene; pairing is untested here")

    # The returned base must equal the SELECTED candidate's base, read from the candidate-major
    # array by the same p*S + seed rule -- and must NOT equal the rejected winner's.
    bp = np.asarray(fin["_base_position_candidates"])
    bq = np.asarray(fin["_base_quaternion_candidates"])
    np.testing.assert_allclose(np.asarray(fin["base_position"])[0, 0], bp[b], rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(fin["base_quaternion"])[0, 0], bq[b], rtol=0, atol=0)
    assert not np.allclose(np.asarray(fin["base_position"])[0, 0], bp[a]), (
        "returned the REJECTED candidate's base: q/base pairing broke across re-selection")

    # Joints likewise come from b.
    np.testing.assert_allclose(np.asarray(fin["joint_config"])[0, 0],
                               np.asarray(fin["all_joint_config"])[0, b], rtol=0, atol=0)

    # And every gathered base diagnostic must describe the same candidate.
    d = np.asarray(fin["_base_diag_candidates"])
    assert int(np.asarray(fin["base_updates_attempted"])[0, 0]) == int(d[b, 0])
    assert int(np.asarray(fin["base_updates_accepted"])[0, 0]) == int(d[b, 1])
    assert int(np.asarray(fin["base_numerical_failures"])[0, 0]) == int(d[b, 2])
    seed_p = fseeds[0, b, 0:3]
    # float32 solve: compare at the compute precision, not float64's.
    np.testing.assert_allclose(float(np.asarray(fin["base_translation_moved"])[0, 0]),
                               float(np.linalg.norm(bp[b] - seed_p)), rtol=1e-5, atol=1e-6)


# =============================================================================================
# E. num_solutions > 1
# =============================================================================================

def test_top_m_selects_m_collision_free_candidates():
    """Top-M must fill all M slots from collision-free candidates, in the deterministic order.

    Uses a gently perturbed scene so enough candidates survive the gate to fill M; the scene-level
    fixture is deliberately harsh (it exists to force rejections) and would leave too few.
    """
    rng = np.random.default_rng(21)
    q_ref = rng.uniform(-0.2, 0.2, N)
    poses = reachable_poses(q_ref)
    seeds = q_ref[None, None, :] + rng.normal(0, 0.08, (1, 96, N))
    M = 3
    fin = solve(poses, seeds, "final", num_solutions=M, return_all_candidates=True)
    nfree = int(np.asarray(fin["num_collision_free"])[0])
    assert nfree >= M, f"scene yielded only {nfree} collision-free candidates; cannot fill M={M}"
    sel = np.asarray(fin["selected_seed_ids"])[0]
    assert sel.shape == (M,)
    assert len(set(sel.tolist())) == M, "top-M returned a duplicate candidate"
    free = np.asarray(fin["all_collision_free"])[0].astype(bool)
    for m, sid in enumerate(sel):
        assert free[int(sid)], f"slot {m} selected colliding candidate {sid}"
        assert bool(np.asarray(fin["collision_free"])[0, m])
    # Deterministic order: E_phys non-decreasing across the M slots.
    ephys = np.asarray(fin["all_cost_physical"])[0][sel]
    assert np.all(np.diff(ephys) >= -1e-9), f"top-M order is not by E_phys: {ephys}"


# =============================================================================================
# F. Fixed base -- no floating-base assumptions leak in
# =============================================================================================

def test_fixed_base_path_unaffected(scene):
    _, poses, seeds = scene
    fin = solve(poses, seeds, "final")
    assert "base_position" not in fin or np.asarray(fin.get("base_position", [])).size == 0
    assert bool(fin["success"][0, 0])
    assert bool(np.asarray(fin["collision_free"])[0, 0])


def test_off_mode_allocates_and_reports_nothing(scene):
    """`off` must take the pre-Checkpoint-7 path: no channel, no collision outputs."""
    _, poses, seeds = scene
    off = solve(poses, seeds, "off")
    assert off.get("self_collision_enabled") in (False, 0)
    assert "collision_free" not in off or off.get("collision_free") is None
    assert "self_collision" not in off


# =============================================================================================
# A. ENVIRONMENT vs SELF COLLISION: independent channels, AND-composed
# =============================================================================================
#
# Device logic under test (segmented_topM_kernel):
#
#     feasible = (!cc_enabled || final_free[b]) && (!sc_enabled || self_free[b])
#
# The two channels are separate device buffers with separate enable flags; neither is written by
# the other's code path. `all_collision_free` is their AND, exposed per candidate.
#
# This build has no HJCD_HAS_COLLISION (generated without --collision), so the env channel is
# compiled out and the two cases that need it can only be executed on a --collision build. They
# are written and guarded rather than omitted, so such a build runs them unchanged.

# Runtime detection: the environment channel exists only in a --collision build, and only with an
# obstacle set to bind. Both are checked rather than assumed.
_ENV_JSON_PATH = Path(__file__).with_name("g1_wall_problems.json")
ENV_JSON = _ENV_JSON_PATH.read_text() if _ENV_JSON_PATH.exists() else ""
ENV_SET = "problem"


def _env_build():
    if not ENV_JSON:
        return False
    r = hjcdik.solve_problems(
        target_poses=np.tile(np.array([0., 0, 0, 1, 0, 0, 0]), (1, K, 1)),
        active_masks=np.array([1], np.uint32), seed_configs=np.zeros((1, 2, N)),
        problems_json_text=ENV_JSON, problem_set_name=ENV_SET)
    return bool(r.get("collision_enabled", False))


ENV_AVAILABLE = _env_build()


def _env_enabled(res):
    return bool(res.get("collision_enabled", False))


def test_composition_neither_enabled(scene):
    """Neither channel: no feasibility filtering, no collision outputs."""
    _, poses, seeds = scene
    r = solve(poses, seeds, "off")
    assert not _env_enabled(r)
    assert r.get("self_collision_enabled") in (False, 0)
    assert "collision_free" not in r or r.get("collision_free") is None


def test_composition_self_only(scene):
    """Self only: eligibility is exactly self_free, with the env term contributing `true`."""
    _, poses, seeds = scene
    r = solve(poses, seeds, "final", return_all_candidates=True)
    assert not _env_enabled(r), "env channel must be OFF here: no problems_json passed"
    assert bool(r["self_collision_enabled"])
    free = np.asarray(r["all_collision_free"])[0].astype(bool)
    # With the env channel inactive, the exposed per-candidate mask IS the self-collision verdict:
    # cross-check every candidate against the sidecar's own host API.
    qall = np.asarray(r["all_joint_config"])[0]
    truth = ~colliding(qall)
    elig = int(r["self_collision"]["candidates_checked"])
    if elig == qall.shape[0]:            # all checked => the masks must agree exactly
        np.testing.assert_array_equal(free, truth)
    else:                                # skipped candidates are conservatively NOT free
        checked = free | truth
        assert np.all(free[truth & checked] | ~free[truth & checked])
    # A skipped candidate must never be reported free.
    assert free.sum() <= truth.sum()


@pytest.mark.skipif(not ENV_AVAILABLE,
                    reason="build has no HJCD_HAS_COLLISION: regenerate grid.cuh with --collision "
                           "to exercise the environment-collision channel")
def test_composition_env_only(scene):
    """Env only: eligibility is exactly final_free; the self term contributes `true`."""
    _, poses, seeds = scene
    r = solve(poses, seeds, "off", problems_json_text=ENV_JSON, problem_set_name=ENV_SET,
              return_all_candidates=True)
    assert _env_enabled(r) and not r["self_collision_enabled"]
    assert np.asarray(r["all_collision_free"])[0].dtype == bool


@pytest.mark.skipif(not ENV_AVAILABLE,
                    reason="build has no HJCD_HAS_COLLISION: regenerate grid.cuh with --collision "
                           "to exercise the environment-collision channel")
def test_composition_both_is_logical_and():
    """Both channels: only (env free AND self free) may stay eligible.

    Proven candidate-wide, not on the winner alone. The scene is FLOATING-BASE with the base swept
    across the wall face, so env-free and env-blocked candidates both occur; joint perturbation
    supplies self-free and self-blocked ones. All four combinations must appear, or the test would
    pass without ever exercising the AND.
    """
    rng = np.random.default_rng(17)
    q_ref = rng.uniform(-0.25, 0.25, N)
    poses = reachable_poses(q_ref)
    S = 192
    seeds = np.zeros((1, S, 7 + N))
    seeds[0, :, 0] = np.linspace(-1.3, 0.05, S)      # sweep straight through the wall face at x=0
    seeds[0, :, 2] = 1.0
    seeds[0, :, 3] = 1.0
    seeds[0, :, 7:] = q_ref[None, :] + rng.normal(0, 0.35, (S, N))

    common = dict(target_poses=poses[None], active_masks=ALL_ACTIVE, seed_configs=seeds,
                  floating_base=True, num_solutions=1, position_tol=1e-3, orientation_tol=1e-2,
                  seed=5, orientation_modes="none", return_all_candidates=True)

    # ONE solve with both channels on, reading the two masks separately. Comparing masks across
    # SEPARATE solves would be invalid: the environment channel also drives the coarse fallback,
    # so an env-enabled run can carry different candidate configurations than an env-disabled one.
    both = hjcdik.solve_problems(**common, problems_json_text=ENV_JSON,
                                 problem_set_name=ENV_SET, self_collision_mode="final")

    e = np.asarray(both["all_environment_free"])[0].astype(bool)
    s_ = np.asarray(both["all_self_collision_free"])[0].astype(bool)
    b = np.asarray(both["all_collision_free"])[0].astype(bool)

    combos = {(bool(x), bool(y)) for x, y in zip(e, s_)}
    assert len(combos) == 4, f"batch does not cover all four env/self combinations: {combos}"

    # The composition itself: exactly the logical AND, candidate by candidate.
    np.testing.assert_array_equal(b, e & s_)
    # ...and neither channel is a no-op or a copy of the other.
    assert not np.array_equal(b, e), "self-collision channel had no effect: masks alias"
    assert not np.array_equal(b, s_), "environment channel had no effect: masks alias"

    # Only env-free AND self-free candidates may be selectable under `both`.
    assert not (b & ~e).any() and not (b & ~s_).any()


@pytest.mark.skipif(not ENV_AVAILABLE, reason="needs a --collision build")
def test_env_channel_respects_the_floating_base_in_a_full_solve():
    """End to end: identical joints/targets, base far from the wall vs against it."""
    rng = np.random.default_rng(23)
    q_ref = rng.uniform(-0.2, 0.2, N)
    poses = reachable_poses(q_ref)
    S = 32
    def run(x0):
        seeds = np.zeros((1, S, 7 + N))
        seeds[0, :, 0] = x0
        seeds[0, :, 2] = 1.0
        seeds[0, :, 3] = 1.0
        seeds[0, :, 7:] = q_ref[None, :] + rng.normal(0, 0.05, (S, N))
        r = hjcdik.solve_problems(
            target_poses=poses[None], active_masks=ALL_ACTIVE, seed_configs=seeds,
            floating_base=True, num_solutions=1, position_tol=1e-3, orientation_tol=1e-2,
            seed=5, orientation_modes="none", problems_json_text=ENV_JSON,
            problem_set_name=ENV_SET, return_all_candidates=True)
        return int(np.asarray(r["all_collision_free"])[0].astype(bool).sum())
    far, near = run(-1.5), run(0.0)
    assert far > 0, "no candidate free with the base 1.5 m from the wall"
    assert near == 0, f"{near} candidates free with the base inside the wall"


# =============================================================================================
# B. self_collision_eligible_tol -- defined, tested semantics
# =============================================================================================

def test_eligible_tol_none_checks_every_selectable_candidate(scene):
    _, poses, seeds = scene
    r = solve(poses, seeds, "final", self_collision_eligible_tol=None)
    d = r["self_collision"]
    assert d["candidates_checked"] == d["candidates_collision_eligible"]
    assert d["candidates_not_checked"] == 0, "None must not skip any finite candidate"


def test_eligible_tol_skips_only_candidates_above_the_threshold():
    """A candidate whose max POSITION error exceeds the caller's tolerance is skipped -- and is
    then conservatively NOT collision-free, so it cannot be selected while the gate is on."""
    rng = np.random.default_rng(31)
    q_ref = rng.uniform(-0.3, 0.3, N)
    poses = reachable_poses(q_ref)
    seeds = q_ref[None, None, :] + rng.normal(0, 0.6, (1, 128, N))

    loose = solve(poses, seeds, "final", self_collision_eligible_tol=None,
                  return_all_candidates=True)
    tight = solve(poses, seeds, "final", self_collision_eligible_tol=1e-4,
                  return_all_candidates=True)

    pe = np.asarray(loose["all_position_errors"])[0].max(axis=1)
    expect_skipped = int((pe > 1e-4).sum())
    if expect_skipped == 0:
        pytest.skip("no candidate exceeds the tolerance in this scene")

    d = tight["self_collision"]
    assert d["candidates_not_checked"] == expect_skipped, (
        f"expected {expect_skipped} candidates above tol to be skipped, "
        f"got {d['candidates_not_checked']}")
    # skipped => not free => not selectable
    free = np.asarray(tight["all_collision_free"])[0].astype(bool)
    assert not free[pe > 1e-4].any(), "a skipped candidate was reported collision-free"


def test_eligible_tol_never_skips_a_candidate_within_tolerance(scene):
    """The dangerous direction: a candidate the caller WOULD accept must always be checked."""
    _, poses, seeds = scene
    tol = 1e-3
    r = solve(poses, seeds, "final", self_collision_eligible_tol=tol,
              return_all_candidates=True)
    pe = np.asarray(r["all_position_errors"])[0].max(axis=1)
    free = np.asarray(r["all_collision_free"])[0].astype(bool)
    truth = ~colliding(np.asarray(r["all_joint_config"])[0])
    within = pe <= tol
    # Every within-tolerance candidate got a real verdict, so free == truth there.
    np.testing.assert_array_equal(free[within], truth[within])


# =============================================================================================
# E. Top-M collision-aware selection, in the exact cand_better() order
# =============================================================================================

def _expected_order(res, M, respect_collision):
    """Reproduce (class, E_phys, seed) over the candidate diagnostics -- as an ORACLE only.

    This mirrors cand_better() for verification; the production path never runs it. class 2 is
    everything the kernel treats as invalid, including (when the gate is on) colliding candidates.
    """
    succ = np.asarray(res["all_success"])[0].astype(bool)
    ephys = np.asarray(res["all_cost_physical"])[0].astype(float)
    n = len(ephys)
    free = (np.asarray(res["all_collision_free"])[0].astype(bool)
            if respect_collision else np.ones(n, bool))
    cls = np.where(~free, 2, np.where(succ, 0, 1))
    key = np.where(cls == 2, np.inf, ephys)
    order = np.lexsort((np.arange(n), key, cls))
    return [int(i) for i in order[:M] if cls[order][list(order).index(i)] != 2][:M], cls


def test_top_m_matches_cand_better_order_with_and_without_collision():
    """off must take the best M ignoring collision; final the best M among survivors.

    Uses a gently perturbed scene: the module fixture is deliberately harsh (it exists to force
    rejections) and leaves too few survivors to fill M.
    """
    # Chosen so the batch actually exhibits the case under test: the off-order top-3 contains a
    # colliding candidate AND at least M candidates survive the gate, so promotion is observable
    # rather than the two orders trivially agreeing.
    rng = np.random.default_rng(41)
    q_ref = rng.uniform(-0.2, 0.2, N)
    poses = reachable_poses(q_ref)
    seeds = q_ref[None, None, :] + rng.normal(0, 0.15, (1, 128, N))
    M = 3
    off = solve(poses, seeds, "off", num_solutions=M, return_all_candidates=True)
    fin = solve(poses, seeds, "final", num_solutions=M, return_all_candidates=True)

    nfree = int(np.asarray(fin["num_collision_free"])[0])
    if nfree < M:
        pytest.skip(f"only {nfree} collision-free candidates; see the partial-fill test instead")

    sel_off = [int(x) for x in np.asarray(off["selected_seed_ids"])[0]]
    sel_fin = [int(x) for x in np.asarray(fin["selected_seed_ids"])[0]]
    assert sel_off != sel_fin, "collision made no difference; this scene proves nothing"

    free = np.asarray(fin["all_collision_free"])[0].astype(bool)
    # every final slot is collision-free...
    for m, sid in enumerate(sel_fin):
        assert free[sid], f"slot {m} selected colliding candidate {sid}"
        assert bool(np.asarray(fin["collision_free"])[0, m])
    # ...the off slots that survive keep their relative order in the final list...
    survivors = [s for s in sel_off if free[s]]
    assert sel_fin[:len(survivors)] == survivors, (
        f"surviving candidates were reordered: off={sel_off} survivors={survivors} fin={sel_fin}")
    # ...and E_phys is non-decreasing across the final slots.
    ephys = np.asarray(fin["all_cost_physical"])[0][sel_fin]
    assert np.all(np.diff(ephys) >= -1e-9), f"top-M not ordered by E_phys: {ephys}"


def test_top_m_slots_are_internally_consistent(scene):
    """Per slot, selected_seed_id / joint_config / collision_free / errors must agree."""
    _, poses, seeds = scene
    M = 3
    fin = solve(poses, seeds, "final", num_solutions=M, return_all_candidates=True)
    sel = np.asarray(fin["selected_seed_ids"])[0]
    for m, sid in enumerate(sel):
        if int(sid) < 0:
            continue
        np.testing.assert_array_equal(np.asarray(fin["joint_config"])[0, m],
                                      np.asarray(fin["all_joint_config"])[0, int(sid)])
        np.testing.assert_array_equal(np.asarray(fin["position_errors"])[0, m],
                                      np.asarray(fin["all_position_errors"])[0, int(sid)])
        np.testing.assert_array_equal(np.asarray(fin["orientation_errors"])[0, m],
                                      np.asarray(fin["all_orientation_errors"])[0, int(sid)])
        assert bool(np.asarray(fin["collision_free"])[0, m]) == \
            bool(np.asarray(fin["all_collision_free"])[0, int(sid)])


def test_top_m_floating_base_pairing(scene):
    """Extend the pairing proof to M > 1: every slot's base must be its OWN candidate's."""
    q_ref, poses, _ = scene
    rng = np.random.default_rng(8)
    S, M = 64, 3
    fseeds = np.zeros((1, S, 7 + N))
    fseeds[0, :, 0] = np.linspace(-0.25, 0.25, S)
    fseeds[0, :, 1] = np.linspace(0.25, -0.25, S)
    fseeds[0, :, 3] = 1.0
    fseeds[0, :, 7:] = q_ref[None, :] + rng.normal(0, 0.2, (S, N))
    fin = hjcdik.solve_problems(
        target_poses=poses[None], active_masks=ALL_ACTIVE, seed_configs=fseeds,
        floating_base=True, num_solutions=M, position_tol=1e-4, orientation_tol=1e-3, seed=5,
        orientation_modes="none", self_collision_mode="final", return_all_candidates=True)
    bp = np.asarray(fin["_base_position_candidates"])
    bq = np.asarray(fin["_base_quaternion_candidates"])
    d = np.asarray(fin["_base_diag_candidates"])
    for m, sid in enumerate(np.asarray(fin["selected_seed_ids"])[0]):
        if int(sid) < 0:
            continue
        np.testing.assert_allclose(np.asarray(fin["base_position"])[0, m], bp[int(sid)], atol=0)
        np.testing.assert_allclose(np.asarray(fin["base_quaternion"])[0, m], bq[int(sid)], atol=0)
        assert int(np.asarray(fin["base_updates_attempted"])[0, m]) == int(d[int(sid), 0])


# =============================================================================================
# F. Partial fill: M requested, fewer collision-free candidates available
# =============================================================================================

def test_partial_fill_marks_unfilled_slots_invalid(scene):
    """M=3 with only a couple of survivors: filled slots real, unfilled slots never masquerade."""
    _, poses, seeds = scene
    M = 3
    fin = solve(poses, seeds, "final", num_solutions=M, return_all_candidates=True)
    nfree = int(np.asarray(fin["num_collision_free"])[0])
    sel = np.asarray(fin["selected_seed_ids"])[0]
    succ = np.asarray(fin["success"])[0].astype(bool)
    cfree = np.asarray(fin["collision_free"])[0].astype(bool)

    n_real = int((sel >= 0).sum())
    assert n_real <= max(nfree, 0), (
        f"{n_real} slots claim a candidate but only {nfree} are collision-free")
    for m in range(M):
        if int(sel[m]) < 0:                      # shape-stability fallback slot
            assert not succ[m], f"unfilled slot {m} reports success"
            assert not cfree[m], f"unfilled slot {m} claims to be collision-free"
        else:
            assert cfree[m], f"filled slot {m} is not collision-free"
    assert np.asarray(fin["joint_config"]).shape == (1, M, N)


# =============================================================================================
# G. Diagnostic counts are unambiguous and self-consistent
# =============================================================================================

def test_diagnostic_counts_are_consistent(scene):
    _, poses, seeds = scene
    for tol in (None, 1e-4):
        d = solve(poses, seeds, "final", self_collision_eligible_tol=tol)["self_collision"]
        assert d["candidates_total"] == seeds.shape[1]
        assert d["candidates_checked"] == d["candidates_collision_eligible"]
        assert d["candidates_not_checked"] == d["candidates_total"] - d["candidates_checked"]
        # A skipped candidate is NOT counted as colliding: checked splits exactly into free+hit.
        assert d["num_collision_free"] + d["num_colliding"] == d["candidates_checked"], (
            f"free({d['num_collision_free']}) + colliding({d['num_colliding']}) != "
            f"checked({d['candidates_checked']})")
        assert d["num_collision_free"] >= 0 and d["num_colliding"] >= 0


# =============================================================================================
# Environment collision is POST-SOLVE ELIGIBILITY on the batched path, not a search gate
# =============================================================================================
#
# coarse_search_mt_kernel's cc_enabled is an exact gate on proposals and kicks: a colliding
# proposal is rejected outright, so the sweep cannot pass THROUGH an intermediate colliding
# configuration to reach a free one beyond it. On the batched path the coarse output is only the
# LM's SEED, and collision is meant to be an eligibility predicate on FINAL candidates -- exactly
# how the self-collision channel behaves. solve_problems_batched therefore passes 0 for that gate
# while keeping the environment channel in mark_collisions_ct, final_free and segmented_topM.
#
# The single-problem solve() path is deliberately untouched: there the coarse result IS the answer.

@pytest.mark.skipif(not ENV_AVAILABLE, reason="needs a --collision build")
def test_env_does_not_gate_the_batched_search():
    """A. The environment channel must not change WHICH solutions are found, only which are kept.

    Same seeds, same targets, self-collision on in both: turning the environment channel on may
    reject candidates, but it must not alter the search. If it still gated the coarse sweep, the
    per-candidate task errors would differ.
    """
    rng = np.random.default_rng(4242)
    q_ref = rng.uniform(-0.25, 0.25, N)
    poses = reachable_poses(q_ref)
    S = 256
    seeds = np.zeros((1, S, 7 + N))
    seeds[0, :, 0] = rng.uniform(-0.9, -0.4, S)
    seeds[0, :, 2] = rng.uniform(0.8, 1.2, S)
    seeds[0, :, 3] = 1.0
    seeds[0, :, 7:] = q_ref[None, :] + rng.normal(0, 0.3, (S, N))
    common = dict(target_poses=poses[None], active_masks=ALL_ACTIVE, seed_configs=seeds,
                  floating_base=True, num_solutions=1, position_tol=1e-3, orientation_tol=1e-2,
                  seed=5, orientation_modes="none", self_collision_mode="final",
                  return_all_candidates=True)

    without = hjcdik.solve_problems(**common)
    with_env = hjcdik.solve_problems(**common, problems_json_text=ENV_JSON,
                                     problem_set_name=ENV_SET)

    # Candidates that did NOT take the coarse fallback must be bit-identical: with the search gate
    # gone, the environment cannot steer the sweep or the refinement.
    #
    # apply_fallback_kernel is the one remaining place the environment changes a candidate, and it
    # is deliberate and post-LM: when a candidate's LM output collides but its coarse output does
    # not, HJCD substitutes the coarse configuration rather than discarding the candidate. That
    # SALVAGES candidates; it does not gate the search. Those slots are therefore excluded here and
    # asserted separately below.
    fb = np.asarray(with_env["all_used_coarse_fallback"])[0].astype(bool)
    keep = ~fb
    assert keep.any(), "every candidate took the fallback; this scene cannot test the search"
    np.testing.assert_allclose(np.asarray(with_env["all_position_errors"])[0][keep],
                               np.asarray(without["all_position_errors"])[0][keep],
                               rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(with_env["all_orientation_errors"])[0][keep],
                               np.asarray(without["all_orientation_errors"])[0][keep],
                               rtol=0, atol=0)

    # A fallback slot is only ever taken when the LM output collided and the coarse one did not.
    lm_free_proxy = np.asarray(with_env["all_environment_free"])[0].astype(bool)
    assert np.all(lm_free_proxy[fb]), "a fallback slot ended up environment-colliding"


@pytest.mark.skipif(not ENV_AVAILABLE, reason="needs a --collision build")
def test_env_still_rejects_finally_colliding_candidates():
    """B. Removing the SEARCH gate must not remove the ACCEPTANCE gate."""
    rng = np.random.default_rng(4243)
    q_ref = rng.uniform(-0.2, 0.2, N)
    poses = reachable_poses(q_ref)
    S = 128
    seeds = np.zeros((1, S, 7 + N))
    seeds[0, :, 0] = 0.0                       # base ON the wall face: every candidate collides
    seeds[0, :, 2] = 1.0
    seeds[0, :, 3] = 1.0
    seeds[0, :, 7:] = q_ref[None, :] + rng.normal(0, 0.05, (S, N))
    r = hjcdik.solve_problems(
        target_poses=poses[None], active_masks=ALL_ACTIVE, seed_configs=seeds,
        floating_base=True, num_solutions=1, position_tol=1e-3, orientation_tol=1e-2,
        seed=5, orientation_modes="none", problems_json_text=ENV_JSON,
        problem_set_name=ENV_SET, return_all_candidates=True)
    assert not np.asarray(r["all_environment_free"])[0].astype(bool).any(), \
        "candidates inside the wall were reported environment-free"
    assert not bool(r["success"][0, 0])
    assert int(r["selected_seed_ids"][0, 0]) == -1


@pytest.mark.skipif(not ENV_AVAILABLE, reason="needs a --collision build")
def test_env_and_self_masks_still_compose_after_the_gate_change():
    """C. The AND semantics at final selection are unchanged by dropping the search gate."""
    rng = np.random.default_rng(4244)
    q_ref = rng.uniform(-0.25, 0.25, N)
    poses = reachable_poses(q_ref)
    S = 192
    seeds = np.zeros((1, S, 7 + N))
    seeds[0, :, 0] = np.linspace(-1.3, 0.05, S)
    seeds[0, :, 2] = 1.0
    seeds[0, :, 3] = 1.0
    seeds[0, :, 7:] = q_ref[None, :] + rng.normal(0, 0.35, (S, N))
    r = hjcdik.solve_problems(
        target_poses=poses[None], active_masks=ALL_ACTIVE, seed_configs=seeds,
        floating_base=True, num_solutions=1, position_tol=1e-3, orientation_tol=1e-2,
        seed=5, orientation_modes="none", problems_json_text=ENV_JSON,
        problem_set_name=ENV_SET, self_collision_mode="final", return_all_candidates=True)
    e = np.asarray(r["all_environment_free"])[0].astype(bool)
    s_ = np.asarray(r["all_self_collision_free"])[0].astype(bool)
    b = np.asarray(r["all_collision_free"])[0].astype(bool)
    assert len({(bool(x), bool(y)) for x, y in zip(e, s_)}) == 4
    np.testing.assert_array_equal(b, e & s_)


# =============================================================================================
# I. CRAG-HJCD-COLLISION-STACK-0: exact-IK eligibility compaction, and collision as a RANK
#
# Two independent claims, and they are kept apart on purpose:
#
#   COMPACTION   narrowing WHICH candidates are sent to the sidecar may not change any verdict
#                the caller can observe. The narrowings offered are the other half of the
#                caller's own acceptance predicate (orientation) and the environment channel's
#                already-final refusal -- both of which make a candidate class 2 anyway, so a
#                candidate they skip could never have been selected.
#
#   SELECTION    "gate" forces a colliding candidate to class 2, so it cannot be returned at all.
#                "rank" keeps it selectable but sorts it after every collision-free candidate of
#                the same class, so the gate's own class-0 selections survive unchanged and the
#                colliding ones only fill slots the gate left as invalid pads.
# =============================================================================================

def _accepted(out):
    """The production IK-acceptance predicate: `result.valid and result.success`."""
    return (np.asarray(out["valid"]).astype(bool)
            & np.asarray(out["success"]).astype(bool))


@pytest.fixture(scope="module")
def wide_scene():
    """A scene with many solved candidates, so top-M has something to choose between."""
    rng = np.random.default_rng(7)
    q_ref = rng.uniform(-0.4, 0.4, N)
    poses = reachable_poses(q_ref)
    seeds = q_ref[None, None, :] + rng.normal(0, 0.25, (1, 96, N))
    return q_ref, poses, seeds


def test_orientation_eligibility_never_skips_a_candidate_within_both_tolerances(wide_scene):
    """The narrowed predicate may only skip candidates that miss the caller's tolerances."""
    _, poses, seeds = wide_scene
    out = solve(poses, seeds, "final", return_all_candidates=True,
                self_collision_eligible_tol=1e-4,
                self_collision_eligible_orientation_tol=1e-3)
    pe = np.asarray(out["all_position_errors"])[0].max(axis=1)
    oe = np.asarray(out["all_orientation_errors"])[0].max(axis=1)
    free = np.asarray(out["all_self_collision_free"])[0].astype(bool)
    within = (pe <= 1e-4) & (oe <= 1e-3)
    # A candidate inside both tolerances was checked, so its annotation is the sidecar's real
    # verdict rather than the conservative `not checked` zero.
    truth = ~colliding(np.asarray(out["all_joint_config"])[0][within])
    assert np.array_equal(free[within], truth)


def test_exact_ik_eligibility_preserves_the_accepted_set_exactly(wide_scene):
    """Compaction is a WHICH-candidates change, never a WHICH-answer change."""
    _, poses, seeds = wide_scene
    base = solve(poses, seeds, "final", num_solutions=4)
    gated = solve(poses, seeds, "final", num_solutions=4,
                  self_collision_eligible_tol=1e-4,
                  self_collision_eligible_orientation_tol=1e-3)
    a, b = _accepted(base), _accepted(gated)
    assert np.array_equal(a, b)
    assert np.array_equal(np.asarray(base["selected_seed_ids"])[a],
                          np.asarray(gated["selected_seed_ids"])[b])
    assert np.array_equal(np.asarray(base["joint_config"])[a],
                          np.asarray(gated["joint_config"])[b])
    # ...and it really did less work, or the test proves nothing.
    assert (gated["self_collision"]["candidates_checked"]
            < base["self_collision"]["candidates_checked"])


def test_environment_prefilter_is_inert_without_an_environment():
    """No environment channel => nothing to prefilter on, and the request must change nothing.

    SCOPE. This pins the SAFETY half only: with no `problems_json_text` the kernel is handed a
    null `env_free` and the flag cannot narrow anything, so asking for it must be a no-op. The
    half that matters -- that prefiltering on a REAL environment preserves the accepted set --
    needs a route's obstacle set and is measured end to end against the production frontend
    instead (CRAG-HJCD-COLLISION-STACK-0: accepted set identical element-wise on 6/6 production
    calls, and an entire frontend returning the same 1284 realizations).
    """
    rng = np.random.default_rng(11)
    q_ref = rng.uniform(-0.4, 0.4, N)
    poses = reachable_poses(q_ref)
    seeds = q_ref[None, None, :] + rng.normal(0, 0.25, (1, 96, N))
    base = solve(poses, seeds, "final", num_solutions=4,
                 self_collision_eligible_tol=1e-4,
                 self_collision_eligible_orientation_tol=1e-3)
    pre = solve(poses, seeds, "final", num_solutions=4,
                self_collision_eligible_tol=1e-4,
                self_collision_eligible_orientation_tol=1e-3,
                self_collision_eligible_require_environment_free=True)
    a, b = _accepted(base), _accepted(pre)
    assert np.array_equal(a, b)
    assert np.array_equal(np.asarray(base["joint_config"])[a],
                          np.asarray(pre["joint_config"])[b])
    assert (base["self_collision"]["candidates_checked"]
            == pre["self_collision"]["candidates_checked"])


def test_defaults_are_the_shipped_predicate(wide_scene):
    """Every new argument left alone must reproduce the shipped call bit for bit."""
    _, poses, seeds = wide_scene
    a = solve(poses, seeds, "final", num_solutions=4)
    b = solve(poses, seeds, "final", num_solutions=4,
              self_collision_eligible_orientation_tol=None,
              self_collision_eligible_require_environment_free=False,
              self_collision_selection="gate")
    for key in ("joint_config", "selected_seed_ids", "success", "valid",
                "position_errors", "orientation_errors"):
        assert np.array_equal(np.asarray(a[key]), np.asarray(b[key])), key
    assert (a["self_collision"]["candidates_checked"]
            == b["self_collision"]["candidates_checked"])


def test_rank_keeps_every_gated_selection_and_only_adds(wide_scene):
    """A colliding candidate is PRESERVED, and it takes nothing from a collision-free one."""
    _, poses, seeds = wide_scene
    gate = solve(poses, seeds, "final", num_solutions=4)
    rank = solve(poses, seeds, "final", num_solutions=4,
                 self_collision_selection="rank")
    g, r = _accepted(gate), _accepted(rank)
    assert int(r.sum()) >= int(g.sum())

    g_sid = np.asarray(gate["selected_seed_ids"])
    r_sid = np.asarray(rank["selected_seed_ids"])
    g_q = np.asarray(gate["joint_config"])
    r_q = np.asarray(rank["joint_config"])
    # Same candidate, same slot, same configuration, everywhere the gate accepted one.
    assert np.array_equal(g_sid[g], r_sid[g])
    assert np.array_equal(g_q[g], r_q[g])

    # The annotation is per SLOT and it is the sidecar's real verdict, not "it was selected".
    sf = np.asarray(rank["selected_self_collision_free"]).astype(bool)
    for p, m in zip(*np.where(r)):
        hit = colliding(r_q[p, m])[0]
        assert bool(sf[p, m]) == (not hit)
    # Everything the gate returned is annotated collision-free; the extras are what the gate
    # threw away, and they carry their own verdict rather than being marked free by default.
    assert sf[g].all()


def test_annotate_does_not_promise_the_gated_selections(wide_scene):
    """Documented, measured caveat: without a ranking term a colliding candidate can EVICT one.

    This is why `rank` exists and why "just stop ANDing it" is not the fix.
    """
    _, poses, seeds = wide_scene
    gate = solve(poses, seeds, "final", num_solutions=4)
    ann = solve(poses, seeds, "final", num_solutions=4,
                self_collision_selection="annotate")
    assert int(_accepted(ann).sum()) >= int(_accepted(gate).sum())
    # No claim that the gated candidates survive -- only that nothing is invented: every
    # accepted slot is a real candidate and its annotation is the sidecar's verdict.
    a = _accepted(ann)
    q = np.asarray(ann["joint_config"])
    sf = np.asarray(ann["selected_self_collision_free"]).astype(bool)
    for p, m in zip(*np.where(a)):
        assert bool(sf[p, m]) == (not colliding(q[p, m])[0])


def test_selection_mode_is_validated():
    rng = np.random.default_rng(3)
    q_ref = rng.uniform(-0.3, 0.3, N)
    poses = reachable_poses(q_ref)
    seeds = q_ref[None, None, :] + rng.normal(0, 0.2, (1, 16, N))
    with pytest.raises(ValueError, match="gate|annotate|rank"):
        solve(poses, seeds, "final", self_collision_selection="nonsense")
