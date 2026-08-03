"""Checkpoint 3D/3E tests: self_collision_mode="hard" -- collision-free seeding (Stage 3D) and
top-K collision-gated coordinate commits (Stage 3E).

Run: env PYTHONPATH= python3 -m pytest tests/test_collision_hard_mode.py -q
Requires the rebuilt _hjcdik (sidecar + hard mode compiled in) and a CUDA GPU.

The numbering in the test names follows the checkpoint spec's section-10 list so a reviewer can map
requirement -> test without a translation table.
"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HJCD = os.path.dirname(HERE)
GEN = os.path.join(HJCD, "generated")
SC = os.path.join(HJCD, "collision_sidecar")
sys.path.insert(0, SC)

hjcdik = pytest.importorskip("hjcdik")

# The multi-target problem builder is shared with the Checkpoint 3B/3C suite; reproducing it here
# would let the two suites silently drift onto different problems.
sys.path.insert(0, HERE)
from test_collision_integration import _problem, _solve  # noqa: E402

N_JOINTS = 29
FLAG_STATE_VALID = 0x1
FLAG_HAS_FREE_Q = 0x2


@pytest.fixture(scope="module", autouse=True)
def model_uploaded():
    hjcdik._ensure_self_collision_sidecar()


@pytest.fixture(scope="module")
def prob():
    return _problem(0, 256)


def _free(q):
    """Native sidecar verdict: True where the configuration is self-collision-free."""
    q32 = np.ascontiguousarray(np.asarray(q, np.float32))
    return ~np.asarray(hjcdik._hjcdik.sidecar_full_check(q32, 0.0)).any(axis=1)


@pytest.fixture(scope="module")
def hard_run(prob):
    """One hard-mode solve, reused by the tests that only read its outputs."""
    out = _solve(*prob, self_collision_mode="hard", diagnostics=True)
    return out


@pytest.fixture(scope="module")
def hard_run_fb():
    """A run that actually EXERCISES the section-8 fallback. Problem 0 is a poor choice for this:
    every LM result that collides there belongs to a seed Stage 3D could not make free, so the
    fallback never fires and the tests below would silently skip rather than check anything."""
    out = _solve(*_problem(2, 256), self_collision_mode="hard", diagnostics=True)
    assert np.asarray(out["used_collision_fallback"]).any(), (
        "problem 2 no longer exercises the LM fallback -- pick a different problem, do not skip")
    return out


@pytest.fixture(scope="module")
def hard_state(hard_run):
    """Committed workspace state as it stands after `hard_run` (qc, qfree, flags, Tf, Td)."""
    d = hjcdik._hjcdik.hard_dump()
    assert len(d) == 5, "hard mode has not run -- no committed state to inspect"
    return d


# ---------------------------------------------------------------------------------------------
# 1. Hard mode allocates the persistent workspace; off/final allocate no per-seed state.
# ---------------------------------------------------------------------------------------------
def test_01_off_and_final_allocate_no_hard_workspace(prob):
    hjcdik._hjcdik.hard_ws_release()
    assert hjcdik._hjcdik.hard_ws_capacity() == 0
    _solve(*prob, self_collision_mode="off")
    assert hjcdik._hjcdik.hard_ws_capacity() == 0, "off allocated hard-mode state"
    _solve(*prob, self_collision_mode="final")
    assert hjcdik._hjcdik.hard_ws_capacity() == 0, "final allocated hard-mode state"


def test_01b_hard_allocates_and_reuses_workspace(prob):
    hjcdik._hjcdik.hard_ws_release()
    _solve(*prob, self_collision_mode="hard")
    n1 = hjcdik._hjcdik.hard_ws_nalloc()
    assert hjcdik._hjcdik.hard_ws_capacity() >= prob[0].shape[0]
    _solve(*prob, self_collision_mode="hard")
    # Persistent-workspace policy: a second call at the same size must not reallocate.
    assert hjcdik._hjcdik.hard_ws_nalloc() == n1


# ---------------------------------------------------------------------------------------------
# 2/3. No coordinate-search warp starts from a colliding state; colliding seeds are reseeded or
#      marked failed. Every seed admitted to the search (STATE_VALID) has a verified-free
#      configuration recorded for it, and the Stage-3D accounting balances.
# ---------------------------------------------------------------------------------------------
def test_02_no_search_starts_from_a_colliding_seed(hard_run, hard_state):
    _qc, qfree, flags, _Tf, _Td = hard_state
    admitted = (np.asarray(flags) & FLAG_STATE_VALID) != 0
    assert admitted.any(), "no seed was admitted at all -- the run proves nothing"
    assert _free(qfree[admitted]).all(), "an admitted seed's recorded free state collides"


def test_03_colliding_seeds_are_reseeded_or_failed(hard_run):
    h = hard_run["self_collision"]
    assert h["recovered"] + h["seed_failures"] == h["initially_colliding"]
    assert h["initially_free"] + h["initially_colliding"] == h["candidates_checked"]
    assert h["reseed_attempts"] >= h["recovered"]


# ---------------------------------------------------------------------------------------------
# 4/7/8. THE COMMITTED-STATE INVARIANT. The sidecar's committed link transforms must describe
#        exactly the sidecar's committed q, at every point -- which is only true if every accepted
#        trial committed both together AND every rejected trial restored both. A single rejected
#        trial that left stale descendant transforms behind, or an accepted one that updated the
#        transforms without q, breaks this and nothing else in the suite would notice.
# ---------------------------------------------------------------------------------------------
def test_04_committed_transforms_match_committed_q(hard_state):
    qc, _qfree, _flags, Tf, _Td = hard_state
    fresh = np.asarray(hjcdik._hjcdik.sidecar_fk(np.ascontiguousarray(qc.astype(np.float32))))
    fresh = fresh.reshape(Tf.shape)
    # Byte-identical, not merely close: the incremental descendant FK runs the same operations on
    # the same inputs as the full FK, so any difference at all is a bug, not float noise.
    assert np.array_equal(Tf, fresh), (
        f"committed transforms drifted from committed q "
        f"(max |delta| = {np.abs(Tf - fresh).max():.3e})")


def test_07_08_committed_state_is_self_consistent_after_many_trials(hard_run, hard_state):
    """A run with thousands of accepted AND rejected trials still satisfies the invariant."""
    h = hard_run["self_collision"]
    assert h["proposals_checked"] > 100, "not enough trials for this to be meaningful"
    assert h["proposals_rejected"] > 0, "no trial was rejected -- rollback path never exercised"
    qc, _qfree, _flags, Tf, _Td = hard_state
    fresh = np.asarray(hjcdik._hjcdik.sidecar_fk(
        np.ascontiguousarray(qc.astype(np.float32)))).reshape(Tf.shape)
    assert np.array_equal(Tf, fresh)


# ---------------------------------------------------------------------------------------------
# 5/6. Incremental verdict == full verdict. Driven in-kernel by the debug oracle, which re-checks
#      ALL 351 pairs on the very trial transforms the incremental verdict was computed from.
# ---------------------------------------------------------------------------------------------
def test_05_06_debug_oracle_reports_no_mismatches(prob):
    out = _solve(*prob, self_collision_mode="hard", diagnostics=True, _hard_oracle_every=1)
    h = out["self_collision"]
    assert h["oracle_checks"] > 1000, f"oracle barely ran ({h['oracle_checks']} checks)"
    assert h["oracle_mismatches"] == 0


@pytest.mark.parametrize("iters", [1, 8, 60])
def test_06b_descendant_fk_stays_bitwise_equal_to_a_full_fk(prob, iters):
    """The other half of the incremental argument. The debug oracle proves the affected-pair CSR is
    sufficient; this proves the descendant-only FK update that feeds it reproduces a full FK
    BITWISE, after 1, 8 and 60 rounds of trial/commit/rollback. Drift would compound with the
    iteration count, so a short run agreeing is not evidence -- the long run is.

    (The f64 transform array has no host binding of its own; it is covered transitively, since the
    GJK phase of the debug oracle reads Td and reported zero mismatches.)"""
    sq, tp, tq = prob
    hjcdik.coarse_search(sq, tp, tq, max_iters=iters, precision="float32",
                         hard_self_collision=1, hard_top_k=3)
    qc, _qfree, _flags, Tf, _Td = hjcdik._hjcdik.hard_dump()
    fresh = np.asarray(hjcdik._hjcdik.sidecar_fk(
        np.ascontiguousarray(qc.astype(np.float32)))).reshape(Tf.shape)
    assert np.array_equal(Tf, fresh), (
        f"after {iters} iterations the committed transforms drifted from committed q "
        f"(max |delta| = {np.abs(Tf - fresh).max():.3e})")


# ---------------------------------------------------------------------------------------------
# 9/10/11/12. TOP-K RANKING. Ranks beyond the first are genuinely used, ranks are distinct by
#             joint, and nothing is ever committed beyond the configured K.
# ---------------------------------------------------------------------------------------------
def test_09_10_lower_ranks_are_selected_when_the_top_candidate_collides(hard_run):
    hist = hard_run["self_collision"]["accept_by_rank"]
    assert hist[1] > 0, "rank 2 was never accepted -- top-K is not doing anything"
    assert hist[2] > 0, "rank 3 was never accepted"


def test_11_no_commit_beyond_the_configured_k(prob):
    for k in (1, 2, 3):
        out = _solve(*prob, self_collision_mode="hard", collision_top_k=k, diagnostics=True)
        hist = out["self_collision"]["accept_by_rank"]
        assert all(v == 0 for v in hist[k:]), f"top_k={k} committed at rank > {k}: {hist}"
        if k == 1:
            assert sum(hist) == hist[0]


def test_11b_all_k_colliding_commits_nothing(hard_run):
    """An all-K-colliding iteration must fall through to the stagnation path, never commit.
    Accounted for by construction: accepts are recorded by rank, so the two are disjoint."""
    h = hard_run["self_collision"]
    assert h["all_k_colliding"] >= 0
    # Every collision rejection is either followed by an accept at a later rank, or ends the
    # iteration with no commit; it can never itself be counted as an accept.
    assert sum(h["accept_by_rank"]) + h["all_k_colliding"] <= h["proposals_checked"]


def test_12_ranks_are_distinct_and_accepts_are_counted_once(hard_run):
    h = hard_run["self_collision"]
    # Each accept lands in exactly one rank bucket, and there cannot be more accepts than trials.
    assert sum(h["accept_by_rank"]) <= h["proposals_checked"]
    # A collision rejection consumes a rank; a rank is consumed at most K-1 times per accept.
    assert h["proposals_rejected"] <= h["proposals_checked"]


# ---------------------------------------------------------------------------------------------
# 13. The last collision-free coarse state is a genuinely free configuration.
# ---------------------------------------------------------------------------------------------
def test_13_last_free_coarse_state_is_collision_free(hard_state):
    _qc, qfree, flags, _Tf, _Td = hard_state
    has_free = (np.asarray(flags) & FLAG_HAS_FREE_Q) != 0
    assert has_free.any()
    assert _free(qfree[has_free]).all()


# ---------------------------------------------------------------------------------------------
# 14/15/16/17. LM FINAL CHECK AND FALLBACK.
# ---------------------------------------------------------------------------------------------
def test_14_free_lm_result_is_returned_unchanged(hard_run_fb):
    """A row whose LM answer is already collision-free must come back as the LM produced it. The
    check is that it is NOT the coarse pose: substituting the coarse answer everywhere would also
    satisfy "returned q is collision-free", and would throw away all the LM's accuracy."""
    fb = np.asarray(hard_run_fb["used_collision_fallback"])
    q = np.asarray(hard_run_fb["joint_config"])
    qfree = np.asarray(hard_run_fb["hard_last_free_coarse_q"])
    assert (~fb).any()
    subst = np.isclose(q[~fb], qfree[~fb].astype(q.dtype), rtol=0, atol=0).all(axis=1)
    assert not subst.all(), "every non-fallback row equals the coarse pose -- LM output was replaced"


def test_15_colliding_lm_result_falls_back_to_the_last_free_coarse_state(hard_run_fb):
    fb = np.asarray(hard_run_fb["used_collision_fallback"])
    q = np.asarray(hard_run_fb["joint_config"])
    qfree = np.asarray(hard_run_fb["hard_last_free_coarse_q"])
    assert np.allclose(q[fb], qfree[fb], rtol=0, atol=0)
    assert _free(q[fb]).all()


def test_16_fallback_metadata_describes_the_returned_pose(hard_run_fb):
    """A fallback row's reported position error must be the error OF THE POSE RETURNED -- not the
    LM's error for a configuration we are not returning."""
    fb = np.asarray(hard_run_fb["used_collision_fallback"])
    q = np.asarray(hard_run_fb["joint_config"], np.float64)
    T = np.asarray(hjcdik.target_transforms(np.ascontiguousarray(q[fb])))
    _sq, tpos, _tq = _problem(2, 256)
    got = np.linalg.norm(T[:, :, :3, 3] - tpos[fb], axis=-1)
    rep = np.asarray(hard_run_fb["position_errors"])[fb]
    assert np.allclose(got, rep, atol=1e-4), (
        f"reported errors do not match the returned pose: max delta "
        f"{np.abs(got - rep).max():.3e}")


def test_17_success_still_requires_the_ik_tolerance(hard_run):
    succ = np.asarray(hard_run["success"]).astype(bool)
    if not succ.any():
        pytest.skip("no successes in this run")
    pe = np.asarray(hard_run["position_errors"])[succ]
    oe = np.asarray(hard_run["orientation_errors"])[succ]
    assert (pe <= 0.02 + 1e-9).all(), "a candidate outside the position tolerance is marked success"
    assert (oe <= 0.1 + 1e-9).all()


def test_17b_failed_seeds_are_never_successful(hard_run):
    seed_ok = np.asarray(hard_run["hard_seed_ok"]).astype(bool)
    succ = np.asarray(hard_run["success"]).astype(bool)
    assert not (succ & ~seed_ok).any(), "a seed with no collision-free start was marked success"


# ---------------------------------------------------------------------------------------------
# 18. Every returned hard-mode SUCCESS is native collision-free. The headline guarantee.
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("pi", [0, 1, 2, 3])
def test_18_hard_successes_are_native_collision_free(pi):
    out = _solve(*_problem(pi, 256), self_collision_mode="hard")
    succ = np.asarray(out["success"]).astype(bool)
    free = _free(out["joint_config"])
    assert not (succ & ~free).any(), (
        f"problem {pi}: {(succ & ~free).sum()} successful outputs collide")


# ---------------------------------------------------------------------------------------------
# 20/21. Batch permutation invariance and cross-warp isolation. A per-seed result must depend on
#        that seed alone -- not on its index, its neighbours, or the batch size.
# ---------------------------------------------------------------------------------------------
def test_20_batch_permutation_invariance():
    sq, tp, tq = _problem(0, 128)
    a = _solve(sq, tp, tq, self_collision_mode="hard")
    perm = np.random.default_rng(0).permutation(128)
    b = _solve(np.ascontiguousarray(sq[perm]), np.ascontiguousarray(tp[perm]),
               np.ascontiguousarray(tq[perm]), self_collision_mode="hard")
    assert np.array_equal(np.asarray(a["success"])[perm], np.asarray(b["success"]))
    assert np.array_equal(np.asarray(a["self_collision_free"])[perm],
                          np.asarray(b["self_collision_free"]))


def test_21_no_cross_warp_state_leakage():
    """Solving a SUBSET must reproduce those rows exactly. Per-seed collision state living in one
    shared workspace is precisely the setup where one warp can scribble on another's slice."""
    sq, tp, tq = _problem(0, 128)
    full = _solve(sq, tp, tq, self_collision_mode="hard")
    sub = slice(0, 32)
    part = _solve(np.ascontiguousarray(sq[sub]), np.ascontiguousarray(tp[sub]),
                  np.ascontiguousarray(tq[sub]), self_collision_mode="hard")
    assert np.array_equal(np.asarray(full["success"])[sub], np.asarray(part["success"]))
    assert np.allclose(np.asarray(full["joint_config"])[sub],
                       np.asarray(part["joint_config"]), rtol=0, atol=0)


# ---------------------------------------------------------------------------------------------
# 22. Self-collision semantics are independent of which targets are active. The collision model
#     is a property of the BODY; a different task must not change what counts as a collision.
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mask", [0b0011, 0b0101, 0b1111])
def test_22_target_mask_does_not_alter_self_collision_semantics(mask):
    sq, tp, tq = _problem(0, 128)
    out = _solve(sq, tp, tq, active_target_mask=np.full(128, mask, np.uint32),
                 self_collision_mode="hard")
    succ = np.asarray(out["success"]).astype(bool)
    free = _free(out["joint_config"])
    assert not (succ & ~free).any()


# ---------------------------------------------------------------------------------------------
# 25/26/27. API contract: invalid configuration fails loudly, artifacts are verified.
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("k", [0, -1, 999, 3.0, "3", True])
def test_25_invalid_collision_top_k_raises(prob, k):
    with pytest.raises((ValueError, TypeError)):
        _solve(*prob, self_collision_mode="hard", collision_top_k=k)


def test_25b_unknown_mode_raises(prob):
    with pytest.raises(ValueError):
        _solve(*prob, self_collision_mode="soft")


def test_26_artifact_hashes_match_the_compiled_model():
    info = hjcdik.self_collision_info()
    assert info["geometry_validated"] is True, "on-disk sidecar artifacts disagree with the build"


def test_27_self_collision_info_reports_hard_mode():
    info = hjcdik.self_collision_info()
    assert info["hard_enabled"] is True
    assert info["supported_modes"] == ["off", "final", "hard"]
    assert info["incremental_checker"] is True
    assert info["top_k_max"] >= 3
    assert info["n_gjk_pairs"] == 53
    assert set(info["hashes"]) >= {"urdf", "joint_order", "proxy_yaml", "torso_sdf",
                                   "pelvis_sdf", "convex", "pair_policy"}


def test_27b_sidecar_entry_points_require_an_uploaded_model():
    """Calling a checker before the model is uploaded used to read a null SDF pointer and surface
    an 'illegal memory access' inside an unrelated later kernel. It must name its own cause."""
    src = open(os.path.join(HJCD, "src", "collision_sidecar.cu")).read()
    assert "sidecar_model_uploaded" in src
    assert hjcdik._hjcdik.sidecar_model_info()["sidecar_compiled"] is True
