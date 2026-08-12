"""End-to-end collision invariant for solve(): a colliding configuration is NEVER returned.

THE LEAK THIS CLOSES. The coarse stage is hard-gated, so its answer is feasible. The LM is not: it
refines a feasible coarse state straight back into the shelf. On bookshelf_small_panda with 1000
verified-collision-free seeds, the coarse output was 0/1000 colliding and the LM output 63/1000 --
and solve() returned the LM result unchecked.

The bypass is reproduced INSIDE this file (test_the_unfiltered_pipeline_still_leaks): running
coarse_search() then refine() by hand is exactly the old pipeline, and it still produces the
colliding outputs. So these tests are not passing by accident -- the failure they guard against is
demonstrably still there the moment you skip the filter.

THE FIX. After the LM, every final candidate is validated with the exact grid_collision::config_free
evaluator. A colliding LM candidate is replaced, per seed, by that seed's own collision-free coarse
result. Because a fallback carries its COARSE task cost -- which is worse -- a better-task-but-
colliding LM candidate can never out-rank a feasible one under the unchanged argmin(cost) ranking.
No finite penalty is used anywhere: feasibility is a hard filter on the candidate set.

Panda-only: the gate is compiled in only when grid.cuh was generated with --collision.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import hjcdik

REPO = Path(__file__).resolve().parents[1]
K = hjcdik.num_targets()
LIM = hjcdik.joint_limits()
LO, HI = LIM[:, 0], LIM[:, 1]
SET = "bookshelf_small_panda"
PTOL, OTOL = 1e-4, 1e-3

pytestmark = pytest.mark.skipif(K != 1, reason="collision build is the single-target Panda")


@pytest.fixture(scope="module")
def probs():
    return (REPO / "tests" / "mb_problems.json").read_text()


@pytest.fixture(scope="module")
def goal(probs):
    p = json.loads(probs)["problems"][SET]
    inst = p[0] if isinstance(p, list) else p
    gp = inst["goal_pose"]
    pos = np.asarray(gp.get("position_xyz", gp.get("position")), dtype=float)
    quat = np.asarray(gp.get("quaternion_wxyz", gp.get("quat_wxyz")), dtype=float)
    return pos, quat / np.linalg.norm(quat)


@pytest.fixture(scope="module")
def seeds(probs):
    """1000 VERIFIED collision-free seeds -- enough that the old 63/1000 leak is reproduced every
    run, not stumbled upon."""
    rng = np.random.default_rng(5)
    cand = np.stack([rng.uniform(LO, HI) for _ in range(4000)])
    free = cand[hjcdik.collision_free(cand, probs, SET, 0)]
    s = np.resize(free, (1000, len(LO)))
    assert np.all(hjcdik.collision_free(s, probs, SET, 0)), "fixture seeds are not collision-free"
    return s


def _targets(goal, B):
    pos, quat = goal
    return (np.repeat(pos[None, None, :], B, axis=0),
            np.repeat(quat[None, None, :], B, axis=0))


def _solve(s, goal, probs=None, precision="float32", **kw):
    P, Q = _targets(goal, len(s))
    base = dict(active_target_mask=np.ones(len(s), dtype=np.uint32),
                position_tol=PTOL, orientation_tol=OTOL,
                coarse_iters=120, lm_iters=60, precision=precision)
    if probs is not None:
        base.update(problems_json_text=probs, problem_set_name=SET, problem_idx=0)
    base.update(kw)
    return hjcdik.solve(s, P, Q, **base)


def _free(q, probs):
    return hjcdik.collision_free(np.asarray(q, dtype=np.float64), probs, SET, 0)


def _unfiltered(s, goal, probs):
    """solve()'s pipeline WITHOUT the final filter, reproduced faithfully.

    Note the dispatch: Panda has ONE active target, so popcount == 1 and coarse_mode="auto" sends it
    LM-ONLY -- the LM is seeded from the raw seeds, not from the coarse output. The coarse stage runs
    only to manufacture a collision-free fallback. Reconstructing this as coarse -> LM would be a
    different pipeline and would not match what solve() actually does.
    """
    P, Q = _targets(goal, len(s))
    m = np.ones(len(s), dtype=np.uint32)
    c = hjcdik.coarse_search(s, P, Q, active_target_mask=m, position_tol=PTOL,
                             orientation_tol=OTOL, max_iters=120,
                             problems_json_text=probs, problem_set_name=SET, problem_idx=0)
    lm = hjcdik.refine(s, P, Q, active_target_mask=m, position_tol=PTOL,
                       orientation_tol=OTOL, max_iters=60)
    return c, lm


# --- the leak is real, and still there without the filter --------------------------------------
def test_the_unfiltered_pipeline_still_leaks(probs, seeds, goal):
    """coarse_search() -> refine() by hand IS the old, unfiltered pipeline. It must still leak;
    otherwise these tests would be guarding against nothing."""
    c, lm = _unfiltered(seeds, goal, probs)
    assert int((~_free(c["joint_config"], probs)).sum()) == 0, "the gated coarse stage leaked"
    n_bad = int((~_free(lm["joint_config"], probs)).sum())
    assert n_bad > 20, (
        f"the unfiltered LM returned only {n_bad} colliding configs -- the fixture no longer "
        f"reproduces the leak, so the filter tests below prove nothing")
    print(f"\n  unfiltered pipeline: coarse 0/1000 colliding, LM {n_bad}/1000 colliding")


# --- 4/9. THE exit condition --------------------------------------------------------------------
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_solve_never_returns_a_colliding_configuration(probs, seeds, goal, precision):
    out = _solve(seeds, goal, probs, precision=precision)
    bad = int((~_free(out["joint_config"], probs)).sum())
    assert out["n_lm_colliding"] > 20, "the LM did not collide at all -- the filter is untested"
    assert bad == 0, f"precision={precision}: solve() returned {bad}/{len(seeds)} COLLIDING configs"
    assert np.all(out["collision_free"])


# --- 1/2/3. normal path, rejection, fallback retention -------------------------------------------
def test_collision_free_lm_results_are_returned_unchanged(probs, seeds, goal):
    """A feasible LM result is kept as-is -- the filter must not disturb the majority path."""
    out = _solve(seeds, goal, probs)
    keep = out["lm_collision_free"]
    assert keep.sum() > 0.8 * len(seeds), "most LM results should be feasible"
    _c, lm = _unfiltered(seeds, goal, probs)
    np.testing.assert_array_equal(out["joint_config"][keep], lm["joint_config"][keep])
    assert not out["used_coarse_fallback"][keep].any(), "a feasible LM result took the fallback"


def test_colliding_lm_results_are_rejected_and_replaced_by_their_coarse_state(probs, seeds, goal):
    out = _solve(seeds, goal, probs)
    fb = out["used_coarse_fallback"]
    assert fb.sum() > 0, "no fallback fired -- the rejection path is untested"
    # every fallback candidate corresponds to a colliding LM result...
    assert np.all(~out["lm_collision_free"][fb])
    # ...and what got returned in its place is collision-free
    assert np.all(_free(out["joint_config"][fb], probs))
    assert out["n_coarse_fallbacks"] == int(fb.sum())
    print(f"\n  {int(fb.sum())} colliding LM results replaced by their collision-free coarse state")


def test_fallback_candidate_carries_its_coarse_metrics_not_the_lm_metrics(probs, seeds, goal):
    """A fallback must report the errors of the configuration actually being RETURNED (the coarse
    one), not the LM's -- otherwise the caller ranks on a pose that was thrown away.

    Tolerance is 1e-5 m, not machine epsilon: the coarse kernel computes in fp32 and reports its own
    fp32 error, while target_residuals recomputes in fp64, so they differ by ~1e-7 m. That is four
    orders of magnitude tighter than the thing being discriminated -- an LM error is ~1e-6 m against
    a coarse error of ~1e-1 m -- so the test still separates them decisively.
    """
    P, Q = _targets(goal, len(seeds))
    m = np.ones(len(seeds), dtype=np.uint32)
    out = _solve(seeds, goal, probs)
    fb = out["used_coarse_fallback"]
    fresh = hjcdik.target_residuals(np.asarray(out["joint_config"][fb], np.float64),
                                    P[fb], Q[fb], active_target_mask=m[fb])
    np.testing.assert_allclose(fresh["position_errors"][:, 0], out["position_errors"][fb, 0],
                               rtol=1e-3, atol=1e-5)

    # and they are NOT the LM's metrics for those seeds. (Note the direction: the colliding LM
    # results are also POOR poses here -- median 0.41 m against the coarse fallback's 0.14 m -- so
    # this is not "coarse is worse", it is simply "different". The metrics must track the config.)
    _c, lm = _unfiltered(seeds, goal, probs)
    assert not np.allclose(out["position_errors"][fb, 0], lm["position_errors"][fb, 0]), (
        "the fallback candidates are still reporting the LM's errors -- the metrics were not "
        "swapped along with the configuration")


# --- 5/6. ranking is on the FILTERED candidate set ------------------------------------------------
def test_selection_prefers_a_feasible_lm_candidate_over_a_worse_fallback(probs, seeds, goal):
    out = _solve(seeds, goal, probs)
    pick = int(np.argmin(out["cost"]))
    assert out["collision_free"][pick], "argmin(cost) selected an infeasible candidate"
    assert not out["used_coarse_fallback"][pick], (
        "the best candidate is a coarse fallback even though feasible LM candidates exist -- "
        "the refined results are being discarded")


def test_a_better_task_but_colliding_lm_candidate_cannot_win(probs, seeds, goal):
    """The defining property of a HARD filter: feasibility outranks task cost, always.

    Measured on bookshelf_small_panda: for 7 of the 82 colliding seeds, the COLLIDING LM result has a
    BETTER task cost than that seed's collision-free coarse fallback. Those are exactly the cases a
    finite collision penalty could be talked out of. solve() must return the worse-cost feasible
    coarse configuration for every one of them.

    (Being straight about scope: on THIS problem the batch-wide argmin was already landing on a
    feasible candidate -- the best feasible LM cost is 0.0, and no colliding candidate beats it. So
    the filter is not rescuing the top-1 pick here; it is rescuing the returned BATCH, which had 82
    colliding entries. The per-seed property below is the one that actually bites.)
    """
    _c, lm = _unfiltered(seeds, goal, probs)
    lm_free = _free(lm["joint_config"], probs)
    out = _solve(seeds, goal, probs)

    fb = out["used_coarse_fallback"]
    tempting = fb & (lm["cost"] < _c["cost"])       # colliding, yet cheaper than its own fallback
    assert tempting.sum() > 0, (
        "no colliding LM candidate was cheaper than its coarse fallback, so this test cannot "
        "distinguish a hard filter from one that merely prefers the better-scoring candidate")

    # every one of them returned the WORSE-cost feasible coarse config, not the cheaper colliding one
    np.testing.assert_array_equal(out["joint_config"][tempting],
                                  np.asarray(_c["joint_config"], out["joint_config"].dtype)[tempting])
    assert np.all(out["cost"][tempting] > lm["cost"][tempting]), (
        "a cheaper COLLIDING candidate was returned over its feasible fallback")
    assert np.all(_free(out["joint_config"][tempting], probs))
    print(f"\n  {int(tempting.sum())} colliding LM candidates were CHEAPER than their feasible "
          f"fallback; all were rejected anyway")


def test_selected_candidate_is_always_feasible(probs, seeds, goal):
    out = _solve(seeds, goal, probs)
    pick = int(np.argmin(out["cost"]))
    assert out["collision_free"][pick]
    assert np.all(_free(out["joint_config"][pick][None, :], probs))


# --- 7. every LM candidate colliding -> best collision-free coarse candidate ----------------------
def test_all_lm_colliding_falls_back_entirely_to_coarse(probs, seeds, goal):
    """Deterministically constructed: build a batch made ONLY of seeds whose LM is known to collide.
    Then every LM candidate collides and the answer must come entirely from the coarse stage."""
    first = _solve(seeds, goal, probs)
    bad_seeds = seeds[~first["lm_collision_free"]]
    assert len(bad_seeds) >= 20, "not enough colliding-LM seeds to build the all-colliding batch"

    out = _solve(bad_seeds, goal, probs)
    assert out["n_lm_collision_free"] == 0, (
        f"{out['n_lm_collision_free']} LM results turned out feasible in the all-colliding batch")
    assert out["n_coarse_fallbacks"] == len(bad_seeds), "not every seed fell back to coarse"
    assert out["n_infeasible"] == 0
    assert np.all(out["collision_free"])
    assert np.all(_free(out["joint_config"], probs))
    pick = int(np.argmin(out["cost"]))
    assert out["used_coarse_fallback"][pick], "the selected answer should be a coarse candidate"
    print(f"\n  all {len(bad_seeds)} LM results collided; all fell back to coarse; 0 colliding out")


# --- 8. collision-disabled solves run no filter ---------------------------------------------------
def test_collision_disabled_runs_no_filter(seeds, goal):
    out = _solve(seeds[:64], goal, probs=None)
    assert out["collision_enabled"] is False
    for k in ("collision_free", "lm_collision_free", "used_coarse_fallback",
              "n_lm_colliding", "n_coarse_fallbacks", "collision_filter_ms"):
        assert k not in out, f"open-world solve emitted '{k}' -- the collision filter ran"


# --- 10. diagnostics must not change the answer ---------------------------------------------------
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_diagnostics_do_not_change_the_filtered_solution(probs, seeds, goal, precision):
    off = _solve(seeds, goal, probs, precision=precision, diagnostics=False)
    on = _solve(seeds, goal, probs, precision=precision, diagnostics=True)
    np.testing.assert_array_equal(off["joint_config"], on["joint_config"])
    np.testing.assert_array_equal(off["cost"], on["cost"])
    np.testing.assert_array_equal(off["collision_free"], on["collision_free"])
    assert off["n_coarse_fallbacks"] == on["n_coarse_fallbacks"]


# --- 12. the Phase-0A stride invariant still holds on the filtered outputs -------------------------
def test_filtered_outputs_are_contiguous_with_real_strides(probs, seeds, goal):
    out = _solve(seeds[:128], goal, probs, diagnostics=True)
    for name, v in sorted(out.items()):
        if not isinstance(v, np.ndarray) or v.ndim == 0:
            continue
        assert 0 not in v.strides, f"{name}: zero stride"
        assert v.flags.c_contiguous, f"{name}: not C-contiguous"


def test_counters_are_self_consistent(probs, seeds, goal):
    out = _solve(seeds, goal, probs)
    B = len(seeds)
    assert out["n_lm_colliding"] + out["n_lm_collision_free"] == B
    assert out["n_coarse_fallbacks"] + out["n_infeasible"] == out["n_lm_colliding"]
    assert int(out["used_coarse_fallback"].sum()) == out["n_coarse_fallbacks"]
    assert int(out["collision_free"].sum()) == B - out["n_infeasible"]
    assert out["collision_filter_ms"] > 0.0
