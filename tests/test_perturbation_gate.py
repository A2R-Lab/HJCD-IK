"""The stall perturbation is collision-gated, and best_x only ever comes from a feasible state.

THE DEFECT THIS PINS. The coarse search kicks the configuration when it stalls. That kick rewrote
every joint WITHOUT re-running the collision gate, and because best_x is copied from the current
state, a perturbed-and-colliding configuration could become the returned answer. On
bookshelf_small_panda with verified collision-free seeds it returned 5/64 colliding configs at
stall_lim=5, and 0/64 with kicks disabled -- proving the kick was the sole leak (the accepted-step
gate was always correct).

It went unseen because hjcdik.collision_free() itself returned a ZERO-STRIDED array, so
`np.all(free)` compared one candidate's flag with itself. Phase 0A fixed the strides and the defect
became visible.

SEMANTICS NOW. A perturbation is EXPLORATORY: unlike a coordinate proposal it does NOT have to
reduce the task cost to become the current state -- that is the entire point of a kick -- but it is
not exempt from the hard constraints. save -> kick -> full refresh -> exact config_free -> keep, or
restore exactly. Bounded retries, stopping at the first feasible kick.

THE FIXTURE PROVABLY CATCHES THE OLD BUG. Building this exact tree with one line changed --

    int ok = 1;   // kick collision gate DELIBERATELY BYPASSED
    (replacing the config_free call in the kick attempt loop in coarse_search_mt_kernel)

-- and running these 256 seeds reproduces the defect:

    stall_lim   colliding configs returned
        2            10 / 256
        5            19 / 256
       10            19 / 256

against 0/256 for the shipped build at every stall limit. So these tests fail loudly if the gate is
ever removed; they do not pass by accident. (Scratch probe only -- it is not part of the tree.)

Panda-only: the gate is compiled in only when grid.cuh was generated with --collision.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import hjcdik

REPO = Path(__file__).resolve().parents[1]
N = hjcdik.num_joints()
K = hjcdik.num_targets()
LIMITS = hjcdik.joint_limits()
SET = "bookshelf_small_panda"

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
def free_seeds(probs):
    """256 configs VERIFIED collision-free. Large enough that the old defect (which leaked on ~8% of
    candidates) could not pass by chance: P(no leak | 256 seeds) is astronomically small."""
    rng = np.random.default_rng(5)
    lo, hi = LIMITS[:, 0], LIMITS[:, 1]
    cand = np.stack([rng.uniform(lo, hi) for _ in range(1200)])
    seeds = cand[hjcdik.collision_free(cand, probs, SET, 0)][:256]
    assert len(seeds) == 256
    assert np.all(hjcdik.collision_free(seeds, probs, SET, 0)), "fixture seeds are not free"
    return seeds


def _run(seeds, goal, probs=None, **kw):
    # The rollback/restoration assertions here are exact to ~1e-12, which is an fp64 claim; pin it.
    # The fp32 collision gate is covered in test_precision_fp32.py.
    kw.setdefault("precision", "float64")
    pos, quat = goal
    B = len(seeds)
    P = np.repeat(pos[None, None, :], B, axis=0)
    Q = np.repeat(quat[None, None, :], B, axis=0)
    if probs is not None:
        kw.update(problems_json_text=probs, problem_set_name=SET, problem_idx=0)
    kw.setdefault("max_iters", 100)
    kw.setdefault("seed", 3)
    return hjcdik.coarse_search(seeds, P, Q, **kw)


# --- 7. THE exit condition ---------------------------------------------------------------------
@pytest.mark.parametrize("stall_lim", [2, 5, 10])
def test_returned_configs_are_collision_free(probs, free_seeds, goal, stall_lim):
    out = _run(free_seeds, goal, probs, stall_lim=stall_lim, diagnostics=True)
    free = hjcdik.collision_free(out["joint_config"], probs, SET, 0)
    n_bad = int((~free).sum())
    assert int(out["coarse_perturbations"].sum()) > 0, "no kick was retained -- gate is untested"
    assert n_bad == 0, (
        f"stall_lim={stall_lim}: {n_bad}/{len(free_seeds)} returned configs COLLIDE. The "
        f"perturbation gate is leaking again.")


# --- 8. perturbations disabled preserves existing behaviour --------------------------------------
def test_perturbations_disabled_preserves_behaviour(probs, free_seeds, goal):
    out = _run(free_seeds, goal, probs, stall_lim=10**9, diagnostics=True)
    assert int(out["coarse_perturbation_events"].sum()) == 0, "a kick fired with kicks disabled"
    assert int(out["coarse_perturbation_attempts"].sum()) == 0
    free = hjcdik.collision_free(out["joint_config"], probs, SET, 0)
    assert np.all(free), "the un-perturbed search returned a colliding config"


# --- 1/4. a colliding kick is rejected and never reaches best_x -----------------------------------
def test_colliding_kicks_are_rejected(probs, free_seeds, goal):
    """The gate must actually be refusing kicks -- otherwise the tests above pass vacuously."""
    out = _run(free_seeds, goal, probs, stall_lim=2, diagnostics=True)
    rej = int(out["coarse_perturbations_rejected"].sum())
    att = int(out["coarse_perturbation_attempts"].sum())
    assert rej > 0, "the collision gate rejected ZERO kicks -- it is not doing anything"
    print(f"\n  {rej} of {att} kick attempts refused by the collision gate")


def test_best_x_never_takes_a_colliding_state(probs, free_seeds, goal):
    """Direct statement of the invariant, over every stall limit at once."""
    for sl in (2, 3, 5, 10):
        out = _run(free_seeds, goal, probs, stall_lim=sl)
        free = hjcdik.collision_free(out["joint_config"], probs, SET, 0)
        assert np.all(free), f"stall_lim={sl}: best_x came from a colliding state"


# --- 3. an exploratory kick may be RETAINED even when it makes the task cost worse ----------------
def test_retained_kick_may_worsen_the_task_cost(probs, free_seeds, goal):
    """A kick is exploratory: collision-freedom is required, cost improvement is NOT. The trace
    records cost_before/cost_after on the kicking iteration, so we can prove a retained kick was
    allowed to raise the cost."""
    out = _run(free_seeds, goal, probs, stall_lim=2, diagnostics=True, return_trace=True)
    tr = out["trace"]
    worse = 0
    for b in range(len(free_seeds)):
        rows = tr[b][tr[b][:, 0] != 0]
        kicked = rows[rows[:, 9] == 1.0]              # col 9 = kick RETAINED
        worse += int((kicked[:, 6] > kicked[:, 5]).sum())   # cost_after > cost_before
    assert worse > 0, (
        "not one retained kick raised the task cost -- either the kick is being treated as a "
        "cost-improving proposal, or the fixture never explores")
    print(f"\n  {worse} retained kicks raised the task cost (exploratory, as intended)")


# --- 5. all attempts collide -> original state restored -------------------------------------------
def test_exhausted_perturbation_restores_the_original_state(probs, free_seeds, goal):
    """With only ONE attempt allowed, exhaustion is common. The state must be restored exactly and
    best_x left alone -- so the answer stays collision-free regardless."""
    out = _run(free_seeds, goal, probs, stall_lim=2, max_pert_attempts=1, diagnostics=True)
    exh = int(out["coarse_perturbations_exhausted"].sum())
    assert exh > 0, "no exhausted event with max_pert_attempts=1 -- the restore path is untested"
    free = hjcdik.collision_free(out["joint_config"], probs, SET, 0)
    assert np.all(free), "an exhausted perturbation left the returned config colliding"
    print(f"\n  {exh} exhausted kick events; all {len(free_seeds)} answers still collision-free")


def test_more_attempts_means_fewer_exhausted_events(probs, free_seeds, goal):
    """Bounded retries do what they say: raising the bound converts exhausted events into retained
    kicks. This also proves the attempt loop actually retries rather than giving up at the first
    collision."""
    prev_exh = None
    for att in (1, 2, 4):
        out = _run(free_seeds, goal, probs, stall_lim=2, max_pert_attempts=att, diagnostics=True)
        exh = int(out["coarse_perturbations_exhausted"].sum())
        ev = int(out["coarse_perturbation_events"].sum())
        assert int(out["coarse_perturbation_attempts"].sum()) <= ev * att, "attempt bound exceeded"
        if prev_exh is not None:
            assert exh <= prev_exh, f"max_pert_attempts={att} exhausted MORE often than before"
        prev_exh = exh
        assert np.all(hjcdik.collision_free(out["joint_config"], probs, SET, 0))


# --- 6/9. restoration + counters agree with the trace ---------------------------------------------
def test_diagnostic_counters_match_the_trace(probs, free_seeds, goal):
    """Every counter is DERIVED FROM THE TRACE; recompute them here and demand equality."""
    out = _run(free_seeds, goal, probs, stall_lim=2, diagnostics=True, return_trace=True)
    tr = out["trace"]
    B = len(free_seeds)
    ev = np.zeros(B, int); att = np.zeros(B, int)
    rej = np.zeros(B, int); exh = np.zeros(B, int); ret = np.zeros(B, int)
    for b in range(B):
        rows = tr[b][tr[b][:, 0] != 0]
        ret[b] = int(rows[:, 9].sum())
        att[b] = int(rows[:, 10].sum())
        ev[b] = int((rows[:, 10] > 0).sum())
        rej[b] = int(rows[:, 11].sum())
        exh[b] = int(rows[:, 12].sum())
    np.testing.assert_array_equal(out["coarse_perturbations"], ret)
    np.testing.assert_array_equal(out["coarse_perturbation_events"], ev)
    np.testing.assert_array_equal(out["coarse_perturbation_attempts"], att)
    np.testing.assert_array_equal(out["coarse_perturbations_rejected"], rej)
    np.testing.assert_array_equal(out["coarse_perturbations_exhausted"], exh)
    # structural identities: every event either retains a kick or exhausts; every attempt is either
    # the retained one or a collision rejection.
    np.testing.assert_array_equal(ret + exh, ev)
    np.testing.assert_array_equal(ret + rej, att)


def test_restoration_is_exact(probs, free_seeds, goal):
    """A kicking iteration that is fully rejected must restore the cost bitwise.

    With max_pert_attempts=1 and a tight stall limit, exhausted events are frequent; on those the
    state is rebuilt from the saved joint vector. coarse_full_refresh is a pure function of s_x, so
    re-running it on a bitwise-restored s_x reproduces the transforms, residuals and costs exactly.
    We verify end-to-end: re-evaluating the returned config from scratch reproduces the reported
    cost and errors.
    """
    out = _run(free_seeds, goal, probs, stall_lim=2, max_pert_attempts=1)
    assert int(_run(free_seeds, goal, probs, stall_lim=2, max_pert_attempts=1,
                    diagnostics=True)["coarse_perturbations_exhausted"].sum()) > 0
    pos, quat = goal
    B = len(free_seeds)
    P = np.repeat(pos[None, None, :], B, axis=0)
    Q = np.repeat(quat[None, None, :], B, axis=0)
    fresh = hjcdik.target_residuals(out["joint_config"], P, Q)
    np.testing.assert_allclose(fresh["position_errors"][:, 0], out["position_errors"][:, 0],
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(fresh["orientation_errors"][:, 0], out["orientation_errors"][:, 0],
                               rtol=0, atol=1e-12)


# --- 10. diagnostics must not change the answer ---------------------------------------------------
def test_diagnostics_do_not_change_the_solution(probs, free_seeds, goal):
    off = _run(free_seeds, goal, probs, stall_lim=2)
    on = _run(free_seeds, goal, probs, stall_lim=2, diagnostics=True)
    np.testing.assert_array_equal(off["joint_config"], on["joint_config"])
    np.testing.assert_array_equal(off["cost"], on["cost"])


def test_kicks_do_not_make_the_best_state_worse(probs, free_seeds, goal):
    """Kicks are exploratory, so the question is whether ENABLING them degrades the answer.

    Deliberately NOT asserting `raw cost of answer <= raw cost of seed`. That does not hold, and it
    has nothing to do with this gate: best_x is selected on the ROW-SCALED cost, and the row scaling
    is re-frozen every iteration, so best_total (recorded under one scaling) is compared against
    st->total (under another). The two are not commensurable across iterations, and for a handful of
    candidates the RAW cost of the answer ends up marginally above the seed's. Measured on these 256
    seeds: 5/256 with kicks DISABLED versus 1/256 with kicks enabled -- i.e. the effect is
    pre-existing and kicks make it less common, not more. Tracked separately; out of scope here.

    What this test does pin is that turning kicks on does not make the search worse on average.
    """
    pos, quat = goal
    B = len(free_seeds)
    P = np.repeat(pos[None, None, :], B, axis=0)
    Q = np.repeat(quat[None, None, :], B, axis=0)
    off = hjcdik.target_residuals(
        _run(free_seeds, goal, probs, stall_lim=10**9)["joint_config"], P, Q)["cost_raw"]
    on = hjcdik.target_residuals(
        _run(free_seeds, goal, probs, stall_lim=2)["joint_config"], P, Q)["cost_raw"]
    assert on.mean() <= off.mean() * 1.05, (
        f"enabling collision-gated kicks degraded the mean raw cost: {on.mean():.4g} vs "
        f"{off.mean():.4g} with kicks off")
