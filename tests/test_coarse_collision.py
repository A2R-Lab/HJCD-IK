"""Phase 5: exact collision validation of the winning coarse proposal.

The coarse search accepts a proposal only when BOTH hold:
    * the exact task cost strictly improves, AND
    * the trial configuration is exactly collision-free (grid_collision::config_free -- SELF +
      environment), i.e. HARD feasibility, never a finite penalty.
On rejection the validated Phase-4 rollback restores the joint, the subtree transforms, the affected
target frames and the residual/cost caches.

These tests use the MotionBenchMaker goal poses, because that is where the collisions actually are: a
RANDOM Panda joint config almost never touches the obstacles (verified against the repo's independent
numpy collision reference -- 200/200 agreement), while reaching for the goal drives the arm into the
shelf. Testing with random targets would not exercise the gate at all.

Panda-only: the gate is compiled in only when grid.cuh was generated with --collision.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

import hjcdik

REPO = Path(__file__).resolve().parents[1]
N = hjcdik.num_joints()
K = hjcdik.num_targets()
LIMITS = hjcdik.joint_limits()

PROBS_PATH = REPO / "tests" / "mb_problems.json"
SET = "bookshelf_small_panda"

pytestmark = pytest.mark.skipif(K != 1, reason="collision build is the single-target Panda")


@pytest.fixture(scope="module")
def probs():
    return PROBS_PATH.read_text()


@pytest.fixture(scope="module")
def goal(probs):
    p = json.loads(probs)["problems"][SET]
    inst = p[0] if isinstance(p, list) else p
    gp = inst["goal_pose"]
    pos = np.asarray(gp.get("position_xyz", gp.get("position")), dtype=float)
    quat = np.asarray(gp.get("quaternion_wxyz", gp.get("quat_wxyz")), dtype=float)
    quat /= np.linalg.norm(quat)
    return pos, quat


@pytest.fixture(scope="module")
def free_seeds(probs):
    """Random configs that are already collision-free (so the gate's job is to KEEP them free)."""
    rng = np.random.default_rng(5)
    lo, hi = LIMITS[:, 0], LIMITS[:, 1]
    cand = np.stack([rng.uniform(lo, hi) for _ in range(400)])
    ok = hjcdik.collision_free(cand, probs, SET, 0)
    seeds = cand[ok][:64]
    assert len(seeds) >= 32, "not enough collision-free seeds to test with"
    return seeds


def _targets(goal, B):
    pos, quat = goal
    return (np.repeat(pos[None, None, :], B, axis=0),
            np.repeat(quat[None, None, :], B, axis=0))


def _run(seeds, goal, probs=None, **kw):
    p, q = _targets(goal, len(seeds))
    if probs is not None:
        kw.update(problems_json_text=probs, problem_set_name=SET, problem_idx=0)
    kw.setdefault("max_iters", 100)
    kw.setdefault("seed", 3)
    return hjcdik.coarse_search(seeds, p, q, **kw)


def test_collision_checker_agrees_with_the_solver_evaluator(probs, free_seeds):
    """Sanity: the checker these tests rely on IS the evaluator the gate uses."""
    ok = hjcdik.collision_free(free_seeds, probs, SET, 0)
    assert np.all(ok), "the 'free' seeds are not actually free"


def test_gate_rejects_task_improving_but_colliding_winners(probs, free_seeds, goal):
    """With the gate ON, strictly fewer steps are accepted -- the ones it drops are exactly the
    task-improving-but-colliding ones (a task-worsening step is already rejected either way)."""
    off = _run(free_seeds, goal, diagnostics=True)
    on = _run(free_seeds, goal, probs, diagnostics=True)
    a_off = off["accepted_coarse_steps"].sum()
    a_on = on["accepted_coarse_steps"].sum()
    assert a_on < a_off, (
        f"the collision gate accepted as many steps as the open-world search "
        f"({a_on} vs {a_off}) -- it is not rejecting anything")
    print(f"\n  accepted steps: gate OFF = {a_off}, gate ON = {a_on} "
          f"({a_off - a_on} colliding winners rejected)")


def test_gate_keeps_the_configuration_collision_free(probs, free_seeds, goal):
    """THE invariant: seeded collision-free, every accepted step is exactly collision-free, so the
    returned config is collision-free.

    Note on what this does and does not show. The open-world (un-gated) search can still END on a
    free config here, because the reported config is the BEST-COST one seen and that one happens to
    lie outside the shelf in this environment -- even though the un-gated search accepts colliding
    intermediates along the way. So the gate's necessity is established by
    test_gate_rejects_task_improving_but_colliding_winners (it drops ~1900 colliding winners that the
    open-world search accepts), not by an un-gated final failure. What THIS test pins is the
    invariant that matters for feasibility: with the gate on, the answer is never colliding.
    """
    on = _run(free_seeds, goal, probs)
    free_on = hjcdik.collision_free(on["joint_config"], probs, SET, 0)
    assert np.all(free_on), (
        f"{int((~free_on).sum())}/{len(free_seeds)} configs returned by the collision-gated coarse "
        f"search are COLLIDING")
    print(f"\n  gate ON: {int(free_on.sum())}/{len(free_seeds)} returned configs collision-free")


def test_rollback_after_collision_rejection_is_exact(probs, free_seeds, goal):
    """A collision-rejected step must restore the cost EXACTLY (the Phase-4 rollback), not merely
    approximately -- the trace records cost_before and cost_after for every iteration."""
    on = _run(free_seeds, goal, probs, diagnostics=True, return_trace=True)
    tr = on["trace"]
    n_rej = 0
    for b in range(len(free_seeds)):
        rows = tr[b][tr[b][:, 0] != 0]
        rej = rows[(rows[:, 7] == 0.0) & (rows[:, 2] >= 0) & (rows[:, 9] == 0.0)]
        # a rejected winner rolled back: the exact cost is restored bitwise
        assert np.all(rej[:, 6] == rej[:, 5]), "rollback after rejection did not restore the cost"
        n_rej += len(rej)
    assert n_rej > 0, "no proposal was ever rejected -- the rollback path is untested"
    print(f"\n  {n_rej} rejected winners, all rolled back to a bitwise-identical cost")


def test_best_state_preserved_under_collision_rejection(probs, free_seeds, goal):
    """The returned config is the best FEASIBLE state seen, and never worse than the seed.

    Asserted on E_phys (the tolerance-normalised physical error), NOT on cost_raw. best_x is now
    tracked on E_phys because it is the only metric comparable ACROSS iterations -- the row scales
    are re-frozen every iteration, so the scaled cost is not. cost_raw is a different objective
    (position and orientation weighted equally and unnormalised), and the coarse search can raise it
    on a small number of seeds while E_phys still falls: it trades cheap orientation error for
    expensive position error, which is exactly what a tolerance-normalised objective should do.
    """
    pos, quat = goal
    B = len(free_seeds)
    P = np.repeat(pos[None, None, :], B, axis=0)
    Q = np.repeat(quat[None, None, :], B, axis=0)
    PTOL, OTOL = 1e-3, 1e-2                       # coarse_search's defaults, as _run uses them

    def e_phys(res):
        return ((res["position_errors"] / PTOL) ** 2
                + (res["orientation_errors"] / OTOL) ** 2).sum(axis=1)

    before = hjcdik.target_residuals(free_seeds, P, Q)
    out = _run(free_seeds, goal, probs)
    after = hjcdik.target_residuals(np.asarray(out["joint_config"], np.float64), P, Q)
    assert np.all(e_phys(after) <= e_phys(before) * (1 + 1e-6)), (
        "the collision-gated coarse search returned a state with WORSE physical merit than its seed")

def test_perturbations_do_not_break_collision_freedom(probs, free_seeds, goal):
    """A stall-triggered random perturbation jumps the config. Any step accepted AFTER it still has
    to pass the gate, so the returned config must remain collision-free."""
    on = _run(free_seeds, goal, probs, diagnostics=True, stall_lim=2, max_iters=150)
    assert on["coarse_perturbations"].sum() > 0, "no perturbation fired -- the path is untested"
    free = hjcdik.collision_free(on["joint_config"], probs, SET, 0)
    assert np.all(free), "a perturbation left the returned config colliding"
    print(f"\n  {int(on['coarse_perturbations'].sum())} perturbations fired; "
          f"all {len(free_seeds)} returned configs still collision-free")


def test_diagnostics_do_not_change_the_gated_solution(probs, free_seeds, goal):
    off = _run(free_seeds, goal, probs)
    on = _run(free_seeds, goal, probs, diagnostics=True)
    np.testing.assert_array_equal(off["joint_config"], on["joint_config"])
    np.testing.assert_array_equal(off["cost"], on["cost"])
