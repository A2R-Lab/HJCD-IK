"""Stage 3 solver knobs: clearance margin, self-collision toggle, warm starts and seeds.

Correctness only; shared-GPU safe. Margins widen environment pairs only; self pairs stay at the
contact-free rule. Warm starts hand accurate configurations straight to the refinement and rank the
converged candidates by proximity to the supplied configuration.
"""
import numpy as np
import pytest

hjcdik = pytest.importorskip("hjcdik")

FAR_SPHERE = [[10.0, 10.0, 10.0, 0.1]]
NEAR_BOX = [[0.4, 0.0, 0.4, 1, 0, 0, 0, 0.3, 0.3, 0.3]]


@pytest.fixture
def solver():
    return hjcdik.Solver()


@pytest.fixture(scope="module")
def targets():
    return hjcdik.Solver().sample_targets(6, seed=11)


# ---------------------------------------------------------------------------------- margin
def test_margin_zero_is_the_contact_free_rule_and_a_huge_margin_blocks_everything(solver, targets):
    solver.set_world(spheres=FAR_SPHERE)
    assert solver.solve(targets, batch_size=256).count.tolist() == [1] * 6
    assert solver.solve(targets, batch_size=256, collision_margin_m=0.0).count.tolist() == [1] * 6
    r = solver.solve(targets, batch_size=256, collision_margin_m=20.0, stats=True)
    assert r.count.tolist() == [0] * 6 and (r.stats.n_coll_in_coarse == 256).all()


def test_margin_is_monotone_on_environment_pairs_and_leaves_self_pairs_alone(solver, targets):
    solver.set_world(boxes=NEAR_BOX)
    rejected = []
    for margin in (0.0, 0.02, 0.1):
        r = solver.solve(targets, batch_size=512, num_solutions=4, collision_margin_m=margin, stats=True)
        rejected.append(r.stats.n_coll_in_coarse.copy())
        assert (r.count > 0).all(), f"a {margin} m margin on a 30 cm box must leave solutions"
    assert (rejected[1] >= rejected[0]).all() and (rejected[2] >= rejected[1]).all()
    assert (rejected[2] > rejected[0]).any()
    # The margin never widens the baked self pairs: an empty world with a margin solves like contact-free.
    solver.clear_world()
    with_margin = solver.solve(targets, batch_size=512, collision_margin_m=0.05, stats=True)
    without = solver.solve(targets, batch_size=512, stats=True)
    assert with_margin.count.tolist() == without.count.tolist() == [1] * 6
    assert with_margin.stats.n_coll_in_coarse.sum() < 512 * 6


def test_margin_and_self_toggle_need_hard_or_both(solver, targets):
    for kwargs in (dict(collision=None, collision_margin_m=0.01), dict(collision="soft", collision_margin_m=0.01),
                   dict(collision=None, self_collision=False), dict(collision="soft", self_collision=False)):
        with pytest.raises(ValueError, match="hard"):
            solver.solve(targets[0], **kwargs)
    with pytest.raises(ValueError, match="non-negative"):
        solver.solve(targets[0], collision_margin_m=-0.01)
    with pytest.raises(ValueError, match="non-negative"):
        solver.solve(targets[0], collision_margin_m=float("nan"))


# ------------------------------------------------------------------------------ self toggle
def test_self_collision_off_skips_the_self_pairs(solver, targets):
    solver.clear_world()
    on = solver.solve(targets, batch_size=512, num_solutions=4, stats=True)
    off = solver.solve(targets, batch_size=512, num_solutions=4, self_collision=False, stats=True)
    assert (on.stats.n_coll_in_coarse > 0).any()
    assert (off.stats.n_coll_in_coarse == 0).all() and (off.stats.n_coll_in_refined == 0).all()
    assert off.count.tolist() == [4] * 6
    solver.set_world(spheres=[[0, 0, 0, 100]])
    assert solver.solve(targets[0], batch_size=64, self_collision=False).count.tolist() == [0]   # env still checked


# ------------------------------------------------------------------------------- warm starts
def test_warm_start_returns_its_own_solution_first(solver, targets):
    solver.clear_world()
    base = solver.solve(targets[:3], batch_size=512, collision=None)
    q_star = base.q[:, 0, :]                                        # (3, N)
    per_target = solver.solve(targets[:3], batch_size=1, num_solutions=4, collision=None,
                              q_init=q_star[:, None, :])           # (T, K=1, N)
    for i in range(3):
        row = per_target[i]
        assert row.count >= 1
        assert np.abs(row.q[0] - q_star[i]).max() < 1e-3, "the warm start's own solution comes first"
        assert row.pos_err_mm[0] < 0.01
        distances = np.abs(row.q - q_star[i]).max(axis=1)
        converged = row.ok(1.0, 1e-3)
        assert np.all(np.diff(distances[converged]) >= -1e-9), "converged rows are ordered by distance"
    shared = solver.solve(targets[0], batch_size=64, q_init=q_star[:1])      # (K, N) shared, hard default
    assert np.abs(shared.q[0, 0] - q_star[0]).max() < 1e-3
    for precision in ("fp32", "fp64"):   # fp32 refinement must not wander from an exact warm start
        r = solver.solve(targets[:3], batch_size=1, num_solutions=4, collision=None, precision=precision,
                         q_init=q_star[:, None, :])
        assert np.abs(r.q[:, 0, :] - q_star).max() < 1e-3, precision


def test_cold_start_with_one_candidate_lands_elsewhere(solver, targets):
    solver.clear_world()
    q_star = solver.solve(targets[0], batch_size=512, collision=None).q[0, 0]
    cold = solver.solve(targets[0], batch_size=1, collision=None)
    warm = solver.solve(targets[0], batch_size=1, collision=None, q_init=q_star[None, :])
    assert np.abs(warm.q[0, 0] - q_star).max() < 1e-3
    assert np.abs(cold.q[0, 0] - q_star).max() > 1e-2, "a single random candidate should not find q* by chance"


def test_warm_start_shapes_and_limits(solver, targets):
    N = solver.info.dof
    with pytest.raises(ValueError, match="batch_size"):
        solver.solve(targets[0], batch_size=2, collision=None, q_init=np.zeros((3, N)))
    with pytest.raises(ValueError, match="shape"):
        solver.solve(targets[0], collision=None, q_init=np.zeros((2, N, 3)))
    with pytest.raises(ValueError, match="shape"):
        solver.solve(targets[:2], collision=None, q_init=np.zeros((3, 1, N)))   # T mismatch
    with pytest.raises(ValueError, match="finite"):
        solver.solve(targets[0], collision=None, q_init=np.full((1, N), np.nan))
    huge = solver.solve(targets[0], batch_size=8, collision=None, q_init=np.full((2, N), 100.0))
    assert huge.count.tolist() == [1]                                 # clamped to the joint limits, still solves


# ------------------------------------------------------------------------------------- seeds
def test_seed_shifts_the_initialisation_deterministically(solver, targets):
    solver.clear_world()
    a = solver.solve(targets[0], batch_size=1, collision=None, seed=1)
    b = solver.solve(targets[0], batch_size=1, collision=None, seed=1)
    c = solver.solve(targets[0], batch_size=1, collision=None, seed=2)
    default = solver.solve(targets[0], batch_size=1, collision=None)
    legacy = solver.solve(targets[0], batch_size=1, collision=None, seed=1337)
    np.testing.assert_array_equal(a.q, b.q)
    np.testing.assert_array_equal(default.q, legacy.q)
    assert np.nanmax(np.abs(a.q - c.q)) > 1e-3
    for seed in (-1, 2**32, 1.5):
        with pytest.raises(ValueError, match="seed"):
            solver.solve(targets[0], collision=None, seed=seed)
