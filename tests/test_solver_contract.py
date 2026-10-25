"""The Solver contract: collision table, result shapes, batch and resource rules.

Correctness only; shared-GPU safe. Every row of the collision table in the Python API reference and
every bullet of the batch/resource contract has a test here.
"""
import threading
import time

import numpy as np
import pytest

hjcdik = pytest.importorskip("hjcdik")

BLOCKING_SPHERE = [[0.0, 0.0, 0.0, 100.0]]      # swallows the whole workspace
FAR_SPHERE = [[10.0, 10.0, 10.0, 0.1]]          # never touches the arm


@pytest.fixture
def solver():
    return hjcdik.Solver()


@pytest.fixture(scope="module")
def targets():
    return hjcdik.Solver().sample_targets(3, seed=5)


# ------------------------------------------------------------------------------ collision table
def test_omitted_collision_defaults_to_hard_on_a_collision_build(solver, targets):
    assert solver.info.collision_enabled
    solver.set_world(spheres=BLOCKING_SPHERE)
    assert solver.solve(targets[0], batch_size=64).count.tolist() == [0]           # omitted = hard
    assert solver.solve(targets[0], batch_size=64, collision=None).count.tolist() == [1]  # explicit None


def test_hard_with_no_world_checks_self_collision_only(solver, targets):
    hard = solver.solve(targets, batch_size=256, collision="hard")
    assert (hard.count >= 0).all() and hard.q.shape == (3, 1, solver.info.dof)
    solver.clear_world()
    again = solver.solve(targets, batch_size=256, collision="hard")
    assert again.count.shape == (3,)


def test_soft_without_a_world_raises_before_any_launch(solver, targets, monkeypatch):
    calls = []
    monkeypatch.setattr(hjcdik._solver._native, "solve_batch", lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match="soft"):
        solver.solve(targets, collision="soft")
    solver.set_world()   # explicitly empty world is the same state
    with pytest.raises(ValueError, match="soft"):
        solver.solve(targets, collision="soft")
    assert calls == []


def test_soft_ranks_without_filtering_and_both_filters(solver, targets):
    solver.set_world(spheres=BLOCKING_SPHERE)
    soft = solver.solve(targets[0], batch_size=256, num_solutions=4, collision="soft")[0]
    both = solver.solve(targets[0], batch_size=256, num_solutions=4, collision="both")[0]
    assert soft.count == 4 and both.count == 0


def test_both_with_empty_world_behaves_as_hard(solver, targets):
    solver.set_world(spheres=FAR_SPHERE)
    solver.clear_world()
    both = solver.solve(targets, batch_size=256, num_solutions=2, collision="both", stats=True)
    hard = solver.solve(targets, batch_size=256, num_solutions=2, collision="hard", stats=True)
    assert both.count.tolist() == hard.count.tolist()
    assert (both.stats.env_cost_max_mm == 0).all()
    assert (hard.stats.env_cost_max_mm == -1).all()


def test_world_requires_a_collision_build_and_rejects_bad_geometry(solver):
    with pytest.raises(ValueError, match="shape"):
        solver.set_world(spheres=[[0, 0, 0]])
    with pytest.raises(ValueError, match="positive"):
        solver.set_world(spheres=[[0, 0, 0, -1]])
    with pytest.raises(ValueError, match="finite"):
        solver.set_world(boxes=[[0, 0, 0, 1, 0, 0, 0, 1, float("nan"), 1]])
    with pytest.raises(ValueError, match="quaternion"):
        solver.set_world(boxes=[[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]])
    with pytest.raises(ValueError, match="positive"):
        solver.set_world(cylinders=[[0, 0, 0, 1, 0, 0, 0, 0.1, 0]])
    with pytest.raises(ValueError, match="positive"):
        solver.set_world(capsules=[[0, 0, 0, 0, 0, 1, 0]])


def test_cylinders_become_capsules_of_the_same_axis(solver, targets):
    cylinder = [[0, 0, 0, 1, 0, 0, 0, 10.0, 10.0]]
    capsule = [[0, 0, -5.0, 0, 0, 5.0, 10.0]]
    solver.set_world(cylinders=cylinder)
    assert solver.world_counts == {"spheres": 0, "capsules": 1, "boxes": 0}
    blocked_c = solver.solve(targets[0], batch_size=128).count.tolist()
    solver.set_world(capsules=capsule)
    blocked_k = solver.solve(targets[0], batch_size=128).count.tolist()
    assert blocked_c == blocked_k == [0]
    arrays = hjcdik.world_arrays(cylinders=[[1, 2, 3, 0.70710678, 0.70710678, 0, 0, 0.5, 2.0]])
    # quaternion = 90 degrees about x: the local +z axis maps to -y
    np.testing.assert_allclose(arrays["capsules"][0], [1, 3, 3, 1, 1, 3, 0.5], atol=1e-6)


# ------------------------------------------------------------------------------ results
def test_single_pose_keeps_the_leading_dimension_and_pads(solver, targets):
    solver.clear_world()
    r = solver.solve(targets[0], batch_size=256, num_solutions=4, collision=None)
    assert len(r) == 1 and r.q.shape == (1, 4, solver.info.dof) and r.pose.shape == (1, 4, 7)
    n = int(r.count[0])
    assert 0 < n <= 4
    assert r.valid.tolist() == [[s < n for s in range(4)]]
    assert np.isnan(r.q[0, n:]).all() and np.isfinite(r.q[0, :n]).all()
    view = r[0]
    assert view.q.shape == (n, solver.info.dof) and view.count == n and view.elapsed_ms == r.elapsed_ms[0]
    assert view.ok(1e9, 1e9).tolist() == [True] * n
    assert r.ok(1e9, 1e9).sum() == n
    with pytest.raises(IndexError):
        r[1]


def test_blocked_targets_pad_with_nan_and_zero_counts(solver, targets):
    solver.set_world(spheres=BLOCKING_SPHERE)
    r = solver.solve(targets, batch_size=64, num_solutions=3)
    assert r.count.tolist() == [0, 0, 0] and not r.valid.any()
    assert np.isnan(r.q).all() and np.isnan(r.pos_err_mm).all()
    assert not r.ok(1e9, 1e9).any()
    assert r[2].q.shape == (0, solver.info.dof)


def test_empty_batch_returns_an_empty_result(solver):
    r = solver.solve(np.zeros((0, 7)), num_solutions=2, collision=None)
    assert len(r) == 0 and r.q.shape == (0, 2, solver.info.dof) and r.count.shape == (0,)


def test_results_are_read_only_and_outlive_the_solver(targets):
    local = hjcdik.Solver()
    r = local.solve(targets, batch_size=128, collision=None)
    for array in (r.q, r.pose, r.pos_err_mm, r.ori_err_rad, r.count, r.valid, r.elapsed_ms):
        assert not array.flags.writeable
    with pytest.raises(ValueError):
        r.q[0, 0, 0] = 1.0
    snapshot = r.q.copy()
    del local
    hjcdik.Solver().solve(targets, batch_size=512, collision=None)
    np.testing.assert_array_equal(r.q, snapshot)


def test_stats_are_per_target_and_do_not_change_outputs(solver, targets):
    solver.clear_world()
    kwargs = dict(batch_size=1, num_solutions=4, precision="fp32", collision=None)
    plain = solver.solve(targets, **kwargs)
    logged = solver.solve(targets, stats=True, **kwargs)
    assert plain.stats is None and len(logged.stats) == 3
    assert plain.count.tolist() == logged.count.tolist()
    np.testing.assert_allclose(plain.q, logged.q, rtol=1e-6, atol=1e-8, equal_nan=True)
    assert logged.stats.batch_size.tolist() == [1, 1, 1]
    assert logged.stats.n_returned.tolist() == logged.count.tolist()
    assert (logged.stats.n_coll_in_coarse == -1).all()     # open-world: unmeasured, never zero
    row = logged[1].stats
    assert set(row) == set(hjcdik.Stats.field_names()) and row["n_returned"] == int(logged.count[1])


# ------------------------------------------------------------------------------ batch and resources
def test_whole_batch_is_validated_before_any_gpu_work(solver, targets, monkeypatch):
    calls = []
    monkeypatch.setattr(hjcdik._solver._native, "solve_batch", lambda *a, **k: calls.append(a))
    bad = np.array(targets, copy=True)
    bad[-1, 3:] = 0
    with pytest.raises(ValueError, match="quaternion"):
        solver.solve(bad, collision=None)
    bad[-1] = targets[-1]
    bad[1, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        solver.solve(bad, collision=None)
    assert calls == []


def test_cuda_failure_raises_solve_error_with_position(solver, targets, monkeypatch):
    def failing(poses, *args):
        raise RuntimeError("target 1: injected CUDA failure")   # the binding names the failing target

    monkeypatch.setattr(hjcdik._solver._native, "solve_batch", failing)
    with pytest.raises(hjcdik.SolveError, match="target 1") as info:
        solver.solve(targets, collision=None)
    assert info.value.target_index == 1 and info.value.completed == 1
    assert isinstance(info.value, RuntimeError)


def test_failed_set_world_keeps_the_previous_world(solver, targets):
    solver.set_world(spheres=BLOCKING_SPHERE)
    with pytest.raises(ValueError):
        solver.set_world(spheres=[[0, 0, 0, -5]])
    assert solver.world_counts["spheres"] == 1
    assert solver.solve(targets[0], batch_size=64).count.tolist() == [0]


def test_two_solvers_hold_independent_worlds(targets):
    blocked, clear = hjcdik.Solver(), hjcdik.Solver()
    blocked.set_world(spheres=BLOCKING_SPHERE)
    clear.set_world(spheres=FAR_SPHERE)
    for _ in range(2):
        assert blocked.solve(targets[0], batch_size=128).count.tolist() == [0]
        assert clear.solve(targets[0], batch_size=128).count.tolist() == [1]


def test_one_world_for_the_whole_multi_target_solve(solver):
    many = solver.sample_targets(24, seed=9)
    solver.set_world(spheres=BLOCKING_SPHERE)
    results = {}

    def solve():
        results["blocked"] = solver.solve(many, batch_size=256)

    worker = threading.Thread(target=solve)
    worker.start()
    time.sleep(0.05)
    solver.set_world(spheres=FAR_SPHERE)     # waits for the solve in flight
    worker.join()
    assert results["blocked"].count.tolist() == [0] * 24
    assert solver.solve(many[:2], batch_size=256).count.tolist() == [1, 1]


def test_solver_restores_the_callers_current_device(solver, targets):
    native = hjcdik._solver._native
    before = native.current_device()
    assert solver.info.device == before
    solver.solve(targets[0], batch_size=64, collision=None)
    solver.set_world(spheres=FAR_SPHERE)
    solver.sample_targets(2)
    assert native.current_device() == before


def test_world_on_another_device_is_rejected(solver, targets):
    other = hjcdik.Solver(device=solver.info.device + 1)
    with pytest.raises(RuntimeError):
        other.set_world(spheres=FAR_SPHERE)   # no such device, or a different one: never silently used


def test_precision_modes_solve_and_auto_tracks_num_solutions(solver, targets):
    for precision in ("fp32", "fp64", "auto"):
        r = solver.solve(targets, batch_size=512, num_solutions=1, precision=precision, collision=None)
        assert r.count.tolist() == [1, 1, 1] and (r.pos_err_mm[:, 0] < 0.01).all()


def test_sample_targets_shape_seed_and_determinism(solver):
    a = solver.sample_targets(5, seed=3)
    b = solver.sample_targets(5, seed=3)
    c = solver.sample_targets(5, seed=4)
    assert a.shape == (5, 7) and a.dtype == np.float64
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    np.testing.assert_allclose(np.linalg.norm(a[:, 3:], axis=1), 1.0, atol=1e-6)
