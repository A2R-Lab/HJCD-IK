"""Warm selection invariants, including out-of-range seeds; never timing assertions."""
from pathlib import Path
import subprocess
import numpy as np
import pytest
import hjcdik


@pytest.mark.parametrize('precision', ['fp32', 'fp64'])
@pytest.mark.parametrize('mode', [None, 'hard', 'soft', 'both'])
@pytest.mark.parametrize('sign', [-1, 1])
def test_extreme_warm_seeds_keep_accurate_solutions(precision, mode, sign):
    solver = hjcdik.Solver()
    solver.set_world(spheres=[[10, 10, 10, .1]])
    targets = solver.sample_targets(4, seed=31)
    seeds = np.full((1, solver.info.dof), sign * 1e7)
    limits = np.array(solver.info.joint_limits)
    for q in (seeds, np.clip(seeds, limits[:, 0], limits[:, 1])):
        out = solver.solve(targets, batch_size=64, num_solutions=4, collision=mode,
                           precision=precision, q_init=q, stats=True)
        assert out.ok(1, .001).any(axis=1).all()
        assert np.isfinite(out.q[out.valid]).all()
        if mode in ('hard', 'both'):
            assert (out.stats.n_returned_coll_free == out.count).all()
    np.testing.assert_array_equal(seeds, np.full_like(seeds, sign * 1e7))


@pytest.mark.parametrize('precision', ['fp32', 'fp64'])
def test_multiple_per_target_extreme_seeds(precision):
    solver = hjcdik.Solver()
    targets = solver.sample_targets(2, seed=31)
    seeds = np.tile(np.array([[-1e7]*7, [1e7]*7]), (2, 1, 1))
    out = solver.solve(targets, batch_size=64, num_solutions=4, collision=None,
                       precision=precision, q_init=seeds)
    assert out.ok(1, .001).any(axis=1).all()


def test_native_selection_order(tmp_path):
    root = Path(__file__).resolve().parents[1]
    binary = tmp_path/'selection'
    subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                    '-I'+str(root/'csrc'), str(root/'tests/native/test_selection.cpp'),
                    '-o', str(binary)], check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
