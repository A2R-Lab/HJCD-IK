"""Public Python API validation and normalization contracts for `hjcdik.Solver`."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import pytest

np = pytest.importorskip("numpy")
hjcdik = pytest.importorskip("hjcdik")

POSE = [0.3, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]


@pytest.fixture(scope="module")
def solver():
    return hjcdik.Solver()


@pytest.mark.parametrize("batch_size,num_solutions", [(0, 1), (-1, 1), (10, 0), (10, -2)])
def test_positive_sizes_required(solver, batch_size, num_solutions):
    with pytest.raises(ValueError):
        solver.solve(POSE, batch_size=batch_size, num_solutions=num_solutions)


@pytest.mark.parametrize("value", ["fp16", "double", -1, 1])
def test_precision_is_validated(solver, value):
    with pytest.raises(ValueError, match="precision"):
        solver.solve(POSE, precision=value)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_target_values_must_be_finite(solver, value):
    for index in range(7):
        bad = list(POSE)
        bad[index] = value
        with pytest.raises(ValueError, match="finite"):
            solver.solve(bad)


def test_zero_quaternion_is_rejected(solver):
    with pytest.raises(ValueError, match="quaternion"):
        solver.solve([0, 0, 0, 0, 0, 0, 0])


@pytest.mark.parametrize("mode", ["invalid", "auto", "HARD", "", 1, True])
def test_collision_mode_is_validated(solver, mode):
    with pytest.raises(ValueError, match="collision must be"):
        solver.solve(POSE, collision=mode)


def test_wrong_robot_name_is_rejected():
    with pytest.raises(ValueError, match="panda"):
        hjcdik.Solver("fetch")
    assert hjcdik.Solver("panda").info.robot == "panda"


def test_target_quaternion_is_normalized_at_boundary(solver):
    target = solver.sample_targets(1, seed=11)[0]
    scaled = list(target[:3]) + [3.0 * x for x in target[3:]]
    normalized = solver.solve(target, batch_size=128, num_solutions=1, collision=None)
    rescaled = solver.solve(scaled, batch_size=128, num_solutions=1, collision=None)
    assert normalized.count.tolist() == rescaled.count.tolist()
    assert np.allclose(normalized.pos_err_mm, rescaled.pos_err_mm, atol=1e-5)
    assert np.allclose(normalized.ori_err_rad, rescaled.ori_err_rad, atol=1e-6)


@pytest.mark.parametrize("scale", [1e-300, 1e300])
def test_extreme_finite_quaternions_normalize_safely(solver, scale):
    target = solver.sample_targets(1, seed=29)[0]
    scaled = list(target[:3]) + [scale * x for x in target[3:]]
    result = solver.solve(scaled, batch_size=128, collision=None)[0]
    assert result.count == 1
    assert np.isfinite(result.pose).all()
    assert result.pos_err_mm[0] < 1.0


def test_oversized_batch_is_rejected_before_allocation(solver):
    with pytest.raises(ValueError, match="indexing"):
        solver.solve(POSE, batch_size=2**31 - 1)
    assert solver.info.max_batch_size == (2**31 - 1 - 255) // (16 * solver.info.dof)


def test_no_visible_gpu_raises_without_terminating_python():
    # A subprocess is essential: the parent already has a CUDA context, and the old
    # generated error path exited the interpreter instead of raising an exception.
    code = """
import hjcdik
try:
    hjcdik.Solver()
except RuntimeError as error:
    assert "CUDA" in str(error)
else:
    raise AssertionError("constructing without a device should fail")
solver = hjcdik.Solver(device=0)   # binding to a device number does not touch CUDA
try:
    solver.solve([0, 0, 0, 1, 0, 0, 0], batch_size=0)
except ValueError:
    pass
else:
    raise AssertionError("invalid input was not rejected before CUDA initialization")
for call in (lambda: solver.sample_targets(1),
             lambda: solver.solve([0, 0, 0, 1, 0, 0, 0], collision=None)):
    try:
        call()
    except RuntimeError as error:
        assert "CUDA" in str(error) or "cuda" in str(error), str(error)
    else:
        raise AssertionError("expected a CUDA exception")
print("interpreter survived")
"""
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert "interpreter survived" in run.stdout


def test_solution_fallback_removes_duplicates(solver):
    target = solver.sample_targets(1, seed=43)[0]
    result = solver.solve(target, batch_size=1, num_solutions=100, collision=None)[0]
    assert result.count > 0
    for index, q in enumerate(result.q):
        for previous in result.q[:index]:
            assert np.max(np.abs(q - previous)) > 1e-7


def test_gpu_entry_points_are_thread_safe(solver):
    targets = solver.sample_targets(2, seed=17)

    def solve(target):
        return int(solver.solve(target, batch_size=128, num_solutions=1, collision=None).count[0])

    with ThreadPoolExecutor(max_workers=2) as pool:
        counts = list(pool.map(solve, targets))
    assert counts == [1, 1]


@pytest.mark.parametrize("num_targets", [0, -1])
def test_sample_targets_requires_positive_count(solver, num_targets):
    with pytest.raises(ValueError, match="num_targets must be positive"):
        solver.sample_targets(num_targets)


def test_sample_targets_rejects_launch_index_overflow_before_cuda():
    run = subprocess.run(
        [sys.executable, "-c", """
import hjcdik
try:
    hjcdik.Solver(device=0).sample_targets(2**31 - 1)
except ValueError as error:
    assert "indexing" in str(error)
else:
    raise AssertionError("overflowing launch count was accepted")
"""], capture_output=True, text=True, timeout=30,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert run.returncode == 0, run.stdout + run.stderr


@pytest.mark.parametrize("target", [
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [[[0, 0, 0, 1, 0, 0, 0]]],
])
def test_target_requires_seven_vectors(solver, target):
    with pytest.raises(ValueError):
        solver.solve(target)


def test_noncontiguous_numpy_target_is_accepted_without_mutation(solver):
    target = np.repeat(solver.sample_targets(1, seed=53)[0], 2)[::2]
    assert not target.flags.c_contiguous
    original = target.copy()
    result = solver.solve(target, batch_size=128, collision=None)
    assert result.count.tolist() == [1]
    np.testing.assert_array_equal(target, original)


def test_result_arrays_own_their_storage_and_survive_later_calls(solver):
    target = solver.sample_targets(1, seed=59)[0]
    result = solver.solve(target, batch_size=128, num_solutions=2, collision=None)
    count = int(result.count[0])
    assert 0 < count <= 2
    shapes = {"q": (1, 2, solver.info.dof), "pose": (1, 2, 7), "pos_err_mm": (1, 2), "ori_err_rad": (1, 2)}
    arrays = {key: getattr(result, key) for key in shapes}
    snapshots = {key: value.copy() for key, value in arrays.items()}
    for key, value in arrays.items():
        assert value.shape == shapes[key]
        assert value.dtype == np.float64
        assert value.flags.owndata and value.flags.c_contiguous and not value.flags.writeable
        assert np.isfinite(value[0, :count]).all()
    del result
    solver.solve(target, batch_size=256, collision=None)
    for key, value in arrays.items():
        np.testing.assert_array_equal(value, snapshots[key])


def test_public_names_have_interactive_help():
    for name in hjcdik.__all__:
        assert len(getattr(hjcdik, name).__doc__ or "") > 40, name
    help_text = hjcdik.Solver.solve.__doc__
    for contract in ("candidate", "num_solutions", "collision", "precision", "float64"):
        assert contract in help_text


def test_build_metadata_matches_header_without_a_visible_gpu():
    run = subprocess.run(
        [sys.executable, "-c", "import json, hjcdik; print(json.dumps(hjcdik.build_info()))"],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert run.returncode == 0, run.stdout + run.stderr
    info = json.loads(run.stdout)
    header = Path(__file__).resolve().parents[1] / "csrc/generated/grid.cuh"
    assert info["grid_header_sha256"] == hashlib.sha256(header.read_bytes()).hexdigest()
    assert info["robot"] == "panda"
    assert info["num_joints"] == 7
    assert info["collision_enabled"] is True
    assert info["ee_target"] == "panda_grasptarget_hand"
    assert all(part.isdigit() for part in info["cuda_compiler_version"].split("."))
    solver = hjcdik.Solver()
    assert solver.info.grid_header_sha256 == info["grid_header_sha256"]
    assert solver.info.ee_frame == info["ee_target"] and solver.info.dof == info["num_joints"]
