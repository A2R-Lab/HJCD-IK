"""Public Python API validation and normalization contracts."""
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


@pytest.mark.parametrize("batch_size,num_solutions", [(0, 1), (-1, 1), (10, 0), (10, -2)])
def test_positive_sizes_required(batch_size, num_solutions):
    target = [0, 0, 0, 1, 0, 0, 0]
    with pytest.raises(ValueError):
        hjcdik.generate_solutions(target, batch_size=batch_size, num_solutions=num_solutions)


@pytest.mark.parametrize("value", [-2, 2, 99])
def test_refine_precision_enum_is_validated(value):
    target = [0, 0, 0, 1, 0, 0, 0]
    with pytest.raises(ValueError, match="refine_fp64"):
        hjcdik.generate_solutions(target, refine_fp64=value)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_target_values_must_be_finite(value):
    target = [0, 0, 0, 1, 0, 0, 0]
    for index in range(7):
        bad = list(target)
        bad[index] = value
        with pytest.raises(ValueError, match="finite"):
            hjcdik.generate_solutions(bad)


def test_zero_quaternion_is_rejected():
    with pytest.raises(ValueError, match="quaternion"):
        hjcdik.generate_solutions([0, 0, 0, 0, 0, 0, 0])


@pytest.mark.parametrize("mode", ["invalid", "auto", "HARD", ""])
def test_collision_mode_is_validated(mode):
    # "auto" (the former HJCD_CC_MODE environment shim) is deliberately not accepted any more.
    target = [0, 0, 0, 1, 0, 0, 0]
    with pytest.raises(ValueError, match="collision_mode"):
        hjcdik.generate_solutions(target, collision_mode=mode)


def test_collision_arguments_are_required():
    target = [0, 0, 0, 1, 0, 0, 0]
    with pytest.raises(ValueError, match="problems_json_text"):
        hjcdik.generate_solutions(target, collision_free=True)
    with pytest.raises(ValueError, match="problem_set_name"):
        hjcdik.generate_solutions(target, collision_free=True, problems_json_text="{}")


def test_target_quaternion_is_normalized_at_boundary():
    target = hjcdik.sample_targets(num_targets=1, seed=11)[0]
    scaled = list(target[:3]) + [3.0 * x for x in target[3:]]
    normalized = hjcdik.generate_solutions(target, batch_size=128, num_solutions=1)
    rescaled = hjcdik.generate_solutions(scaled, batch_size=128, num_solutions=1)
    assert normalized["count"] == rescaled["count"]
    assert np.allclose(normalized["pos_errors"], rescaled["pos_errors"], atol=1e-5)
    assert np.allclose(normalized["ori_errors"], rescaled["ori_errors"], atol=1e-6)


@pytest.mark.parametrize("scale", [1e-300, 1e300])
def test_extreme_finite_quaternions_normalize_safely(scale):
    target = hjcdik.sample_targets(num_targets=1, seed=29)[0]
    scaled = list(target[:3]) + [scale * x for x in target[3:]]
    result = hjcdik.generate_solutions(scaled, batch_size=128)
    assert result["count"] == 1
    assert np.isfinite(result["pose"]).all()
    assert result["pos_errors"][0] < 1.0


def test_oversized_batch_is_rejected_before_allocation():
    with pytest.raises(ValueError, match="indexing"):
        hjcdik.generate_solutions([0, 0, 0, 1, 0, 0, 0], batch_size=2**31 - 1)


def test_no_visible_gpu_raises_without_terminating_python():
    # A subprocess is essential: the parent already has a CUDA context, and the old
    # generated error path exited the interpreter instead of raising an exception.
    code = """
import hjcdik
try:
    hjcdik.generate_solutions([0, 0, 0, 1, 0, 0, 0], batch_size=0)
except ValueError:
    pass
else:
    raise AssertionError("invalid input was not rejected before CUDA initialization")
for call in (lambda: hjcdik.sample_targets(1),
             lambda: hjcdik.generate_solutions([0, 0, 0, 1, 0, 0, 0])):
    try:
        call()
    except RuntimeError as error:
        assert "CUDA" in str(error)
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


def test_stats_append_across_precision_modes_and_existing_calls(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = hjcdik.sample_targets(1, seed=31)[0]
    for precision in (1, 0, 1):
        hjcdik.generate_solutions(target, batch_size=32, refine_fp64=precision, write_stats=True)
    lines = (tmp_path / "ik_stats.csv").read_text().splitlines()
    assert len(lines) == 4
    assert lines[0].startswith("b_size,krep,")
    assert all(line.startswith("32,") for line in lines[1:])


def test_stats_write_failure_raises_and_next_solve_recovers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ik_stats.csv").mkdir()
    target = hjcdik.sample_targets(1, seed=37)[0]
    with pytest.raises(RuntimeError, match="cannot open ik_stats.csv"):
        hjcdik.generate_solutions(target, batch_size=32, write_stats=True)
    assert hjcdik.generate_solutions(target, batch_size=32)["count"] == 1


def test_solution_fallback_removes_duplicates():
    target = hjcdik.sample_targets(1, seed=43)[0]
    result = hjcdik.generate_solutions(target, batch_size=1, num_solutions=100)
    configurations = np.asarray(result["joint_config"])
    assert result["count"] > 0
    for index, q in enumerate(configurations):
        for previous in configurations[:index]:
            assert np.max(np.abs(q - previous)) > 1e-7


def test_gpu_entry_points_are_thread_safe():
    targets = hjcdik.sample_targets(num_targets=2, seed=17)

    def solve(target):
        return hjcdik.generate_solutions(target, batch_size=128, num_solutions=1)["count"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        counts = list(pool.map(solve, targets))
    assert counts == [1, 1]


@pytest.mark.parametrize("num_targets", [0, -1])
def test_sample_targets_requires_positive_count(num_targets):
    with pytest.raises(ValueError, match="num_targets must be positive"):
        hjcdik.sample_targets(num_targets=num_targets)


def test_sample_targets_rejects_launch_index_overflow_before_cuda():
    run = subprocess.run(
        [sys.executable, "-c", """
import hjcdik
try:
    hjcdik.sample_targets(2**31 - 1)
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
    [[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]],
])
def test_target_requires_one_seven_vector(target):
    with pytest.raises(TypeError):
        hjcdik.generate_solutions(target)


def test_noncontiguous_numpy_target_is_accepted_without_mutation():
    target = np.repeat(hjcdik.sample_targets(1, seed=53)[0], 2)[::2]
    assert not target.flags.c_contiguous
    original = target.copy()
    result = hjcdik.generate_solutions(target, batch_size=128)
    assert result["count"] == 1
    np.testing.assert_array_equal(target, original)


def test_result_arrays_own_their_storage_and_survive_later_calls():
    target = hjcdik.sample_targets(1, seed=59)[0]
    result = hjcdik.generate_solutions(target, batch_size=128, num_solutions=2)
    count = result["count"]
    assert 0 < count <= 2
    shapes = {"joint_config": (count, hjcdik.num_joints()), "pose": (count, 7),
              "pos_errors": (count,), "ori_errors": (count,)}
    arrays = {key: result[key] for key in shapes}
    snapshots = {key: value.copy() for key, value in arrays.items()}
    for key, value in arrays.items():
        assert value.shape == shapes[key]
        assert value.dtype == np.float64
        assert value.flags.owndata and value.flags.c_contiguous
        assert np.isfinite(value).all()
    del result
    hjcdik.generate_solutions(target, batch_size=256)
    for key, value in arrays.items():
        np.testing.assert_array_equal(value, snapshots[key])


def test_public_functions_have_interactive_help():
    for name in hjcdik.__all__:
        assert len(getattr(hjcdik, name).__doc__ or "") > 60
    help_text = hjcdik.generate_solutions.__doc__
    for contract in ("millimeters", "radians", "scalar-first", "does NOT", "zero", "TypeError"):
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
    assert info["num_joints"] == hjcdik.num_joints()
    assert info["collision_enabled"] == hjcdik.collision_enabled()
    assert info["ee_target"] == "panda_grasptarget_hand"
    assert all(part.isdigit() for part in info["cuda_compiler_version"].split("."))
