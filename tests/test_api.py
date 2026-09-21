"""Public Python API validation and normalization contracts."""
import math
import os
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


def test_collision_mode_is_validated():
    target = [0, 0, 0, 1, 0, 0, 0]
    with pytest.raises(ValueError, match="collision_mode"):
        hjcdik.generate_solutions(target, collision_mode="invalid")


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
