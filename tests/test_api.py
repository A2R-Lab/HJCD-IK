"""Public Python API validation and normalization contracts."""
import math
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


@pytest.mark.parametrize("num_targets", [0, -1])
def test_sample_targets_requires_positive_count(num_targets):
    with pytest.raises(ValueError, match="num_targets must be positive"):
        hjcdik.sample_targets(num_targets=num_targets)
