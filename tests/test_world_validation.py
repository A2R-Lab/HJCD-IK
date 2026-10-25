"""Scale-robust conversion and validation before changing a world or resolving a robot."""
import numpy as np
import pytest
import hjcdik


@pytest.mark.parametrize('scale', [1e-320, 1e-200, 1., 1e30])
def test_world_quaternion_scale(scale):
    q = np.array([1., 1., 0., 0.]) * scale
    box = np.r_[[1, 2, 3], q, [.1, .2, .3]]
    cylinder = np.r_[[1, 2, 3], q, [.5, 2.]]
    arrays = hjcdik.world_arrays(boxes=[box], cylinders=[cylinder])
    assert all(a.dtype == np.float64 and a.flags.c_contiguous for a in arrays.values())
    np.testing.assert_allclose(arrays['boxes'][0, 3:7], [2**-.5, 2**-.5, 0, 0])
    np.testing.assert_allclose(arrays['capsules'][0], [1, 3, 3, 1, 1, 3, .5], atol=1e-14)
    # Validation must not normalize the caller's array in place.
    np.testing.assert_array_equal(box[3:7], q)
    solver = hjcdik.Solver()
    solver.set_world(boxes=[box], cylinders=[cylinder])
    assert solver.world_counts == dict(boxes=1, capsules=1, spheres=0)


def test_derived_endpoints_rejected_before_upload(monkeypatch):
    from hjcdik import _solver
    solver = hjcdik.Solver()
    solver.set_world(spheres=[[0, 0, 0, 100]])
    previous = solver._world
    def no_upload(*a, **k):
        pytest.fail('invalid converted world reached upload')
    monkeypatch.setattr(_solver._native, 'World', no_upload)
    maximum = float(np.finfo(np.float32).max)
    with pytest.raises(ValueError, match='fp32'):
        solver.set_world(cylinders=[[0, 0, maximum, 1, 0, 0, 0, 1, maximum]])
    assert solver._world is previous
    for quat in ([0, 0, 0, 0], [np.nan, 0, 0, 0], [np.inf, 0, 0, 0]):
        with pytest.raises(ValueError):
            solver.set_world(boxes=[[0, 0, 0, *quat, 1, 1, 1]])
        assert solver._world is previous


@pytest.mark.parametrize('device', [-1, np.int64(-2)])
def test_negative_device_rejected_before_resolution(monkeypatch, device):
    def no_resolve(*a, **k):
        pytest.fail('negative device reached model resolution')
    monkeypatch.setattr(hjcdik.Solver, '_resolve', no_resolve)
    with pytest.raises(ValueError, match='device.*non-negative'):
        hjcdik.Solver('missing.urdf', device=device)


def test_explicit_numpy_device():
    assert hjcdik.Solver(device=np.int64(0)).info.device == 0
