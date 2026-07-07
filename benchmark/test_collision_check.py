"""CPU unit tests for benchmark/collision_check.py (pure numpy; no GPU).

Lives outside tests/ on purpose: tests/ is auto-marked gpu_proof (every item goes in the GPU-signed
receipt), and this is a CPU-only unit. Run explicitly:  pytest benchmark/test_collision_check.py
"""
import numpy as np
import pytest

from collision_check import (
    sphere_cuboid_signed_distance,
    sphere_cylinder_signed_distance,
    config_is_collision_free,
    quat_wxyz_to_rot,
)

IDENT = [0, 0, 0, 1, 0, 0, 0]  # pose at origin, identity orientation


# ---- cuboid ----
def test_box_sphere_clear():
    # unit box (half=0.5) at origin; sphere far on +x
    d = sphere_cuboid_signed_distance([2, 0, 0], 0.1, IDENT, [1, 1, 1])
    assert d == pytest.approx(1.4, abs=1e-9)          # 2 - 0.5 - 0.1


def test_box_sphere_grazing_collision():
    d = sphere_cuboid_signed_distance([0.55, 0, 0], 0.1, IDENT, [1, 1, 1])
    assert d == pytest.approx(-0.05, abs=1e-9)        # 0.05 - 0.1
    assert d < 0


def test_box_sphere_center_inside():
    d = sphere_cuboid_signed_distance([0, 0, 0], 0.1, IDENT, [1, 1, 1])
    assert d == pytest.approx(-0.6, abs=1e-9)         # inside depth 0.5 + r 0.1
    assert d < 0


def test_box_corner_distance():
    # sphere off a corner of the unit box: nearest point is the corner (0.5,0.5,0.5)
    c = [0.5 + 0.3, 0.5 + 0.4, 0.5, 0.0]
    d = sphere_cuboid_signed_distance(c[:3], 0.0, IDENT, [1, 1, 1])
    assert d == pytest.approx(0.5, abs=1e-9)          # sqrt(0.3^2+0.4^2)


def test_box_rotation_invariance():
    # rotate a non-cubic box 90 deg about z; a point that was clear stays clear by the same margin.
    dims = [2.0, 0.4, 0.4]
    # 90deg about z: quat wxyz = [cos45, 0,0, sin45]
    s2 = np.sqrt(0.5)
    pose_rot = [0, 0, 0, s2, 0, 0, s2]
    # point along +y at 1.5 : after rotating the box, its long axis (was x) points along y
    d_rot = sphere_cuboid_signed_distance([0, 1.5, 0], 0.1, pose_rot, dims)
    # equivalent unrotated: point along +x at 1.5 vs long-axis box
    d_ref = sphere_cuboid_signed_distance([1.5, 0, 0], 0.1, IDENT, dims)
    assert d_rot == pytest.approx(d_ref, abs=1e-9)


# ---- cylinder ----
def test_cyl_radial_clear():
    d = sphere_cylinder_signed_distance([0.5, 0, 0], 0.1, IDENT, 0.2, 1.0)
    assert d == pytest.approx(0.2, abs=1e-9)          # 0.5-0.2-0.1


def test_cyl_radial_collision():
    d = sphere_cylinder_signed_distance([0.25, 0, 0], 0.1, IDENT, 0.2, 1.0)
    assert d == pytest.approx(-0.05, abs=1e-9)
    assert d < 0


def test_cyl_axial_boundary():
    # above the flat top: axial gap 0.1 exactly equals r -> grazing (0)
    d = sphere_cylinder_signed_distance([0, 0, 0.6], 0.1, IDENT, 0.2, 1.0)
    assert d == pytest.approx(0.0, abs=1e-9)


def test_cyl_center_inside():
    d = sphere_cylinder_signed_distance([0, 0, 0], 0.1, IDENT, 0.2, 1.0)
    assert d < 0


# ---- config-level mask ----
def test_config_free_and_colliding():
    world = {
        "cuboid": {"box": {"dims": [1, 1, 1], "pose": IDENT}},
        "cylinder": {"post": {"radius": 0.1, "height": 2.0, "pose": [1, 0, 0, 1, 0, 0, 0]}},
    }
    clear = np.array([[3, 3, 3, 0.1], [0, 3, 0, 0.1]])          # both spheres far away
    assert config_is_collision_free(clear, world) is True
    hits_box = np.array([[3, 3, 3, 0.1], [0.4, 0, 0, 0.2]])     # 2nd sphere in the box
    assert config_is_collision_free(hits_box, world) is False
    hits_cyl = np.array([[1.0, 0, 0, 0.05]])                    # inside the post
    assert config_is_collision_free(hits_cyl, world) is False


def test_empty_world_is_free():
    assert config_is_collision_free(np.array([[0, 0, 0, 1.0]]), {}) is True
    assert config_is_collision_free(np.array([[0, 0, 0, 1.0]]), {"cuboid": {}, "cylinder": {}}) is True


def test_margin_requires_clearance():
    world = {"cuboid": {"box": {"dims": [1, 1, 1], "pose": IDENT}}}
    sphere = np.array([[0.62, 0, 0, 0.1]])                      # gap = 0.12-0.10 = 0.02
    assert config_is_collision_free(sphere, world, margin=0.0) is True
    assert config_is_collision_free(sphere, world, margin=0.05) is False   # needs 5cm clearance


def test_quat_identity():
    assert np.allclose(quat_wxyz_to_rot([1, 0, 0, 0]), np.eye(3))
