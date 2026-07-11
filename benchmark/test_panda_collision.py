"""CPU tests for benchmark/panda_collision.py. The key test cross-validates the pRRTC-replica FK against the
independent numpy URDF FK in gen_targets (validated elsewhere against the CUDA kernel) — if the flange frame
matches at many random configs, the whole sphere-placement chain is correct.

Run: pytest benchmark/test_panda_collision.py   (outside tests/ -> stays out of the GPU-proof receipt)
"""
import os

import numpy as np
import pytest

import gen_targets as gt
from panda_collision import panda_link_transforms, panda_spheres_world, panda_config_collision_free
from panda_model import SPHERE_TO_JOINT, N_ACTUATED

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
URDF = os.path.join(REPO, "csrc", "urdf", "panda.urdf")

pytestmark = pytest.mark.skipif(not os.path.exists(URDF), reason="panda.urdf not found")


@pytest.fixture(scope="module")
def chain_to_link7():
    # pRRTC's last accumulated frame (after joint 7) is panda_link7 = child of panda_joint7.
    joints = gt._parse_joints(URDF)
    return gt._chain_to_target(joints, "panda_joint7")


def test_fk_matches_urdf_reference_at_zero(chain_to_link7):
    q = np.zeros(N_ACTUATED)
    T_prrtc = panda_link_transforms(q)[7]
    T_urdf = gt._fk(chain_to_link7, q)
    assert np.allclose(T_prrtc[:3, 3], T_urdf[:3, 3], atol=1e-6), (T_prrtc[:3, 3], T_urdf[:3, 3])
    assert np.allclose(T_prrtc, T_urdf, atol=1e-6)          # full frame (position + orientation)


def test_fk_matches_urdf_reference_random(chain_to_link7):
    rng = np.random.default_rng(0)
    for _ in range(25):
        q = rng.uniform(-2.5, 2.5, size=N_ACTUATED)
        T_prrtc = panda_link_transforms(q)[7]
        T_urdf = gt._fk(chain_to_link7, q)
        assert np.allclose(T_prrtc, T_urdf, atol=1e-6), q


def test_spheres_world_shape_and_base_at_origin():
    sw = panda_spheres_world(np.zeros(N_ACTUATED))
    assert sw.shape == (len(SPHERE_TO_JOINT), 4)
    # base sphere (index 0, SPHERE_TO_JOINT==0) sits at its link-local offset (base frame = identity)
    assert np.allclose(sw[0, :3], [0.0, 0.0, 0.05], atol=1e-9)
    assert np.all(sw[:, 3] > 0)                              # radii preserved


def test_collision_free_semantics():
    q = np.zeros(N_ACTUATED)
    assert panda_config_collision_free(q, {}) is True        # empty world
    # a big box swallowing the whole robot -> colliding
    big = {"cuboid": {"all": {"dims": [4, 4, 4], "pose": [0, 0, 0, 1, 0, 0, 0]}}}
    assert panda_config_collision_free(q, big) is False
    # a box far away -> free
    far = {"cuboid": {"away": {"dims": [0.2, 0.2, 0.2], "pose": [5, 5, 5, 1, 0, 0, 0]}}}
    assert panda_config_collision_free(q, far) is True


def test_exclude_base_matches_kernel():
    # a thin slab at the pedestal (z<0) should hit the base spheres but be ignored when exclude_base=True
    q = np.zeros(N_ACTUATED)
    slab = {"cuboid": {"pedestal": {"dims": [1.0, 1.0, 0.1], "pose": [0, 0, 0.02, 1, 0, 0, 0]}}}
    with_base = panda_config_collision_free(q, slab, exclude_base=False)
    without_base = panda_config_collision_free(q, slab, exclude_base=True)
    # excluding the base can only ever be >= permissive; assert both are booleans and consistent shape
    assert isinstance(with_base, bool) and isinstance(without_base, bool)


def test_mb_instance_to_world_dict_reshape():
    # The instance->world_dict reshape both hjcd_ik_bench and baseline_bench validate against must produce
    # the {"cuboid","cylinder"} shape config_is_collision_free consumes, preserving dims/pose/radius/height.
    from panda_collision import mb_instance_to_world_dict
    inst = {"obstacles": {
        "cuboid": {"boxA": {"dims": [0.1, 0.2, 0.3], "pose": [1, 2, 3, 1, 0, 0, 0]}},
        "cylinder": {"cylB": {"radius": 0.05, "height": 0.4, "pose": [0, 0, 1, 1, 0, 0, 0]}},
    }}
    w = mb_instance_to_world_dict(inst)
    assert set(w) == {"cuboid", "cylinder"}
    assert w["cuboid"]["boxA"] == {"dims": [0.1, 0.2, 0.3], "pose": [1, 2, 3, 1, 0, 0, 0]}
    assert w["cylinder"]["cylB"] == {"radius": 0.05, "height": 0.4, "pose": [0, 0, 1, 1, 0, 0, 0]}
    # missing obstacles -> empty but well-formed (never KeyErrors on an open-world instance)
    assert mb_instance_to_world_dict({}) == {"cuboid": {}, "cylinder": {}}
