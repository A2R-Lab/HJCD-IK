"""Offline query accounting, frame, and oracle contracts (no competitor installation)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))
from query_results import best_candidate, query_record, validate_groups, pose_errors, panda_hand_target
from panda_collision import mb_instance_to_world_dict
from collision_check import config_is_collision_free


def test_empty_query_is_serialized_and_counts_as_an_attempt():
    result = dict(count=0, pos_errors=[], ori_errors=[], joint_config=[])
    r = query_record(result, solver="test", problem_set="s", problem_idx=0,
                     batch=8, target=[0, 0, 0, 1, 0, 0, 0], ee_target="panda_hand_joint", elapsed_ms=2)
    assert r["q"] is None and r["status"] == "empty" and r["time_ms"] == 2
    json.dumps(r, allow_nan=False)
    assert len(validate_groups([r], {"s": [{}]})[("s", "test", 8)]) == 1
    with pytest.raises(ValueError, match="incomplete"):
        validate_groups([r], {"s": [{}, {}]})
    with pytest.raises(ValueError, match="duplicate"):
        validate_groups([r, r], {"s": [{}]})
    with pytest.raises(ValueError, match="missing query group"):
        validate_groups([r], {"s": [{}], "missing_set": [{}]})


def test_candidate_selection_requires_both_errors_on_one_candidate():
    r = dict(count=3, pos_errors=[.01, 8, 1], ori_errors=[.2, .001, .01])
    assert best_candidate(r) == 2


def test_hand_to_tcp_conversion_preserves_physical_hand_goal():
    from gen_targets import _fk, _quat_from_R
    from query_results import _pose_chain
    q = np.array([0, -.3, .1, -1.5, .2, 1.8, .4])
    hand = _fk(_pose_chain("panda_hand_joint"), q)
    goal = [*hand[:3, 3], *_quat_from_R(hand[:3, :3])]
    target = panda_hand_target(goal, "panda_grasptarget_hand")
    pe, oe = pose_errors(q, target, "panda_grasptarget_hand")
    assert pe < 1e-8 and oe < 1e-7
    assert np.linalg.norm(np.asarray(goal[:3]) - target[:3]) == pytest.approx(.105)


def test_sphere_obstacles_cannot_disappear_from_validation():
    world = mb_instance_to_world_dict({"obstacles": {"sphere": [dict(radius=1, position=[0, 0, 0])]}})
    assert not config_is_collision_free([[0, 0, 0, .1]], world)
    assert config_is_collision_free([[3, 0, 0, .1]], world)
    for fn in (lambda: mb_instance_to_world_dict({"obstacles": {"mesh": {}}}),
               lambda: config_is_collision_free([[0, 0, 0, .1]], {"mesh": {}})):
        with pytest.raises(ValueError, match="unsupported"):
            fn()


def test_mesh_oracle_constructs_spheres_without_optional_fcl():
    from collision_oracles import MeshOracle
    from types import SimpleNamespace
    objects = []
    manager = SimpleNamespace(add_object=lambda *a, **k: objects.append((a, k)))
    mo = MeshOracle.__new__(MeshOracle)
    mo._world_cache = {}
    mo._trimesh = SimpleNamespace(collision=SimpleNamespace(CollisionManager=lambda: manager),
                                  primitives=SimpleNamespace(Sphere=lambda **k: k))
    mo._world_manager({"sphere": {"s": {"radius": .2, "position": [0, 0, 0]}}})
    assert objects[0][0] == ("sphere:s", {"radius": .2})
