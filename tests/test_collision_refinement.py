"""Small cases also usable under Racecheck/Synccheck; no timing assertions."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import hjcdik

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))


@pytest.mark.parametrize("precision,warps", [(0, 1), (0, 3), (1, 3), (1, 4)])
@pytest.mark.parametrize("stop,repair", [(0, 0), (1, 0), (2, 4), (3, 4)])
def test_collision_refinement_partial_blocks(monkeypatch, precision, warps, stop, repair):
    monkeypatch.setenv("HJCD_LM_WARPS", str(warps))
    monkeypatch.setenv("HJCD_CC_STOP", str(stop))
    monkeypatch.setenv("HJCD_REPAIR_ATTEMPTS", str(repair))
    root = Path(__file__).resolve().parents[1]
    p = json.loads((root / "tests/mb_problems.json").read_text())["problems"]["bookshelf_small_panda"][0]
    from query_results import panda_hand_target, pose_errors
    from panda_collision import mb_instance_to_world_dict, panda_config_collision_free
    g = p["goal_pose"]
    target = panda_hand_target(g["position_xyz"] + g["quaternion_wxyz"], hjcdik.build_info()["ee_target"])
    r = hjcdik.generate_solutions(target, batch_size=8, num_solutions=4, refine_fp64=precision,
        collision_free=True, collision_mode="both", problems_json_text=json.dumps({"problems": {"audit": [p]}}),
        problem_set_name="audit")
    assert 0 <= r["count"] <= 4
    for i, q in enumerate(r["joint_config"]):
        assert np.isfinite(q).all()
        assert panda_config_collision_free(q, mb_instance_to_world_dict(p), model="hjcd")
        pe, oe = pose_errors(q, target, hjcdik.build_info()["ee_target"])
        assert abs(pe - r["pos_errors"][i]) < .001
        assert abs(oe - r["ori_errors"][i]) < 1e-5


@pytest.mark.parametrize("precision", [0, 1])
@pytest.mark.parametrize("repair", [0, 4])
def test_collision_repair_never_returns_a_blocked_fallback(monkeypatch, precision, repair):
    monkeypatch.setenv("HJCD_REPAIR_ATTEMPTS", str(repair))
    monkeypatch.setenv("HJCD_CC_STOP", "3")
    target = hjcdik.sample_targets(1, seed=42)[0]
    scene = json.dumps({"problems": {"blocked": [{"obstacles": {
        "sphere": [{"radius": 10, "position": [0, 0, 0]}]}}]}})
    r = hjcdik.generate_solutions(target, batch_size=8, num_solutions=1, refine_fp64=precision,
        collision_free=True, collision_mode="hard", problems_json_text=scene, problem_set_name="blocked")
    assert r["count"] == 0
    assert r["joint_config"].shape == (0, 7)
