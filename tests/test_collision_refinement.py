"""Small cases also usable under Racecheck/Synccheck; no timing assertions."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import hjcdik

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))


@pytest.mark.parametrize("precision,warps", [("fp32", 1), ("fp32", 3), ("fp64", 3), ("fp64", 4)])
@pytest.mark.parametrize("stop,repair", [(0, 0), (1, 0), (2, 4), (3, 4)])
def test_collision_refinement_partial_blocks(monkeypatch, precision, warps, stop, repair):
    monkeypatch.setenv("HJCD_LM_WARPS", str(warps))
    monkeypatch.setenv("HJCD_CC_STOP", str(stop))
    monkeypatch.setenv("HJCD_REPAIR_ATTEMPTS", str(repair))
    root = Path(__file__).resolve().parents[1]
    p = json.loads((root / "tests/mb_problems.json").read_text())["problems"]["bookshelf_small_panda"][0]
    from query_results import panda_hand_target, pose_errors
    from panda_collision import mb_instance_to_world_dict, panda_config_collision_free
    solver = hjcdik.Solver()
    solver.set_world(**hjcdik.worlds.from_mbm_problem(p))
    target = panda_hand_target(hjcdik.worlds.goal_pose(p), solver.info.ee_frame)
    r = solver.solve(target, batch_size=8, num_solutions=4, precision=precision, collision="both")[0]
    assert 0 <= r.count <= 4
    for i, q in enumerate(r.q):
        assert np.isfinite(q).all()
        assert panda_config_collision_free(q, mb_instance_to_world_dict(p), model="hjcd")
        pe, oe = pose_errors(q, target, solver.info.ee_frame)
        assert abs(pe - r.pos_err_mm[i]) < .001
        assert abs(oe - r.ori_err_rad[i]) < 1e-5


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
@pytest.mark.parametrize("repair", [0, 4])
def test_collision_repair_never_returns_a_blocked_fallback(monkeypatch, precision, repair):
    monkeypatch.setenv("HJCD_REPAIR_ATTEMPTS", str(repair))
    monkeypatch.setenv("HJCD_CC_STOP", "3")
    solver = hjcdik.Solver()
    target = solver.sample_targets(1, seed=42)[0]
    solver.set_world(spheres=[[0, 0, 0, 10]])
    r = solver.solve(target, batch_size=8, num_solutions=1, precision=precision, collision="hard")[0]
    assert r.count == 0
    assert r.q.shape == (0, 7)
