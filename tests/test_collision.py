"""Collision-free correctness/regression tests over MotionBenchMaker problem sets.

Exercises generate_solutions(collision_free=True, ...) end-to-end (the env-collision scoring runs after
the LM refine). Asserts the path runs, returns a reachable solution on a sampled set of problems, and
independently validates every returned configuration. This guards the strict filtering contract plus the
path's interaction with multi-warp and precision defaults.

Requires a CUDA GPU + built `hjcdik` + tests/mb_problems.json; skips cleanly otherwise.
"""
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
hjcdik = pytest.importorskip("hjcdik")

HERE = Path(__file__).parent
MB_PATH = HERE / "mb_problems.json"
sys.path.insert(0, str(HERE.parent / "benchmark"))
from panda_collision import mb_instance_to_world_dict, panda_config_collision_free  # noqa: E402


@pytest.mark.parametrize("obstacles,match", [
    (None, "obstacles must be an object"),
    ([], "obstacles must be an object"),
    ({"mesh": {}}, "unsupported obstacle type"),
    ({"spheres": {}}, "unsupported obstacle type"),
    ({"sphere": [{"radius": -1, "position": [0, 0, 0]}]}, "positive"),
    ({"sphere": [{"radius": 1e300, "position": [0, 0, 0]}]}, "finite"),
    ({"sphere": [{"radius": 1, "position": [0, 0]}]}, "three"),
    ({"cuboid": [{"dims": [1, 0, 1], "pose": [0, 0, 0, 1, 0, 0, 0]}]}, "positive"),
    ({"cuboid": [{"dims": [1, 1, 1], "pose": [0, 0, 0, 0, 0, 0, 0]}]}, "quaternion"),
    ({"cylinder": [{"radius": 1, "height": -1, "pose": [0, 0, 0, 1, 0, 0, 0]}]}, "positive"),
])
def test_malformed_collision_geometry_is_rejected(obstacles, match):
    text = json.dumps({"problems": {"malformed": [{"obstacles": obstacles}]}})
    with pytest.raises(ValueError, match=match):
        hjcdik.generate_solutions(
            [0.4, 0, 0.4, 1, 0, 0, 0], batch_size=32, collision_free=True,
            problems_json_text=text, problem_set_name="malformed",
        )


def test_explicit_empty_scene_and_valid_scene_after_parse_error():
    target = hjcdik.sample_targets(1, seed=47)[0]
    kwargs = dict(batch_size=128, collision_free=True, problem_set_name="recovery")
    with pytest.raises(ValueError, match="obstacles"):
        hjcdik.generate_solutions(
            target, problems_json_text='{"problems":{"recovery":[{}]}}', **kwargs)
    result = hjcdik.generate_solutions(
        target, problems_json_text='{"problems":{"recovery":[{"obstacles":{}}]}}', **kwargs)
    assert np.isfinite(result["pose"]).all()


def _goal7(entry):
    gp = entry["goal_pose"]
    return list(gp["position_xyz"]) + list(gp["quaternion_wxyz"])


@pytest.mark.skipif(not MB_PATH.exists(), reason="tests/mb_problems.json missing")
def test_collision_free_runs_and_solves(monkeypatch):
    monkeypatch.delenv("HJCD_CC_MODE", raising=False)
    text = MB_PATH.read_text()
    problems = json.loads(text)["problems"]
    set_name = sorted(problems.keys())[0]          # e.g. bookshelf_small_panda
    n = min(5, len(problems[set_name]))
    solved = 0
    for i in range(n):
        tgt = _goal7(problems[set_name][i])
        out = hjcdik.generate_solutions(
            tgt, batch_size=2000, num_solutions=4, collision_free=True,
            problems_json_text=text, problem_set_name=set_name, problem_idx=i)
        world = mb_instance_to_world_dict(problems[set_name][i])
        for q in np.asarray(out["joint_config"]):
            assert panda_config_collision_free(q, world), (
                f"hard collision mode returned a colliding configuration for {set_name}[{i}]")
        if out["count"] > 0:
            pe = float(np.min(np.array(out["pos_errors"], dtype=float)))
            if pe < 1.0:   # sub-mm reachable solution found
                solved += 1
    assert solved >= max(1, n - 1), f"collision-free solved only {solved}/{n} on {set_name}"


@pytest.mark.skipif(not MB_PATH.exists(), reason="tests/mb_problems.json missing")
def test_collision_free_matches_unconstrained_reach():
    """The collision-free solve should still reach the goal pose (collision scoring is post-hoc)."""
    text = MB_PATH.read_text()
    problems = json.loads(text)["problems"]
    set_name = sorted(problems.keys())[0]
    tgt = _goal7(problems[set_name][0])
    out = hjcdik.generate_solutions(
        tgt, batch_size=2000, num_solutions=4, collision_free=True,
        problems_json_text=text, problem_set_name=set_name, problem_idx=0)
    assert out["count"] > 0
    assert float(np.min(np.array(out["pos_errors"], dtype=float))) < 1.0


@pytest.mark.skipif(not MB_PATH.exists(), reason="tests/mb_problems.json missing")
def test_hard_mode_returns_no_solution_when_every_candidate_collides(monkeypatch):
    monkeypatch.delenv("HJCD_CC_MODE", raising=False)
    source = json.loads(MB_PATH.read_text())["problems"]
    base = dict(source[sorted(source.keys())[0]][0])
    base["obstacles"] = {"cuboid": {"all": {
        "dims": [4.0, 4.0, 4.0], "pose": [0, 0, 0, 1, 0, 0, 0]
    }}}
    text = json.dumps({"problems": {"strict": [base]}})
    out = hjcdik.generate_solutions(
        _goal7(base), batch_size=256, num_solutions=4, collision_free=True,
        problems_json_text=text, problem_set_name="strict", problem_idx=0)
    assert out["count"] == 0
    assert out["joint_config"].shape == (0, hjcdik.num_joints())

    soft = hjcdik.generate_solutions(
        _goal7(base), batch_size=256, num_solutions=4, collision_free=True,
        problems_json_text=text, problem_set_name="strict", problem_idx=0,
        collision_mode="soft")
    assert soft["count"] == 4  # ranking mode is explicitly non-filtering


@pytest.mark.skipif(not MB_PATH.exists(), reason="tests/mb_problems.json missing")
def test_invalid_collision_problem_is_rejected():
    source = json.loads(MB_PATH.read_text())["problems"]
    base = dict(source[sorted(source.keys())[0]][0])
    base["valid"] = False
    text = json.dumps({"problems": {"invalid": [base]}})
    with pytest.raises(RuntimeError, match="marked invalid"):
        hjcdik.generate_solutions(
            _goal7(base), collision_free=True, problems_json_text=text,
            problem_set_name="invalid", problem_idx=0)


@pytest.mark.skipif(not MB_PATH.exists(), reason="tests/mb_problems.json missing")
def test_collision_problem_index_is_checked():
    text = MB_PATH.read_text()
    problems = json.loads(text)["problems"]
    set_name = sorted(problems)[0]
    target = _goal7(problems[set_name][0])
    with pytest.raises(RuntimeError, match="problem_idx out of range"):
        hjcdik.generate_solutions(
            target, collision_free=True, problems_json_text=text,
            problem_set_name=set_name, problem_idx=10**9)


@pytest.mark.skipif(not MB_PATH.exists(), reason="tests/mb_problems.json missing")
def test_collision_cache_includes_environment_contents(monkeypatch):
    """The same set/index with changed JSON must not reuse device geometry from the first call."""
    monkeypatch.delenv("HJCD_CC_MODE", raising=False)
    source = json.loads(MB_PATH.read_text())["problems"]
    base = dict(source[sorted(source.keys())[0]][0])
    blocked = dict(base)
    blocked["obstacles"] = {"cuboid": {"all": {
        "dims": [4.0, 4.0, 4.0], "pose": [0, 0, 0, 1, 0, 0, 0]
    }}}
    clear = dict(base)
    clear["obstacles"] = {"cuboid": {"far": {
        "dims": [0.1, 0.1, 0.1], "pose": [10, 10, 10, 1, 0, 0, 0]
    }}}
    kwargs = dict(batch_size=512, num_solutions=1, collision_free=True,
                  problem_set_name="same-key", problem_idx=0)
    first = hjcdik.generate_solutions(
        _goal7(base), problems_json_text=json.dumps({"problems": {"same-key": [blocked]}}), **kwargs)
    second = hjcdik.generate_solutions(
        _goal7(base), problems_json_text=json.dumps({"problems": {"same-key": [clear]}}), **kwargs)
    assert first["count"] == 0
    assert second["count"] > 0
