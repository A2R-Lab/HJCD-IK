"""Collision-free path smoke/regression over the MotionBenchMaker problem sets.

Exercises generate_solutions(collision_free=True, ...) end-to-end (the env-collision scoring runs after
the LM refine). Asserts the path runs and returns a reachable, low-error solution on a sampled set of
problems — a guard that the collision path (and its interaction with the multi-warp / precision defaults)
keeps working. NOT a tight collision-quality baseline (that's a backlog item); this is a no-crash +
solved smoke.

Requires a CUDA GPU + built `hjcdik` + tests/mb_problems.json; skips cleanly otherwise.
"""
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
hjcdik = pytest.importorskip("hjcdik")

HERE = Path(__file__).parent
MB_PATH = HERE / "mb_problems.json"


def _goal7(entry):
    gp = entry["goal_pose"]
    return list(gp["position_xyz"]) + list(gp["quaternion_wxyz"])


@pytest.mark.skipif(not MB_PATH.exists(), reason="tests/mb_problems.json missing")
def test_collision_free_runs_and_solves():
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
