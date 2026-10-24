"""Collision-free correctness over MotionBenchMaker problem sets, and the worlds loaders.

Exercises `Solver.set_world` + `solve(collision=...)` end-to-end (the env-collision scoring runs
after the LM refine). Asserts the path runs, returns a reachable solution on a sampled set of
problems, and independently validates every returned configuration. This guards the strict
filtering contract plus the path's interaction with multi-warp and precision defaults.

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
from query_results import panda_hand_target  # noqa: E402

worlds = hjcdik.worlds


@pytest.fixture
def solver():
    return hjcdik.Solver()


@pytest.mark.parametrize("obstacles,match", [
    (None, "obstacles must be an object"),
    ([], "obstacles must be an object"),
    ({"mesh": {}}, "unsupported obstacle type"),
    ({"spheres": {}}, "unsupported obstacle type"),
    ({"sphere": [{"radius": -1, "position": [0, 0, 0]}]}, "positive"),
    ({"sphere": [{"radius": 1, "position": [0, 0]}]}, "three"),
    ({"cuboid": [{"dims": [1, 0, 1], "pose": [0, 0, 0, 1, 0, 0, 0]}]}, "positive"),
    ({"cuboid": [{"dims": [1, 1, 1], "pose": [0, 0, 0, 0, 0, 0, 0]}]}, "quaternion"),
    ({"cylinder": [{"radius": 1, "height": -1, "pose": [0, 0, 0, 1, 0, 0, 0]}]}, "positive"),
    # Legacy Euler forms (pre-2026-10 `box` schema / position + orientation_euler_xyz) are gone: `pose` only.
    ({"box": [{"position": [0, 0, 0], "orientation_euler_xyz": [0, 0, 0], "half_extents": [1, 1, 1]}]},
     "unsupported obstacle type"),
    ({"cuboid": [{"position": [0, 0, 0], "orientation_euler_xyz": [0, 0, 0], "dims": [1, 1, 1]}]},
     "cuboid obstacle requires 'pose'"),
    ({"cylinder": [{"radius": 0.1, "height": 0.2}]}, "cylinder obstacle requires 'pose'"),
    ({"cuboid": [{"pose": [0, 0, 0, 1, 0, 0, 0]}]}, "cuboid obstacle requires 'dims'"),
    ({"sphere": [{"radius": 0.1}]}, "sphere obstacle requires 'position'"),
    ({"cylinder": [{"radius": 0.1, "pose": [0, 0, 0, 1, 0, 0, 0]}]}, "missing height/length"),
])
def test_malformed_mbm_geometry_is_rejected(obstacles, match):
    with pytest.raises(ValueError, match=match):
        worlds.from_mbm_problem({"obstacles": obstacles})


def test_oversized_sphere_is_rejected_at_upload(solver):
    with pytest.raises(ValueError, match="fp32"):
        solver.set_world(**worlds.from_mbm_problem({"obstacles": {"sphere": [{"radius": 1e300, "position": [0, 0, 0]}]}}))


def test_explicit_empty_scene_and_bare_obstacle_objects():
    with pytest.raises(ValueError, match="obstacles"):
        worlds.from_mbm_problem({})
    empty = worlds.from_mbm_problem({"obstacles": {}})
    assert all(value.shape[0] == 0 for value in empty.values())
    bare = worlds.from_mbm_problem({"sphere": [{"radius": 0.1, "position": [1, 2, 3]}]})
    np.testing.assert_array_equal(bare["spheres"], [[1, 2, 3, 0.1]])


def test_problem_selection_checks_set_index_and_validity():
    document = json.loads(MB_PATH.read_text())
    with pytest.raises(KeyError):
        worlds.select_problem(document, "no-such-set", 0)
    with pytest.raises(IndexError, match="out of range"):
        worlds.select_problem(document, "box_panda", 10**9)
    invalid = {"problems": {"bad": [dict(document["problems"]["box_panda"][0], valid=False)]}}
    with pytest.raises(ValueError, match="marked invalid"):
        worlds.select_problem(invalid, "bad", 0)
    loaded = worlds.from_mbm_file(MB_PATH, "box_panda", 0)
    assert loaded["boxes"].shape[1] == 10 and loaded["cylinders"].shape[1] == 9


def _goal7(entry, ee_frame):
    return panda_hand_target(worlds.goal_pose(entry), ee_frame)


@pytest.mark.skipif(not MB_PATH.exists(), reason="tests/mb_problems.json missing")
def test_collision_free_runs_and_solves(solver):
    problems = json.loads(MB_PATH.read_text())["problems"]
    set_name = sorted(problems.keys())[0]          # e.g. bookshelf_small_panda
    n = min(5, len(problems[set_name]))
    solved = 0
    for i in range(n):
        solver.set_world(**worlds.from_mbm_problem(problems[set_name][i]))
        out = solver.solve(_goal7(problems[set_name][i], solver.info.ee_frame),
                           batch_size=2000, num_solutions=4, collision="hard")[0]
        world = mb_instance_to_world_dict(problems[set_name][i])
        for q in out.q:
            assert panda_config_collision_free(q, world, model="hjcd"), (
                f"hard collision mode returned a colliding configuration for {set_name}[{i}]")
        if out.count > 0 and float(out.pos_err_mm.min()) < 1.0:   # sub-mm reachable solution found
            solved += 1
    assert solved >= max(1, n - 1), f"collision-free solved only {solved}/{n} on {set_name}"


@pytest.mark.skipif(not MB_PATH.exists(), reason="tests/mb_problems.json missing")
def test_collision_free_matches_unconstrained_reach(solver):
    """The collision-free solve should still reach the goal pose (collision scoring is post-hoc)."""
    problems = json.loads(MB_PATH.read_text())["problems"]
    set_name = sorted(problems.keys())[0]
    solver.set_world(**worlds.from_mbm_problem(problems[set_name][0]))
    out = solver.solve(_goal7(problems[set_name][0], solver.info.ee_frame),
                       batch_size=2000, num_solutions=4, collision="hard")[0]
    assert out.count > 0
    assert float(out.pos_err_mm.min()) < 1.0


@pytest.mark.skipif(not MB_PATH.exists(), reason="tests/mb_problems.json missing")
def test_hard_mode_returns_no_solution_when_every_candidate_collides(solver):
    source = json.loads(MB_PATH.read_text())["problems"]
    base = dict(source[sorted(source.keys())[0]][0])
    base["obstacles"] = {"cuboid": {"all": {"dims": [4.0, 4.0, 4.0], "pose": [0, 0, 0, 1, 0, 0, 0]}}}
    solver.set_world(**worlds.from_mbm_problem(base))
    target = _goal7(base, solver.info.ee_frame)
    out = solver.solve(target, batch_size=256, num_solutions=4, collision="hard")[0]
    assert out.count == 0
    assert out.q.shape == (0, solver.info.dof)

    soft = solver.solve(target, batch_size=256, num_solutions=4, collision="soft")[0]
    assert soft.count == 4  # ranking mode is explicitly non-filtering


@pytest.mark.skipif(not MB_PATH.exists(), reason="tests/mb_problems.json missing")
def test_world_replacement_never_reuses_old_geometry(solver):
    """Changing the world with the same target must not reuse device geometry from the first call."""
    source = json.loads(MB_PATH.read_text())["problems"]
    base = dict(source[sorted(source.keys())[0]][0])
    blocked = {"cuboid": {"all": {"dims": [4.0, 4.0, 4.0], "pose": [0, 0, 0, 1, 0, 0, 0]}}}
    clear = {"cuboid": {"far": {"dims": [0.1, 0.1, 0.1], "pose": [10, 10, 10, 1, 0, 0, 0]}}}
    target = _goal7(base, solver.info.ee_frame)
    kwargs = dict(batch_size=512, num_solutions=1, collision="hard")
    for precision in ("fp32", "fp64"):
        solver.set_world(**worlds.from_mbm_problem(blocked))
        assert solver.solve(target, precision=precision, **kwargs).count.tolist() == [0]
        solver.set_world(**worlds.from_mbm_problem(clear))
        assert solver.solve(target, precision=precision, **kwargs).count.tolist() == [1]
        with pytest.raises(ValueError):
            solver.set_world(**worlds.from_mbm_problem({"obstacles": {"sphere": [{"radius": -1, "position": [0, 0, 0]}]}}))
        assert solver.solve(target, precision=precision, **kwargs).count.tolist() == [1]
        solver.clear_world()
        assert solver.solve(target, precision=precision, **kwargs).count.tolist() == [1]
