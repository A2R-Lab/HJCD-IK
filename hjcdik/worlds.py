"""Convenience loaders that turn MotionBenchMaker problem JSON into `Solver.set_world` arguments.

These are helpers for the benchmark's dataset format, not part of the solver API. Obstacle poses are
`[x, y, z, qw, qx, qy, qz]`; `cuboid = {dims, pose}` with full side lengths; `cylinder = {radius,
height|length, pose}` (axis local +z); `sphere = {radius, pose|position}`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_TYPES = ("sphere", "cuboid", "cylinder")


def _shapes(collection):
    if isinstance(collection, list):
        return list(collection)
    if isinstance(collection, dict):
        return list(collection.values())
    raise ValueError("expected shape collection to be an array or object")


def _field(shape, key, shape_name):
    if not isinstance(shape, dict) or key not in shape:
        raise ValueError(f"{shape_name} obstacle requires '{key}'")
    return shape[key]


def _vec(values, n, what):
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (n,):
        raise ValueError(f"{what} must have exactly {'three' if n == 3 else n} values")
    if not np.isfinite(array).all():
        raise ValueError("obstacle geometry must contain finite numbers")
    return array


def _pose(values):
    pose = np.asarray(values, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError("expected pose as [x, y, z, qw, qx, qy, qz]")
    if not np.isfinite(pose).all():
        raise ValueError("obstacle geometry must contain finite numbers")
    if np.abs(pose[3:]).max() == 0:
        raise ValueError("obstacle quaternion must be non-zero")
    return pose


def _positive(value, what="obstacle dimensions, radius, and length"):
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("obstacle geometry must contain finite numbers")
    if not number > 0:
        raise ValueError(f"{what} must be positive")
    return number


def from_mbm_problem(problem: dict) -> dict:
    """`set_world` keyword arguments for one MotionBenchMaker problem instance.

    Accepts the instance (`{"obstacles": {...}, ...}`) or the bare obstacles object. A missing
    obstacles object is an error; use `{"obstacles": {}}` for an explicitly empty scene.
    Returns `boxes (M,10)`, `cylinders (M,9)` and `spheres (M,4)`; `set_world` converts cylinders.
    """
    if not isinstance(problem, dict):
        raise ValueError("collision problem must be an object")
    wrapped = "obstacles" in problem
    root = problem["obstacles"] if wrapped else problem
    if not isinstance(root, dict):
        raise ValueError("obstacles must be an object")
    if not wrapped and not root:
        raise ValueError("collision problem must specify obstacles (use {} for an empty scene)")
    for key in root:
        if key not in _TYPES:
            raise ValueError(f"unsupported obstacle type: {key}")
    spheres, boxes, cylinders = [], [], []
    for o in _shapes(root.get("sphere", [])):
        if isinstance(o, dict) and "pose" in o:
            center = _pose(o["pose"])[:3]
        else:
            center = _vec(_field(o, "position", "sphere"), 3, "obstacle vector")
        spheres.append([*center, _positive(_field(o, "radius", "sphere"))])
    for o in _shapes(root.get("cuboid", [])):
        pose = _pose(_field(o, "pose", "cuboid"))
        dims = _vec(_field(o, "dims", "cuboid"), 3, "obstacle vector")
        if not (dims > 0).all():
            raise ValueError("obstacle dimensions must be positive")
        boxes.append([*pose, *dims])
    for o in _shapes(root.get("cylinder", [])):
        pose = _pose(_field(o, "pose", "cylinder"))
        if "height" in o:
            length = _positive(o["height"])
        elif "length" in o:
            length = _positive(o["length"])
        else:
            raise ValueError("cylinder obstacle is missing height/length")
        cylinders.append([*pose, _positive(_field(o, "radius", "cylinder")), length])
    return {
        "boxes": np.asarray(boxes, dtype=np.float64).reshape(-1, 10),
        "cylinders": np.asarray(cylinders, dtype=np.float64).reshape(-1, 9),
        "spheres": np.asarray(spheres, dtype=np.float64).reshape(-1, 4),
    }


def select_problem(document: dict, set_name: str, index: int) -> dict:
    """One problem instance from a parsed MotionBenchMaker document; rejects invalid problems."""
    problems = document.get("problems")
    if not isinstance(problems, dict) or set_name not in problems:
        raise KeyError(f"problem set {set_name!r} not found")
    instances = problems[set_name]
    if not isinstance(instances, list):
        raise ValueError("problem set is not an array")
    if not 0 <= int(index) < len(instances):
        raise IndexError(f"problem index {index} out of range for {len(instances)} problems")
    instance = instances[int(index)]
    if isinstance(instance, dict) and "valid" in instance and not instance["valid"]:
        raise ValueError("collision problem is marked invalid")
    return instance


def from_mbm_text(text: str, set_name: str, index: int) -> dict:
    """`set_world` arguments for problem `index` of `set_name` in a JSON document string."""
    return from_mbm_problem(select_problem(json.loads(text), set_name, index))


def from_mbm_file(path, set_name: str, index: int) -> dict:
    """`set_world` arguments for problem `index` of `set_name` in a JSON file."""
    return from_mbm_text(Path(path).read_text(), set_name, index)


def goal_pose(problem: dict) -> np.ndarray:
    """The instance's `goal_pose` as `[x, y, z, qw, qx, qy, qz]` (a `panda_hand` pose in the dataset)."""
    gp = problem.get("goal_pose") if isinstance(problem, dict) else None
    if not gp:
        raise ValueError("problem has no goal_pose")
    position = gp.get("position_xyz") or gp.get("position")
    quaternion = gp.get("quaternion_wxyz") or gp.get("quat_wxyz")
    if position is None or quaternion is None:
        raise ValueError("goal_pose needs position_xyz and quaternion_wxyz")
    return np.asarray([*position, *quaternion], dtype=np.float64)
