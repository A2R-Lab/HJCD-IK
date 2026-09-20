from typing import Literal, Sequence, TypedDict

import numpy as np

CollisionMode = Literal["hard", "soft", "both", "auto"]


class IKResult(TypedDict):
    joint_config: np.ndarray
    pose: np.ndarray
    pos_errors: np.ndarray
    ori_errors: np.ndarray
    count: int


class BuildInfo(TypedDict):
    num_joints: int
    collision_enabled: bool


def generate_solutions(
    target_pose: Sequence[float],
    batch_size: int = 2000,
    num_solutions: int = 1,
    collision_free: bool = False,
    problems_json_text: str = "",
    problem_set_name: str = "",
    problem_idx: int = 0,
    refine_fp64: Literal[-1, 0, 1] = -1,
    write_stats: bool = False,
    collision_mode: CollisionMode = "hard",
) -> IKResult: ...


def sample_targets(
    num_targets: int,
    seed: int = 0,
) -> list[list[float]]: ...


def num_joints() -> int: ...


def collision_enabled() -> bool: ...


def build_info() -> BuildInfo: ...

