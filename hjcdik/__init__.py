"""HJCD-IK: GPU-parallel inverse kinematics for one compiled robot.

    import hjcdik
    solver = hjcdik.Solver()
    targets = solver.sample_targets(10, seed=0)
    result = solver.solve(targets, num_solutions=4)
"""
import os

if os.name == "nt":
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        cuda_bin = os.path.join(cuda_path, "bin")
        if os.path.isdir(cuda_bin):
            os.add_dll_directory(cuda_bin)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_bin = os.path.join(conda_prefix, "Library", "bin")
        if os.path.isdir(conda_bin):
            os.add_dll_directory(conda_bin)

from ._hjcdik import build_info
from ._build import Artifact, BuildError, build, list_builds, load, sphere_presets
from ._solver import Info, Result, SolveError, Solver, Stats, TargetResult, world_arrays
from . import worlds

__all__ = [
    "Solver",
    "Result",
    "TargetResult",
    "Stats",
    "Info",
    "SolveError",
    "BuildError",
    "Artifact",
    "build",
    "load",
    "list_builds",
    "sphere_presets",
    "build_info",
    "world_arrays",
    "worlds",
]
