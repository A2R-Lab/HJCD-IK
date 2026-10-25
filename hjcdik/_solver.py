"""User-facing solver: `Solver`, `Result`, `TargetResult`, `Stats`, `Info`, `SolveError`.

Pure Python over the low-level `_hjcdik` bindings. The native layer solves one target per call; this
module validates whole batches before any GPU work, loops over targets under one lock, and assembles
fixed-capacity, NaN-padded host arrays that never reference device memory.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, fields
from typing import Optional, Sequence, Union

import numpy as np

from . import _hjcdik as _native   # the packaged robot; artifacts bring their own module
from ._build import Artifact, BuildError, build as _build_artifact, import_artifact, load as _load_artifact

_COLLISION_CODES = {None: 0, "hard": 1, "soft": 2, "both": 3}
_OMITTED = object()
_INT_MAX = 2**31 - 1


class SolveError(RuntimeError):
    """A CUDA or launch failure inside `Solver.solve`.

    `target_index` is the target being solved when the failure happened and `completed` the number
    of targets finished before it. No partial `Result` is returned. A context-invalidating CUDA
    error may still require restarting the process.
    """

    def __init__(self, message: str, *, target_index: int, completed: int):
        super().__init__(message)
        self.target_index = target_index
        self.completed = completed


@dataclass(frozen=True)
class Info:
    """Identity of the compiled robot and the device a `Solver` is bound to."""

    robot: str
    dof: int
    ee_frame: str
    collision_enabled: bool
    grid_header_sha256: str
    cuda_compiler_version: str
    device: int
    max_batch_size: int
    precisions: tuple = ("fp32", "fp64")
    artifact: Optional[str] = None   # store path of a runtime build; None for the packaged robot
    joint_names: tuple = ()
    joint_types: tuple = ()
    joint_units: tuple = ()
    joint_limits: tuple = ()


@dataclass(frozen=True)
class Stats:
    """Per-target diagnostics, one array of shape (T,) per field; unmeasured values are -1.

    Accuracy tallies use position error < 5 mm and orientation error < 1e-3 rad. In `hard`/`both` mode
    the collision tallies include self and environment checks; in `soft` mode they count environment
    penetration only. `env_cost_*_mm` summarise refined-candidate penetration in `soft`/`both` mode.
    """

    batch_size: np.ndarray
    n_refined: np.ndarray
    n_ik_accurate: np.ndarray
    n_coll_free_refined: np.ndarray
    n_feasible: np.ndarray
    n_ik_lost: np.ndarray
    n_coll_in_refined: np.ndarray
    n_coll_in_coarse: np.ndarray
    env_cost_min_mm: np.ndarray
    env_cost_max_mm: np.ndarray
    env_cost_mean_mm: np.ndarray
    n_returned: np.ndarray
    n_returned_ik_accurate: np.ndarray
    n_returned_coll_free: np.ndarray
    pct_returned_coll_free: np.ndarray
    n_returned_feasible: np.ndarray

    @classmethod
    def field_names(cls) -> tuple:
        return tuple(f.name for f in fields(cls))

    def __len__(self) -> int:
        return int(self.batch_size.shape[0])

    def __getitem__(self, index: int) -> dict:
        """Plain Python scalars for one target."""
        return {name: getattr(self, name)[index].item() for name in self.field_names()}


@dataclass(frozen=True)
class TargetResult:
    """Un-padded view of one target's candidates (`count` rows)."""

    q: np.ndarray            # (count, N) radians
    pose: np.ndarray         # (count, 7) metres, wxyz
    pos_err_mm: np.ndarray   # (count,)
    ori_err_rad: np.ndarray  # (count,)
    count: int
    elapsed_ms: float
    stats: Optional[dict]

    def ok(self, pos_mm: float, ori_rad: float) -> np.ndarray:
        """Pose accuracy only, never physical safety."""
        return (self.pos_err_mm < pos_mm) & (self.ori_err_rad < ori_rad)


@dataclass(frozen=True)
class Result:
    """Fixed-capacity, NaN-padded results for T targets; immutable host arrays.

    Rows `s >= count[t]` are NaN. `valid` marks the real rows. The arrays never reference device
    memory, survive later solves and the solver's destruction, and are read-only.
    """

    q: np.ndarray            # (T, S, N)
    pose: np.ndarray         # (T, S, 7)
    pos_err_mm: np.ndarray   # (T, S)
    ori_err_rad: np.ndarray  # (T, S)
    count: np.ndarray        # (T,) int64
    valid: np.ndarray        # (T, S) bool
    elapsed_ms: np.ndarray   # (T,) host-observed per-target solve time; diagnostic only
    stats: Optional[Stats]

    def __len__(self) -> int:
        return int(self.count.shape[0])

    def ok(self, pos_mm: float, ori_rad: float) -> np.ndarray:
        """Candidates inside both tolerances, (T, S) bool. Pose accuracy only, never physical safety."""
        with np.errstate(invalid="ignore"):
            return self.valid & (self.pos_err_mm < pos_mm) & (self.ori_err_rad < ori_rad)

    def __getitem__(self, index: int) -> TargetResult:
        if not isinstance(index, (int, np.integer)):
            raise TypeError("index one target at a time")
        T = len(self)
        if index < -T or index >= T:
            raise IndexError(f"target index {index} out of range for {T} targets")
        index = int(index) % T if T else 0
        n = int(self.count[index])
        return TargetResult(
            q=self.q[index, :n], pose=self.pose[index, :n],
            pos_err_mm=self.pos_err_mm[index, :n], ori_err_rad=self.ori_err_rad[index, :n],
            count=n, elapsed_ms=float(self.elapsed_ms[index]),
            stats=None if self.stats is None else self.stats[index],
        )


def _as_matrix(name: str, value, width: int) -> np.ndarray:
    """Float64 (M, width) array; None or an empty sequence is (0, width). Finite values only."""
    if value is None:
        return np.zeros((0, width), dtype=np.float64)
    array = np.array(value, dtype=np.float64, copy=True)
    if array.size == 0:
        return np.zeros((0, width), dtype=np.float64)
    if array.ndim == 1 and array.shape[0] == width:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape (M, {width}); got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    if np.abs(array).max() > np.finfo(np.float32).max:
        raise ValueError(f"{name} values must fit fp32")
    return np.ascontiguousarray(array)


def _check_quaternions(name: str, q: np.ndarray) -> None:
    if q.shape[0] and (np.abs(q).max(axis=1) == 0).any():
        raise ValueError(f"{name} quaternions must be non-zero")


def _check_positive(name: str, values: np.ndarray, what: str) -> None:
    if values.size and not (values > 0).all():
        raise ValueError(f"{name} {what} must be positive")


def _rotation_z_axis(q: np.ndarray) -> np.ndarray:
    """Third column of R(q) for rows of scalar-first quaternions (normalised here)."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)], axis=1)


def world_arrays(*, boxes=None, capsules=None, spheres=None, cylinders=None) -> dict:
    """Validate `set_world` inputs and convert cylinders to capsules.

    Returns float64 arrays `spheres (M,4)`, `capsules (M,7)`, `boxes (M,10)` in the robot base frame.
    A cylinder `[cx cy cz, qw qx qy qz, r, L]` becomes the capsule with endpoints `c +/- (L/2) z_axis`,
    a conservative approximation whose rounded ends extend `r` beyond each flat face.
    """
    spheres = _as_matrix("spheres", spheres, 4)
    _check_positive("spheres", spheres[:, 3], "radii")
    capsules = _as_matrix("capsules", capsules, 7)
    _check_positive("capsules", capsules[:, 6], "radii")
    boxes = _as_matrix("boxes", boxes, 10)
    _check_quaternions("boxes", boxes[:, 3:7])
    _check_positive("boxes", boxes[:, 7:10], "side lengths")
    cylinders = _as_matrix("cylinders", cylinders, 9)
    if cylinders.shape[0]:
        _check_quaternions("cylinders", cylinders[:, 3:7])
        _check_positive("cylinders", cylinders[:, 7:9], "radius and length")
        axis = _rotation_z_axis(cylinders[:, 3:7])
        half = 0.5 * cylinders[:, 8:9] * axis
        converted = np.concatenate([cylinders[:, :3] - half, cylinders[:, :3] + half, cylinders[:, 7:8]], axis=1)
        capsules = np.concatenate([capsules, converted], axis=0)
    return {"spheres": spheres, "capsules": np.ascontiguousarray(capsules), "boxes": boxes}


def _failed_target_index(message: str) -> int:
    """The target index the binding names in a failure message ('... target 3: ...'), else 0."""
    import re
    m = re.search(r"target (\d+):", message)
    return int(m.group(1)) if m else 0


def _readonly(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


class Solver:
    """IK solver for one compiled robot: the packaged one, or a runtime build from the artifact store.

    `Solver()` is the packaged robot. `Solver("<name>")` is the packaged robot when the name matches it,
    a shipped preset (`hjcdik.presets()`, built on first use) or a prepared build of that name
    (`hjcdik.build` / `hjcdik build`). `Solver("<path>.urdf", ee_link=..., spheres=...)` builds the
    robot on first use (`allow_build=False` only loads). One
    instance binds to a CUDA device at construction and owns one obstacle world. Calls from several
    threads are safe and serialised; they are not concurrent on the GPU.
    """

    def __init__(self, robot=None, *, ee_link: Optional[str] = None, spheres=None,
                 device: Optional[int] = None, allow_build: bool = True, cache_dir=None):
        self._artifact: Optional[Artifact] = None
        self._native = self._resolve(robot, ee_link, spheres, allow_build, cache_dir)
        info = self._native.build_info()
        import json
        joints = json.loads(self._native.joint_model_json())["joints"]
        dof = int(info["num_joints"])
        self._device = self._native.current_device() if device is None else int(device)
        self._lock = threading.RLock()
        self._world = None
        artifact = self._artifact
        self.info = Info(
            robot=info["robot"], dof=dof, ee_frame=info["ee_target"],
            collision_enabled=bool(info["collision_enabled"]),
            grid_header_sha256=info["grid_header_sha256"] or (artifact.grid_header_sha256 if artifact else ""),
            cuda_compiler_version=info["cuda_compiler_version"] or (artifact.cuda_compiler_version if artifact else ""),
            device=self._device, max_batch_size=(_INT_MAX - 255) // (16 * dof),
            artifact=None if artifact is None else str(artifact.path),
            joint_names=tuple(j["name"] for j in joints),
            joint_types=tuple(j["type"] for j in joints),
            joint_units=tuple(j["unit"] for j in joints),
            joint_limits=tuple(tuple(j["limits"]) for j in joints),
        )

    def _resolve(self, robot, ee_link, spheres, allow_build, cache_dir):
        packaged = _native.build_info()
        if robot is None or robot == packaged["robot"]:
            if ee_link is not None and ee_link != packaged["ee_target"]:
                raise ValueError(f"the packaged {packaged['robot']!r} solves the {packaged['ee_target']!r} frame, "
                                 f"not {ee_link!r}; build it from a URDF: Solver('<robot>.urdf', ee_link=...)")
            return _native
        if not isinstance(robot, (str, os.PathLike)):
            raise TypeError("robot is None, a build name, or a URDF path")
        text = str(robot)
        if text.lower().endswith(".urdf") or os.sep in text:
            if ee_link is None:
                raise ValueError("building from a URDF needs ee_link=<fixed joint of the end-effector frame>")
            if allow_build:
                self._artifact = _build_artifact(text, ee_link=ee_link, spheres=spheres, cache_dir=cache_dir)
            else:
                from ._build import default_name, _resolve_spheres
                self._artifact = _load_artifact(default_name(text, ee_link, _resolve_spheres(spheres)), cache_dir=cache_dir)
        else:
            if ee_link is not None or spheres is not None:
                raise ValueError("ee_link and spheres apply to a URDF build; a build name is already complete")
            from ._build import presets
            preset = presets().get(text)
            if preset is not None and allow_build:
                self._artifact = _build_artifact(preset["urdf"], ee_link=preset["ee_link"],
                                                 spheres=preset["spheres"], name=text, cache_dir=cache_dir)
            else:
                self._artifact = _load_artifact(text, cache_dir=cache_dir)
        return import_artifact(self._artifact)

    # ----------------------------------------------------------------------------- world
    def set_world(self, *, boxes=None, capsules=None, spheres=None, cylinders=None) -> None:
        """Replace the obstacle world (robot base frame, metres, scalar-first quaternions).

        `boxes (M,10) = [c, q_wxyz, full side lengths]`, `capsules (M,7) = [p0, p1, r]`,
        `spheres (M,4) = [c, r]`, `cylinders (M,9) = [c, q_wxyz, r, L]` (converted to capsules).
        The new world is built and uploaded before the old one is released; a failed call leaves the
        previous world intact. Calling with no primitives installs the empty world.
        """
        if not self.info.collision_enabled:
            raise ValueError("this build has no collision geometry; worlds cannot be installed")
        arrays = world_arrays(boxes=boxes, capsules=capsules, spheres=spheres, cylinders=cylinders)
        world = self._native.World(arrays["spheres"], arrays["capsules"], arrays["boxes"], self._device)
        with self._lock:
            self._world = world

    def clear_world(self) -> None:
        """Install the empty world. Self-collision is still checked under `hard` and `both`."""
        with self._lock:
            self._world = None

    @property
    def world_counts(self) -> dict:
        """Primitive counts of the installed world: spheres, capsules (cylinders included), boxes."""
        world = self._world
        if world is None:
            return {"spheres": 0, "capsules": 0, "boxes": 0}
        s, c, b = world.counts
        return {"spheres": s, "capsules": c, "boxes": b}

    # --------------------------------------------------------------------------- sampling
    def sample_targets(self, num_targets: int, seed: int = 0) -> np.ndarray:
        """Reachable poses `(num_targets, 7)` from a seeded Halton sequence over the joint limits.

        Poses are `[x, y, z, qw, qx, qy, qz]` through the compiled forward kinematics; they are not
        filtered for self or environment collision.
        """
        if int(num_targets) <= 0:
            raise ValueError("num_targets must be positive")
        if int(num_targets) > _INT_MAX - 255:
            raise ValueError("num_targets must fit CUDA launch indexing")
        with self._lock:
            poses = self._native.sample_targets(int(num_targets), int(seed), self._device)
        return np.asarray(poses, dtype=np.float64).reshape(-1, 7)

    # ------------------------------------------------------------------------------ solve
    def solve(self, poses, *, num_solutions: int = 1, batch_size: int = 2000,
              collision=_OMITTED, precision: str = "auto", stats: bool = False,
              collision_margin_m: float = 0.0, self_collision: bool = True,
              q_init=None, seed: Optional[int] = None) -> Result:
        """Solve IK for one pose `(7,)` or a batch `(T, 7)`; returns a `Result` with `T` targets.

        `batch_size` is the number of candidate configurations explored per target, not a target count.
        `num_solutions` is the fixed result capacity per target; fewer (including zero) may be valid.
        `collision` is `None`, `"hard"`, `"soft"` or `"both"`; omitted means `"hard"` on a collision
        build and `None` on an open-world build. `precision` is `"auto"` (fp64 refinement for one
        solution, fp32 otherwise), `"fp32"` or `"fp64"`; I/O is float64 either way. `stats=True`
        collects per-target diagnostics without changing the solution.

        `collision_margin_m` (metres, `hard`/`both` only) is the clearance below which a checked pair
        counts as a hit for environment pairs. Self pairs keep zero margin; zero is
        the contact-free rule. `self_collision=False` (`hard`/`both` only) skips the self pairs.
        `q_init` warm-starts the coarse search: `(K, N)` configurations shared by every target or
        `(T, K, N)` per target, `K <= batch_size`, clamped to the joint limits; the remaining
        `batch_size - K` candidates start from the seeded random initialisation. `seed` (unsigned
        32-bit; default 1337, the historical constant) shifts that initialisation. Seeds and warm
        starts change which valid candidates are found, never their validity.
        """
        poses = self._validate_poses(poses)
        num_solutions, batch_size = int(num_solutions), int(batch_size)
        if num_solutions <= 0:
            raise ValueError("num_solutions must be positive")
        if batch_size <= 0 or batch_size > self.info.max_batch_size:
            raise ValueError(f"batch_size must be in [1, {self.info.max_batch_size}] (CUDA indexing)")
        if collision is _OMITTED:
            collision = "hard" if self.info.collision_enabled else None
        if collision not in _COLLISION_CODES:
            raise ValueError("collision must be None, 'hard', 'soft' or 'both'")
        if collision is not None and not self.info.collision_enabled:
            raise ValueError("this build has no collision geometry; pass collision=None")
        if precision not in ("auto", "fp32", "fp64"):
            raise ValueError("precision must be 'auto', 'fp32' or 'fp64'")
        fp64 = (num_solutions == 1) if precision == "auto" else precision == "fp64"
        code = _COLLISION_CODES[collision]
        margin = float(collision_margin_m)
        if not np.isfinite(margin) or margin < 0:
            raise ValueError("collision_margin_m must be finite and non-negative")
        if (margin > 0 or not self_collision) and collision not in ("hard", "both"):
            raise ValueError("collision_margin_m and self_collision apply to collision='hard' or 'both'")
        if seed is None:
            seed = 1337
        if int(seed) != seed or not 0 <= int(seed) < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        inits = self._validate_q_init(q_init, len(poses), batch_size)

        with self._lock:
            world = self._world
            if collision == "soft" and (world is None or world.empty):
                raise ValueError("collision='soft' ranks by environment penetration and needs a non-empty world")
            if len(poses) == 0:
                return self._empty(num_solutions, bool(stats))
            try:
                raw = self._native.solve_batch(poses, batch_size, num_solutions, code, fp64, bool(stats), world,
                                               self._device, margin, bool(self_collision), int(seed), inits,
                                               num_solutions)
            except RuntimeError as error:
                index = _failed_target_index(str(error))
                raise SolveError(str(error), target_index=index, completed=index) from error
        return self._assemble(raw, bool(stats))

    def _empty(self, S: int, stats: bool) -> Result:
        N = self.info.dof
        empty_stats = None
        if stats:
            empty_stats = Stats(**{name: _readonly(np.zeros(0)) for name in Stats.field_names()})
        return Result(q=_readonly(np.empty((0, S, N))), pose=_readonly(np.empty((0, S, 7))),
                      pos_err_mm=_readonly(np.empty((0, S))), ori_err_rad=_readonly(np.empty((0, S))),
                      count=_readonly(np.zeros(0, dtype=np.int64)), valid=_readonly(np.zeros((0, S), dtype=bool)),
                      elapsed_ms=_readonly(np.zeros(0)), stats=empty_stats)

    def _validate_q_init(self, q_init, T: int, batch_size: int):
        """`(K, N)` shared or `(T, K, N)` per target -> one contiguous float64 array (or None)."""
        if q_init is None:
            return None
        N = self.info.dof
        array = np.array(q_init, dtype=np.float64, copy=True)
        if not ((array.ndim == 2 and array.shape[1] == N) or
                (array.ndim == 3 and array.shape[0] == T and array.shape[2] == N)):
            raise ValueError(f"q_init must have shape (K, {N}) or (T={T}, K, {N}); got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("q_init must be finite")
        K = array.shape[-2]
        if K > batch_size:
            raise ValueError(f"q_init has {K} rows but batch_size is {batch_size}; K must not exceed batch_size")
        return np.ascontiguousarray(array)

    def _validate_poses(self, poses) -> np.ndarray:
        array = np.array(poses, dtype=np.float64, copy=True)
        if array.ndim == 1:
            if array.shape != (7,):
                raise ValueError(f"a pose is [x, y, z, qw, qx, qy, qz]; got shape {array.shape}")
            array = array[None, :]
        if array.ndim != 2 or array.shape[1] != 7:
            raise ValueError(f"poses must have shape (7,) or (T, 7); got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("target poses must be finite")
        if array.shape[0] and (np.abs(array[:, 3:]).max(axis=1) == 0).any():
            raise ValueError("target quaternions must be non-zero")
        if array.shape[0] and np.abs(array[:, :3]).max() > np.finfo(np.float32).max:
            raise ValueError("target positions exceed fp32 range")
        return array

    def _assemble(self, raw: dict, stats: bool) -> Result:
        """Wrap the binding's (T, S, ...) arrays read-only."""
        count = raw["count"]
        S = raw["joint_config"].shape[1]
        valid = np.arange(S)[None, :] < count[:, None]
        result_stats = None
        if stats:
            result_stats = Stats(**{name: _readonly(raw["stats"][name]) for name in Stats.field_names()})
        return Result(q=_readonly(raw["joint_config"]), pose=_readonly(raw["pose"]),
                      pos_err_mm=_readonly(raw["pos_errors"]), ori_err_rad=_readonly(raw["ori_errors"]),
                      count=_readonly(count), valid=_readonly(valid), elapsed_ms=_readonly(raw["elapsed_ms"]),
                      stats=result_stats)
