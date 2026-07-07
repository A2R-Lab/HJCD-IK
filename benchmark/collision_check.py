"""Analytic sphere-vs-primitive collision check (CPU, numpy) for validating a solver's IK solutions.

Used for the Table II PyRoki collision-validation column: given a robot's collision spheres placed in the
world frame at a solution q, decide whether the configuration is collision-free against a MotionBenchMaker
world (`cuboid` / `cylinder` primitives). This is solver-agnostic and dependency-light (numpy only) — the
same shared sphere model validates every solver, so the Table II column is an apples-to-apples measurement.

World dict format (from benchmark/baseline_bench.py `mb_instance_to_world_dict`):
    {"cuboid":   {name: {"dims":[dx,dy,dz], "pose":[x,y,z, qw,qx,qy,qz]}},
     "cylinder": {name: {"radius":r, "height":h, "pose":[x,y,z, qw,qx,qy,qz]}}}
Cuboid `dims` are FULL side lengths; cylinder axis is local +z, centered at the pose origin.

Signed-distance convention: distance from the sphere SURFACE to the primitive surface. >= 0 is clear,
< 0 is penetration. A config is collision-free iff every sphere clears every obstacle by `margin` (metres).

This module is a pure-CPU unit (no GPU, no cuRobo, no jax). Tests: benchmark/test_collision_check.py
(run explicitly: `pytest benchmark/test_collision_check.py` — it lives outside tests/ so it stays out of
the GPU-proof receipt, which is GPU-only).
"""
from __future__ import annotations

import numpy as np


def quat_wxyz_to_rot(q_wxyz) -> np.ndarray:
    """Rotation matrix (3x3) for a unit quaternion [w, x, y, z]. Normalizes defensively."""
    w, x, y, z = np.asarray(q_wxyz, dtype=float)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def _to_local(center, pose) -> np.ndarray:
    """Transform a world point into the primitive's local frame (pose = [x,y,z, qw,qx,qy,qz])."""
    pose = np.asarray(pose, dtype=float)
    t = pose[:3]
    R = quat_wxyz_to_rot(pose[3:7])
    return R.T @ (np.asarray(center, dtype=float) - t)


def sphere_cuboid_signed_distance(center, radius, pose, dims) -> float:
    """Signed distance from a sphere (world `center`, `radius`) to an axis-aligned-in-its-frame box.
    `dims` are full side lengths; `pose` is [x,y,z, qw,qx,qy,qz]. >= 0 clear, < 0 penetrating."""
    c = _to_local(center, pose)
    half = 0.5 * np.asarray(dims, dtype=float)
    q = np.abs(c) - half
    outside = np.linalg.norm(np.maximum(q, 0.0))         # distance when the center is outside the box
    inside = min(float(np.max(q)), 0.0)                  # negative depth when the center is inside
    return outside + inside - float(radius)


def sphere_cylinder_signed_distance(center, radius, pose, cyl_radius, height) -> float:
    """Signed distance from a sphere to a finite cylinder (axis = local +z, centered at the pose origin).
    `pose` is [x,y,z, qw,qx,qy,qz]. >= 0 clear, < 0 penetrating."""
    c = _to_local(center, pose)
    d_radial = float(np.hypot(c[0], c[1])) - float(cyl_radius)
    d_axial = abs(float(c[2])) - 0.5 * float(height)
    d = np.array([d_radial, d_axial])
    outside = np.linalg.norm(np.maximum(d, 0.0))
    inside = min(float(np.max(d)), 0.0)
    return outside + inside - float(radius)


def sphere_min_signed_distance(center, radius, world_dict) -> float:
    """Minimum signed distance from one sphere to ALL obstacles in `world_dict` (min == worst/closest).
    Returns +inf for an empty world."""
    best = np.inf
    for _name, o in (world_dict.get("cuboid") or {}).items():
        best = min(best, sphere_cuboid_signed_distance(center, radius, o["pose"], o["dims"]))
    for _name, o in (world_dict.get("cylinder") or {}).items():
        best = min(best, sphere_cylinder_signed_distance(center, radius, o["pose"],
                                                         o["radius"], o["height"]))
    return best


def config_is_collision_free(spheres_world, world_dict, margin: float = 0.0) -> bool:
    """True iff every sphere clears every obstacle by at least `margin` metres.
    `spheres_world`: (N,4) array of [x,y,z, radius] in the world frame. Empty world -> True."""
    spheres_world = np.atleast_2d(np.asarray(spheres_world, dtype=float))
    if not world_dict or not (world_dict.get("cuboid") or world_dict.get("cylinder")):
        return True
    for s in spheres_world:
        # collision-free requires every sphere to clear every obstacle by >= margin metres
        # (margin=0 -> touching allowed; matches the kernel's CC_SPHERE_MARGIN_MM=0).
        if sphere_min_signed_distance(s[:3], s[3], world_dict) < margin:
            return False
    return True
