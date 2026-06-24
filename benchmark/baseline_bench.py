"""HJCD-IK competitor baselines harness — PyRoki and cuRobo.

Run with `--mode pyroki` or `--mode curobo`. This is the baseline counterpart to
`benchmark/hjcd_ik_bench.py` (which benchmarks HJCD-IK itself); it lives behind the optional
`baselines` install extra (PyRoki / cuRobo / JAX / torch are NOT core deps). See docs/source/user_guide/benchmarks/results.rst.

The paper's "Batch" axis maps to:  HJCD-IK batch_size  ==  cuRobo num_seeds  ==  PyRoki num_seeds_init.

Collision-free targets come from a MotionBenchMaker problem set; set the MB_JSON_PATH env var to point
at it (defaults to the repo's tests/mb_problems.json).
"""
from __future__ import annotations  # lazy annotations: cuRobo types in signatures stay strings when cuRobo absent

# Disable JAX prealloc etc
import os
import gc
import tempfile
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Standard Library
import argparse
import time

# Third Party
import numpy as np
import torch

# cuRobo (optional, v2 API): only needed for --mode curobo and the PyRoki collision-free collision check.
# Imported lazily so PyRoki open-world / DoF runs work without cuRobo. The harness targets the cuRobo v2
# API (curobo>=0.8 / main — the classic curobo.wrap.reacher.ik_solver path is gone); see docs/source/user_guide/benchmarks/results.rst.
# v2 runtime needs a kernel backend: `pip install 'cuda-core[cu13]'` (no compile) — handled by the installer.
# _HAS_CUROBO gates every cuRobo-dependent path.
try:
    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    from curobo.types import DeviceCfg, GoalToolPose, Pose
    from curobo.scene import Cuboid, Cylinder, Scene
    from curobo.robot_builder import RobotBuilder
    _HAS_CUROBO = True
except ImportError as _curobo_err:
    _HAS_CUROBO = False
    _CUROBO_IMPORT_ERR = _curobo_err


def setup_curobo_logger(level="error"):
    """Quiet cuRobo logging (v2-safe; no-op if the logging hook moved)."""
    try:
        from curobo._src.util.logging import setup_logger
        setup_logger(level)
    except Exception:
        pass

# set seeds
torch.manual_seed(2)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


import xml.etree.ElementTree as ET
from io import StringIO

import jax
import jaxlie
import jaxls
import pyroki as pk
import yourdfpy
from jax import lax
from jax import numpy as jnp
from robot_descriptions.loaders.yourdfpy import load_robot_description


import json

import math

from functools import partial

MB_JSON_PATH = os.environ.get(
    "MB_JSON_PATH",
    str(Path(__file__).resolve().parents[1] / "tests" / "mb_problems.json"),
)

def make_batched_pyroki_ik_many(num_seeds_init: int, k: int):
    solve_fn = partial(ik_beam.solve_ik_many, num_seeds_init=num_seeds_init, k=k)
    return jax.jit(jax.vmap(solve_fn))

def quat_mul_wxyz(a, b):
    """(a ⊗ b) with wxyz quats."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dtype=np.float32)

def quat_norm_wxyz(q):
    q = np.asarray(q, dtype=np.float32)
    n = float(np.linalg.norm(q))
    if n > 0.0:
        return q / n
    return np.array([1, 0, 0, 0], dtype=np.float32)

def rotate_quat_90deg_local_y_wxyz(q_wxyz, sign=+1):
    """
    Local rotation: q' = q ⊗ d
    d = [cos(45°), 0, sin(45°)*sign, 0] in wxyz
    """
    s = np.float32(0.7071067811865476)
    d = np.array([s, 0.0, np.float32(sign)*s, 0.0], dtype=np.float32)
    return quat_norm_wxyz(quat_mul_wxyz(np.asarray(q_wxyz, dtype=np.float32), d))

def _cylinders_list(inst: dict):
    cyl = inst.get("obstacles", {}).get("cylinder", {})
    if isinstance(cyl, dict):
        return [(k, v) for k, v in cyl.items() if isinstance(v, dict) and "pose" in v]
    if isinstance(cyl, list):
        out = []
        for i, v in enumerate(cyl):
            if isinstance(v, dict) and "pose" in v:
                out.append((f"cylinder{i}", v))
        return out
    return []

def associate_goal_to_closest_cylinder(inst: dict, eps: float = 1e-4):

    gp = inst["goal_pose"]
    gx, gy, gz = gp["position_xyz"]

    cyls = _cylinders_list(inst)
    if not cyls:
        raise RuntimeError("No cylinders found.")

    best = None
    best_key = None
    for name, c in cyls:
        pose = c["pose"]  # [x,y,z,qw,qx,qy,qz] wxyz
        cx, cy, cz = float(pose[0]), float(pose[1]), float(pose[2])
        dx, dy, dz = gx - cx, gy - cy, gz - cz
        dist2 = dx*dx + dy*dy + dz*dz

        match_axes = 0
        if abs(dx) <= eps: match_axes += 1
        if abs(dy) <= eps: match_axes += 1
        if abs(dz) <= eps: match_axes += 1

        key = (-match_axes, dist2)
        if best_key is None or key < best_key:
            best_key = key
            best = c

    return best

def mb_instance_to_cylinder_goal(inst: dict, eps: float = 1e-4, rot_sign: int = +1):
    cyl = associate_goal_to_closest_cylinder(inst, eps=eps)

    cyl_pose = cyl["pose"]  # [x,y,z,qw,qx,qy,qz] wxyz
    cx, cy = float(cyl_pose[0]), float(cyl_pose[1])

    gp = inst["goal_pose"]
    gz = float(gp["position_xyz"][2])

    pos = np.array([cx, cy, gz], dtype=np.float32)
    wxyz = np.array(gp["quaternion_wxyz"], dtype=np.float32)

    return wxyz, pos

def load_mb_problem_set(path: str, problem_set: str):
    with open(path, "r") as f:
        D = json.load(f)
    return D["problems"][problem_set]

def mb_instance_to_world_dict(inst: dict) -> dict:
    world = {"cuboid": {}, "cylinder": {}}

    obs = inst.get("obstacles", {})

    # Cuboids
    cub = obs.get("cuboid", {})
    for name, o in cub.items():
        world["cuboid"][name] = {
            "dims": o["dims"],      # [x,y,z]
            "pose": o["pose"],      # [x,y,z,qw,qx,qy,qz]
        }

    # Cylinders (mb_problems.json stores radius/height/pose)
    cyl = obs.get("cylinder", {})
    for name, o in cyl.items():
        world["cylinder"][name] = {
            "radius": o["radius"],
            "height": o["height"],
            "pose": o["pose"],      # [x,y,z,qw,qx,qy,qz]
        }

    return world

def mb_instance_to_goal(inst: dict):
    gp = inst["goal_pose"]
    pos = np.array(gp["position_xyz"], dtype=np.float32)
    wxyz = np.array(gp["quaternion_wxyz"], dtype=np.float32)
    return wxyz, pos


def _collision_free_or_unknown(robot_file: str, world_dict: dict, q_torch) -> "bool | None":
    """cuRobo-based collision check, or None when it can't be computed (collision-freeness unknown).
    Lets PyRoki collision-free IK still report timing/accuracy; the None propagates to a blank
    'collision_free' so Table II stays honest about what was measured. Returns None if cuRobo is absent
    OR the (v2) external sphere-collision query isn't wired (see check_collision_free_curobo)."""
    if not _HAS_CUROBO:
        return None
    try:
        return bool(check_collision_free_curobo(robot_file, world_dict, q_torch)[0])
    except NotImplementedError:
        return None


def check_collision_free_curobo(robot_file: str, world_dict: dict, q) -> np.ndarray:
    """Mask (N,) where True == collision-free, for externally-supplied joint configs q against world_dict.

    cuRobo v2 follow-up: this validates *another* solver's q (PyRoki's) against the scene, which needs the
    robot's collision spheres at q (kinematics) fed into `scene_collision_checker.get_sphere_distance`. The
    v2 sphere-export path is not yet wired here, so we raise NotImplementedError and the caller degrades to
    'unknown' (blank Table II PyRoki collision column). cuRobo's OWN collision-free IK (Table II --mode
    curobo) is fully supported — it uses the in-optimizer collision cost (self_collision_check + scene)."""
    raise NotImplementedError(
        "cuRobo-v2 external sphere-collision check (for the PyRoki Table II column) not wired yet; "
        "cuRobo's own collision-free IK is supported. See docs/source/user_guide/benchmarks/results.rst."
    )

def benchmark_pyroki_on_mb_problems(
    robot_file: str,
    problem_set: str,
    num_instances: int,
    use_self_collision: bool = False,
):
    instances = load_mb_problem_set(MB_JSON_PATH, problem_set)[:num_instances]

    # thresholds
    position_threshold = 0.005
    rotation_threshold = 0.05

    rows = []
    for idx, inst in enumerate(instances):
        world_dict = mb_instance_to_world_dict(inst)
        target_wxyz, target_pos = mb_instance_to_goal(inst)

        # PyRoki batched inputs
        target_wxyz_jax = jnp.asarray(target_wxyz[None, :])
        target_pos_jax  = jnp.asarray(target_pos[None, :])

        # JIT warmup once
        jax.block_until_ready(batched_ik(target_wxyz_jax, target_pos_jax))

        t0 = time.time()
        sol_jax = batched_ik(target_wxyz_jax, target_pos_jax)
        jax.block_until_ready(sol_jax)
        dt = (time.time() - t0)

        # FK and errors
        fk = batched_fk(sol_jax)
        fk_wxyz = np.array(fk[0, 0:4])
        fk_pos  = np.array(fk[0, 4:7])

        pos_err = np.linalg.norm(fk_pos - target_pos)
        ori_err = np.linalg.norm(np.array((jaxlie.SO3(target_wxyz).inverse() @ jaxlie.SO3(fk_wxyz)).log()))

        pose_success = (pos_err < position_threshold) and (ori_err < rotation_threshold)

        # collision-check PyRoki's joint solution with cuRobo RobotWorld (None if cuRobo absent)
        q_torch = torch.from_numpy(np.array(sol_jax)).to("cuda")
        collision_free = _collision_free_or_unknown(robot_file, world_dict, q_torch)

        rows.append({
            "instance": idx,
            "time_ms": dt * 1000.0,
            "pos_err_m": float(pos_err),
            "ori_err_rad": float(ori_err),
            "pose_success": bool(pose_success),
            "collision_free": (None if collision_free is None else bool(collision_free)),
            "success_both": (None if collision_free is None else bool(pose_success and collision_free)),
        })

    return rows



def newton_raphson(f, x, iters):
    """Use the Newton-Raphson method to find a root of the given function."""

    def update(x, _):
        y = x - f(x) / jax.grad(f)(x)
        return y, None

    x, _ = lax.scan(update, 1.0, length=iters)
    return x


def roberts_sequence(num_points, dim, root):
    # From https://gist.github.com/carlosgmartin/1fd4e60bed526ec8ae076137ded6ebab.
    basis = 1 - (1 / root ** (1 + jnp.arange(dim)))

    n = jnp.arange(num_points)
    x = n[:, None] * basis[None, :]
    x, _ = jnp.modf(x)

    return x


class PyrokiIkBeamHelper:
    def __init__(self, ee_link_name: str = "panda_hand_tcp", urdf_path: str | None = None):
        # urdf_path != None -> load a custom URDF (e.g. the chained DoF variants for Table III);
        # default = the built-in 7-DoF panda (prismatic fingers fixed, as the original harness did).
        if urdf_path is not None:
            urdf = yourdfpy.URDF.load(urdf_path)
        else:
            urdf = load_robot_description("panda_description")
            xml_tree = urdf.write_xml()
            for joint in xml_tree.findall('.//joint[@type="prismatic"]'):
                joint.set("type", "fixed")
                for tag in ("axis", "limit", "dynamics"):
                    child = joint.find(tag)
                    if child is not None:
                        joint.remove(child)
            urdf = yourdfpy.URDF.load(StringIO(ET.tostring(xml_tree.getroot(), encoding="unicode")))
        assert urdf.validate()

        # yourdfpy => pyroki
        robot = pk.Robot.from_urdf(urdf)
        target_link_index = jnp.array(robot.links.names.index(ee_link_name))

        self.robot = robot
        exp = robot.joints.num_actuated_joints
        self.root = newton_raphson(lambda x: x ** (exp + 1) - x - 1, 1.0, 10_000)
        self.target_link_index = target_link_index

    def solve_ik(self, target_wxyz: jax.Array, target_position: jax.Array) -> jax.Array:
        num_seeds_init: int = 64
        num_seeds_final: int = 4

        total_steps: int = 16
        init_steps: int = 6

        def solve_one(
            initial_q: jax.Array, lambda_initial: float | jax.Array, max_iters: int
        ) -> tuple[jax.Array, jaxls.SolveSummary]:
            """Solve IK problem with a single initial condition. We'll vmap
            over initial_q to solve problems in parallel."""
            joint_var = robot.joint_var_cls(0)
            factors = [
                # pk.costs.pose_cost(
                pk.costs.pose_cost_analytic_jac(
                    robot,
                    joint_var,
                    jaxlie.SE3.from_rotation_and_translation(
                        jaxlie.SO3(target_wxyz), target_position
                    ),
                    self.target_link_index,
                    pos_weight=10.0,
                    ori_weight=5.0,
                ),
                pk.costs.limit_cost(
                    robot,
                    joint_var,
                    weight=50.0,
                ),
            ]
            sol, summary = (
                jaxls.LeastSquaresProblem(factors, [joint_var])
                .analyze()
                .solve(
                    initial_vals=jaxls.VarValues.make(
                        [joint_var.with_value(initial_q)]
                    ),
                    verbose=False,
                    linear_solver="dense_cholesky",
                    termination=jaxls.TerminationConfig(
                        max_iterations=max_iters,
                        early_termination=False,
                    ),
                    trust_region=jaxls.TrustRegionConfig(lambda_initial=lambda_initial),
                    return_summary=True,
                )
            )
            return sol[joint_var], summary

        vmapped_solve = jax.vmap(solve_one, in_axes=(0, 0, None))

        # Create initial seeds, but this time with quasi-random sequence.
        robot = self.robot
        initial_qs = robot.joints.lower_limits + roberts_sequence(
            num_seeds_init, robot.joints.num_actuated_joints, self.root
        ) * (robot.joints.upper_limits - robot.joints.lower_limits)

        # Optimize the initial seeds.
        initial_sols, summary = vmapped_solve(
            initial_qs, jnp.full(initial_qs.shape[:1], 10.0), init_steps
        )

        # Get the best initial solutions.
        best_initial_sols = jnp.argsort(
            summary.cost_history[jnp.arange(num_seeds_init), -1]
        )[:num_seeds_final]

        # Optimize more for the best initial solutions.
        best_sols, summary = vmapped_solve(
            initial_sols[best_initial_sols],
            summary.lambda_history[jnp.arange(num_seeds_init), -1][best_initial_sols],
            total_steps - init_steps,
        )
        return best_sols[
            jnp.argmin(
                summary.cost_history[jnp.arange(num_seeds_final), summary.iterations]
            )
        ]

    def solve_ik_param(
        self,
        target_wxyz: jax.Array,
        target_position: jax.Array,
        num_seeds_init: int,
        num_seeds_final: int = 4,
        total_steps: int = 16,
        init_steps: int = 6,
    ) -> jax.Array:
        robot = self.robot  

        def solve_one(
            initial_q: jax.Array, lambda_initial: float | jax.Array, max_iters: int
        ):
            joint_var = robot.joint_var_cls(0)
            factors = [
                pk.costs.pose_cost_analytic_jac(
                    robot,
                    joint_var,
                    jaxlie.SE3.from_rotation_and_translation(
                        jaxlie.SO3(target_wxyz), target_position
                    ),
                    self.target_link_index,
                    pos_weight=10.0,
                    ori_weight=5.0,
                ),
                pk.costs.limit_cost(robot, joint_var, weight=50.0),
            ]
            sol, summary = (
                jaxls.LeastSquaresProblem(factors, [joint_var])
                .analyze()
                .solve(
                    initial_vals=jaxls.VarValues.make([joint_var.with_value(initial_q)]),
                    verbose=False,
                    linear_solver="dense_cholesky",
                    termination=jaxls.TerminationConfig(
                        max_iterations=max_iters,
                        early_termination=False,
                    ),
                    trust_region=jaxls.TrustRegionConfig(lambda_initial=lambda_initial),
                    return_summary=True,
                )
            )
            return sol[joint_var], summary

        vmapped_solve = jax.vmap(solve_one, in_axes=(0, 0, None))

        # seeds: shape depends on num_seeds_init
        initial_qs = robot.joints.lower_limits + roberts_sequence(
            num_seeds_init, robot.joints.num_actuated_joints, self.root
        ) * (robot.joints.upper_limits - robot.joints.lower_limits)

        initial_sols, summary = vmapped_solve(
            initial_qs, jnp.full((num_seeds_init,), 10.0), init_steps
        )

        best_initial = jnp.argsort(summary.cost_history[jnp.arange(num_seeds_init), -1])[
            :num_seeds_final
        ]

        best_sols, summary2 = vmapped_solve(
            initial_sols[best_initial],
            summary.lambda_history[jnp.arange(num_seeds_init), -1][best_initial],
            total_steps - init_steps,
        )

        return best_sols[
            jnp.argmin(summary2.cost_history[jnp.arange(num_seeds_final), summary2.iterations])
        ]
    
    def solve_ik_many(
        self,
        target_wxyz: jax.Array,
        target_position: jax.Array,
        num_seeds_init: int,
        k: int = 4,
        num_seeds_final: int = 4,
        total_steps: int = 16,
        init_steps: int = 6,
    ) -> jax.Array:
        """
        Returns up to k candidate joint solutions (k, dof), sorted by final cost (best first).
        """
        robot = self.robot

        def solve_one(initial_q: jax.Array, lambda_initial: float | jax.Array, max_iters: int):
            joint_var = robot.joint_var_cls(0)
            factors = [
                pk.costs.pose_cost_analytic_jac(
                    robot,
                    joint_var,
                    jaxlie.SE3.from_rotation_and_translation(
                        jaxlie.SO3(target_wxyz), target_position
                    ),
                    self.target_link_index,
                    pos_weight=10.0,
                    ori_weight=5.0,
                ),
                pk.costs.limit_cost(robot, joint_var, weight=50.0),
            ]
            sol, summary = (
                jaxls.LeastSquaresProblem(factors, [joint_var])
                .analyze()
                .solve(
                    initial_vals=jaxls.VarValues.make([joint_var.with_value(initial_q)]),
                    verbose=False,
                    linear_solver="dense_cholesky",
                    termination=jaxls.TerminationConfig(
                        max_iterations=max_iters,
                        early_termination=False,
                    ),
                    trust_region=jaxls.TrustRegionConfig(lambda_initial=lambda_initial),
                    return_summary=True,
                )
            )
            return sol[joint_var], summary

        vmapped_solve = jax.vmap(solve_one, in_axes=(0, 0, None))

        # seeds (num_seeds_init, dof)
        initial_qs = robot.joints.lower_limits + roberts_sequence(
            num_seeds_init, robot.joints.num_actuated_joints, self.root
        ) * (robot.joints.upper_limits - robot.joints.lower_limits)

        # initial pass
        initial_sols, summary = vmapped_solve(
            initial_qs, jnp.full((num_seeds_init,), 10.0), init_steps
        )
        init_cost = summary.cost_history[jnp.arange(num_seeds_init), -1]

        # keep best num_seeds_final
        keep = jnp.argsort(init_cost)[: min(num_seeds_final, num_seeds_init)]

        # refine
        refined_sols, summary2 = vmapped_solve(
            initial_sols[keep],
            summary.lambda_history[jnp.arange(num_seeds_init), -1][keep],
            total_steps - init_steps,
        )

        # final costs at termination iter
        idx = jnp.arange(refined_sols.shape[0])
        final_cost = summary2.cost_history[idx, summary2.iterations]

        order = jnp.argsort(final_cost)
        out = refined_sols[order]
        return out[: min(k, out.shape[0])]

    def forward_kinematics(self, q: jax.Array | np.ndarray) -> jax.Array:
        return self.robot.forward_kinematics(jnp.asarray(q))[self.target_link_index]


# Batched helpers for IK and FK.
ik_beam = PyrokiIkBeamHelper("panda_hand_tcp")
ik_beam_hand = PyrokiIkBeamHelper("panda_hand")
batched_ik = jax.jit(jax.vmap(ik_beam.solve_ik))
batched_fk = jax.jit(jax.vmap(ik_beam.forward_kinematics))
batched_fk_hand = jax.jit(jax.vmap(ik_beam_hand.forward_kinematics))

from functools import partial

def make_batched_pyroki_ik(num_seeds_init: int, beam: PyrokiIkBeamHelper = None):
    beam = beam or ik_beam
    solve_fn = partial(beam.solve_ik_param, num_seeds_init=num_seeds_init)
    return jax.jit(jax.vmap(solve_fn))
    
    
def eval_one_mb_instance(robot_file: str, inst: dict, batched_ik_fn):
    # thresholds
    position_threshold = 0.005
    rotation_threshold = 0.05

    world_dict = mb_instance_to_world_dict(inst)
    #target_wxyz, target_pos = mb_instance_to_goal(inst)
    target_wxyz, target_pos = mb_instance_to_cylinder_goal(inst, eps=1e-4, rot_sign=+1)

    target_wxyz_jax = jnp.asarray(target_wxyz[None, :])
    target_pos_jax  = jnp.asarray(target_pos[None, :])

    # warm compile seed count
    jax.block_until_ready(batched_ik_fn(target_wxyz_jax, target_pos_jax))

    t0 = time.time()
    sol_jax = batched_ik_fn(target_wxyz_jax, target_pos_jax)
    jax.block_until_ready(sol_jax)
    dt_ms = (time.time() - t0) * 1000.0

    fk = batched_fk(sol_jax)
    fk_wxyz = np.array(fk[0, 0:4])
    fk_pos  = np.array(fk[0, 4:7])

    pos_err = float(np.linalg.norm(fk_pos - target_pos))
    ori_err = float(np.linalg.norm(np.array(
        (jaxlie.SO3(target_wxyz).inverse() @ jaxlie.SO3(fk_wxyz)).log()
    )))

    pose_success = (pos_err < position_threshold) and (ori_err < rotation_threshold)

    # collision check with cuRobo (None if cuRobo absent)
    q_torch = torch.from_numpy(np.array(sol_jax)).to("cuda")
    collision_free = _collision_free_or_unknown(robot_file, world_dict, q_torch)

    return dt_ms, pos_err, ori_err, pose_success, collision_free

def print_one_solution(robot_file: str, inst: dict, batched_ik_fn, idx: int):
    world_dict = mb_instance_to_world_dict(inst)
    #target_wxyz, target_pos = mb_instance_to_goal(inst)
    target_wxyz, target_pos = mb_instance_to_cylinder_goal(inst, eps=1e-4, rot_sign=+1)

    target_wxyz_jax = jnp.asarray(target_wxyz[None, :])
    target_pos_jax  = jnp.asarray(target_pos[None, :])

    # compile/warm
    jax.block_until_ready(batched_ik_fn(target_wxyz_jax, target_pos_jax))

    t0 = time.time()
    sol_jax = batched_ik_fn(target_wxyz_jax, target_pos_jax)
    jax.block_until_ready(sol_jax)
    dt_ms = (time.time() - t0) * 1000.0

    sol_np = np.array(sol_jax[0])

    fk = batched_fk(sol_jax)
    fk_wxyz = np.array(fk[0, 0:4])
    fk_pos  = np.array(fk[0, 4:7])

    pos_err = float(np.linalg.norm(fk_pos - target_pos))
    ori_err = float(np.linalg.norm(np.array(
        (jaxlie.SO3(target_wxyz).inverse() @ jaxlie.SO3(fk_wxyz)).log()
    )))

    q_torch = torch.from_numpy(np.array(sol_jax)).to("cuda")
    collision_free = _collision_free_or_unknown(robot_file, world_dict, q_torch)

    print("\n==== PyRoki single-solution dump ====")
    print(f"problem_idx: {idx}")
    print(f"time_ms: {dt_ms:.4f}")
    print(f"target_pos:  {target_pos.tolist()}")
    print(f"target_wxyz: {target_wxyz.tolist()}")
    print(f"q_solution:  {sol_np.tolist()}")
    print(f"fk_pos:      {fk_pos.tolist()}")
    print(f"fk_wxyz:     {fk_wxyz.tolist()}")
    print(f"pos_err_m:   {pos_err:.6e}")
    print(f"ori_err_rad: {ori_err:.6e}")
    print(f"collision_free: {collision_free}")
    print("====================================\n")


# ======================== cuRobo benchmark helpers ========================

def load_goal_yaml(goal_file: str) -> np.ndarray:
    """Load an ik_dataset.yml → [N,7] float32 as [x,y,z,qw,qx,qy,qz] (wxyz)."""
    import yaml
    with open(goal_file, "r") as f:
        goal_dict = yaml.safe_load(f)
    goals = [g["position"] + g["quaternion"] for g in goal_dict["goals"]]
    return np.array(goals, dtype=np.float32)


def sample_fk_goals_with_curobo(ik_solver, num_goals: int) -> np.ndarray:
    """Sample joint configs, run FK, return [N,7] as [x,y,z,qw,qx,qy,qz] (wxyz). (cuRobo v2.)"""
    q_sample = ik_solver.sample_configs(num_goals)
    while q_sample.shape[0] == 0:
        q_sample = ik_solver.sample_configs(num_goals)
    tool = ik_solver.tool_frames[0]
    pose = ik_solver.kinematics.get_link_poses(q_sample, [tool])   # Pose: pos (N,1,3), quat (N,1,4) wxyz
    pos_np  = pose.position.reshape(-1, 3).detach().cpu().numpy()
    quat_np = pose.quaternion.reshape(-1, 4).detach().cpu().numpy()
    return np.concatenate([pos_np, quat_np], axis=1)


def sample_fk_goals_pyroki(num_goals: int) -> np.ndarray:
    robot = ik_beam.robot
    lower = np.array(robot.joints.lower_limits)
    upper = np.array(robot.joints.upper_limits)
    dof = int(robot.joints.num_actuated_joints)
    q_samples = (lower + np.random.rand(num_goals, dof) * (upper - lower)).astype(np.float32)
    fk_out = np.array(batched_fk(jnp.asarray(q_samples)))
    return np.concatenate([fk_out[:, 4:7], fk_out[:, 0:4]], axis=1)  # [x,y,z, qw,qx,qy,qz]


def eval_pyroki_on_goal7(goal7: np.ndarray, batched_ik_fn, batched_fk_fn=None) -> tuple:
    position_threshold = 0.005
    rotation_threshold = 0.05
    batched_fk_fn = batched_fk_fn or batched_fk
    target_pos  = goal7[0:3]
    target_wxyz = goal7[3:7]
    target_wxyz_jax = jnp.asarray(target_wxyz[None, :])
    target_pos_jax  = jnp.asarray(target_pos[None, :])
    t0 = time.time()
    sol_jax = batched_ik_fn(target_wxyz_jax, target_pos_jax)
    jax.block_until_ready(sol_jax)
    dt_ms = (time.time() - t0) * 1000.0
    fk = batched_fk_fn(sol_jax)
    fk_wxyz = np.array(fk[0, 0:4])
    fk_pos  = np.array(fk[0, 4:7])
    pos_err = float(np.linalg.norm(fk_pos - target_pos))
    ori_err = float(np.linalg.norm(np.array(
        (jaxlie.SO3(jnp.asarray(target_wxyz)).inverse() @ jaxlie.SO3(jnp.asarray(fk_wxyz))).log()
    )))
    return dt_ms, pos_err, ori_err, (pos_err < position_threshold) and (ori_err < rotation_threshold)


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.replace(" ", ",").split(",") if x.strip()]


def convert_numpy_scalars(data):
    if isinstance(data, dict):
        return {k: convert_numpy_scalars(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_numpy_scalars(x) for x in data]
    if isinstance(data, (np.generic, jnp.ndarray)):
        return data.item() if np.isscalar(data) else data.tolist()
    return data


def _write_yaml_safe(data, path):
    """Dump the aux .yml results without cuRobo's write_yaml. Falls back to .json if PyYAML is absent.
    (The canonical CSV is the artifact make_tables.py consumes; this .yml is just a convenience copy.)"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        import yaml
        with open(path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False)
    except Exception:
        import json
        jpath = (path[:-4] + ".json") if path.endswith(".yml") else (path + ".json")
        with open(jpath, "w") as f:
            json.dump(data, f, indent=2)


# ---- cuRobo v2 plumbing (replaces the classic curobo.wrap.reacher.ik_solver path) ----
# The benchmark always solves ONE target pose per call (B=1) with many seeds; we keep that contract.
# `tensor_args` is a v2 DeviceCfg (has .device / .dtype) so the warmup/run helpers stay unchanged.

def _world_dict_to_scene(world_dict):
    """MB/world dict ({'cuboid': {name:{dims,pose}}, 'cylinder': {name:{radius,height,pose}}}) -> v2 Scene.
    Poses are [x,y,z, qw,qx,qy,qz]. Returns None for an empty world."""
    if not world_dict:
        return None
    cuboids, cylinders = [], []
    for name, o in (world_dict.get("cuboid") or {}).items():
        cuboids.append(Cuboid(name=str(name), pose=list(o["pose"]), dims=list(o["dims"])))
    for name, o in (world_dict.get("cylinder") or {}).items():
        cylinders.append(Cylinder(name=str(name), pose=list(o["pose"]),
                                  radius=float(o["radius"]), height=float(o["height"])))
    if not cuboids and not cylinders:
        return None
    return Scene(cuboid=cuboids or None, cylinder=cylinders or None)


def _subtree_urdf(urdf_path, base_link):
    """Return a URDF path whose kinematic root is `base_link` (trim links/joints outside its subtree).

    cuRobo v2's RobotBuilder builds from the full URDF tree; restricting to base_link's subtree keeps the
    DoF count fair (e.g. Fetch arm rooted at arm_mount_link, not the mobile base). No-op (returns the input
    path) when base_link is already the URDF root."""
    import tempfile
    import xml.etree.ElementTree as ET
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    children = {}
    for j in root.findall("joint"):
        children.setdefault(j.find("parent").get("link"), []).append(j)
    keep_links, keep_joints, stack = {base_link}, [], [base_link]
    while stack:
        for j in children.get(stack.pop(), []):
            c = j.find("child").get("link")
            keep_joints.append(j); keep_links.add(c); stack.append(c)
    all_links = {l.get("name") for l in root.findall("link")}
    if keep_links == all_links:
        return urdf_path                          # base_link already roots the whole tree
    new_root = ET.Element("robot", root.attrib)
    for m in root.findall("material"):
        new_root.append(m)
    for l in root.findall("link"):
        if l.get("name") in keep_links:
            new_root.append(l)
    keep_ids = {id(j) for j in keep_joints}
    for j in root.findall("joint"):
        if id(j) in keep_ids:
            new_root.append(j)
    fd, out = tempfile.mkstemp(prefix="curobo_subtree_", suffix=".urdf")
    os.close(fd)
    ET.ElementTree(new_root).write(out)
    return out


def _build_ik_solver(robot, *, scene, num_seeds, position_threshold, use_cuda_graph, self_collision):
    """Create a v2 InverseKinematics solver. `robot` is a v2 robot spec (yml name / dict / built-robot
    yml path). Returns (ik_solver, device_cfg)."""
    tensor_args = DeviceCfg()
    kwargs = dict(
        robot=robot,
        num_seeds=num_seeds,
        position_tolerance=position_threshold,
        use_cuda_graph=use_cuda_graph,
        self_collision_check=self_collision,
        max_batch_size=1,                          # benchmark solves one pose per call
    )
    if scene is not None:
        kwargs["scene_model"] = scene
        kwargs["collision_cache"] = {"cuboid": 64, "cylinder": 64, "mesh": 16}
    return InverseKinematics(InverseKinematicsCfg.create(**kwargs)), tensor_args


def make_curobo_solver_from_world_dict(
    robot_file: str,
    world_dict: dict,
    collision_free: bool,
    high_precision: bool,
    use_cuda_graph: bool,
    num_seeds: int,
    robot_urdf: str = "",
    base_link: str = "panda_link0",
    ee_link: str = "",
):
    """cuRobo v2 IK solver for a bundled robot config (default franka.yml == Panda). When collision_free,
    the MB obstacles (world_dict) are loaded as a v2 Scene and the optimizer enforces collision avoidance.

    When `robot_urdf`+`ee_link` are given, the robot is built from that URDF with that tool frame instead of
    the bundled `robot_file`. This is required for the collision-free (Table II) MotionBenchMaker problems,
    which are posed in the `panda_grasptarget` frame — franka.yml's bundled tool is `panda_hand` (~10 cm
    offset), which otherwise yields a constant ~100 mm error at 0% success."""
    scene = _world_dict_to_scene(world_dict) if collision_free else None
    robot = _curobo_robot_yml_from_urdf(robot_urdf, base_link, ee_link) if (robot_urdf and ee_link) else robot_file
    return _build_ik_solver(
        robot, scene=scene, num_seeds=num_seeds,
        position_threshold=0.001 if high_precision else 0.005,
        use_cuda_graph=use_cuda_graph, self_collision=collision_free)


def make_curobo_solver_from_urdf(urdf_path, base_link, ee_link, *, high_precision, use_cuda_graph, num_seeds):
    """Open-world cuRobo v2 IK solver from an arbitrary URDF (Fetch + the DoF variants for Table III).
    Builds a robot config via RobotBuilder rooted at base_link, no collisions."""
    robot_yml = _curobo_robot_yml_from_urdf(urdf_path, base_link, ee_link)
    return _build_ik_solver(
        robot_yml, scene=None, num_seeds=num_seeds,
        position_threshold=0.001 if high_precision else 0.005,
        use_cuda_graph=use_cuda_graph, self_collision=False)


_URDF_ROBOT_CACHE = {}

def _curobo_robot_yml_from_urdf(urdf_path, base_link, ee_link):
    """Build (once, cached) a cuRobo v2 robot config .yml from a URDF + base/ee links; return its path."""
    key = (os.path.abspath(urdf_path), base_link, ee_link)
    if key in _URDF_ROBOT_CACHE:
        return _URDF_ROBOT_CACHE[key]
    sub = _subtree_urdf(urdf_path, base_link)
    builder = RobotBuilder(urdf_path=os.path.abspath(sub), tool_frames=[ee_link])
    cfg = builder.build()
    out = os.path.join(tempfile.gettempdir(),
                       f"curobo_robot_{os.path.splitext(os.path.basename(urdf_path))[0]}_{ee_link}.yml")
    builder.save(cfg, out)
    _URDF_ROBOT_CACHE[key] = out
    return out


def _solve_one_pose(ik_solver, goal7_wxyz, return_seeds):
    """Solve a single pose [x,y,z, qw,qx,qy,qz] (wxyz) and return the raw v2 result."""
    pos = torch.tensor(goal7_wxyz[None, :3], dtype=torch.float32, device="cuda")
    quat = torch.tensor(goal7_wxyz[None, 3:7], dtype=torch.float32, device="cuda")
    tool = ik_solver.tool_frames[0]
    goal = GoalToolPose.from_poses({tool: Pose(position=pos, quaternion=quat)}, num_goalset=1)
    return ik_solver.solve_pose(goal, return_seeds=return_seeds)


def warmup_curobo_solver(ik_solver, tensor_args, goal7_wxyz: np.ndarray, repeat: int = 2):
    assert goal7_wxyz.shape == (7,)
    for _ in range(max(1, int(repeat))):
        _ = _solve_one_pose(ik_solver, goal7_wxyz, return_seeds=1)
        torch.cuda.synchronize()


def run_curobo_on_goal_batch(ik_solver, goal_batch_np: np.ndarray, tensor_args, print_k: int = 1):
    """Solve one pose (goal_batch_np is [1,7]) returning up to print_k best seeds. Mirrors the classic
    return tuple: (time_s, succ_pct, pos98_m, ori98_rad, sols[K,dof], pos_errs[K], ori_errs[K])."""
    k = max(1, int(print_k))
    goal7 = goal_batch_np.reshape(-1)
    st_time = time.time()
    result = _solve_one_pose(ik_solver, goal7, return_seeds=k)
    torch.cuda.synchronize()
    total_time = time.time() - st_time

    # v2 shapes: success/position_error/rotation_error (B=1, K); solution (B=1, K, dof)
    success = result.success.reshape(-1).detach().cpu().numpy().astype(bool)
    pos_all = result.position_error.reshape(-1).detach().cpu().numpy()
    ori_all = result.rotation_error.reshape(-1).detach().cpu().numpy()
    sol_all = result.solution.reshape(success.shape[0], -1).detach().cpu().numpy()

    succ = int(success.sum())
    succ_pct = 100.0 * succ / success.shape[0]
    order_all = np.argsort(pos_all + ori_all)
    if succ > 0:
        succ_idx = np.where(success)[0]
        order = succ_idx[np.argsort(pos_all[succ_idx] + ori_all[succ_idx])]
        pos98 = float(np.percentile(pos_all[succ_idx], 98))
        ori98 = float(np.percentile(ori_all[succ_idx], 98))
    else:
        best_i = int(order_all[0])
        pos98, ori98, order = float(pos_all[best_i]), float(ori_all[best_i]), order_all
    pick = order[:k]
    return total_time, succ_pct, pos98, ori98, sol_all[pick], pos_all[pick], ori_all[pick]


if __name__ == "__main__":
    if _HAS_CUROBO:
        setup_curobo_logger("error")
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--high_precision", action="store_true", default=False)
    parser.add_argument("--file_name", type=str, default="results")
    parser.add_argument("--robot_file", type=str, default="franka.yml")
    parser.add_argument("--problem_set", type=str, default="bookshelf_small_panda")
    parser.add_argument("--num_instances", type=int, default=100)
    parser.add_argument(
        "--mode", type=str, choices=["pyroki", "curobo"], default="pyroki",
        help="Which solver to benchmark.",
    )
    parser.add_argument(
        "--seed_list", type=str, default="1,10,100,1000,2000",
        help="Comma/space-separated seed sweep. PyRoki: num_seeds_init. cuRobo: IKSolverConfig.num_seeds.",
    )
    parser.add_argument("--collision_free", action="store_true", default=False)
    parser.add_argument("--use_cuda_graph", action="store_true", default=True)

    parser.add_argument("--print_idx", type=int, default=-1)
    parser.add_argument("--print_seeds", type=int, default=0)
    parser.add_argument("--solutions_out", type=str, default="")
    parser.add_argument("--solutions_k", type=int, default=50)
    parser.add_argument("--solutions_seed", type=int, default=2000)
    parser.add_argument("--mmd_dump", type=str, default="",
                        help="Write a per-target joint-config dump (JSON) over --goal_file targets for "
                             "MMD/Table IV (benchmark/run_mmd.py), then exit. Uses solutions_k best of solutions_seed.")
    parser.add_argument(
        "--goal_file", type=str, default=None,
        help="Path to ik_dataset.yml. When set (and --collision_free is not set), uses these as targets.",
    )
    parser.add_argument(
        "--num_goals", type=int, default=100,
        help="Number of FK-sampled goals when --collision_free is not set and --goal_file is not given.",
    )
    parser.add_argument(
        "--world_file", type=str, default="collision_test.yml",
        help="cuRobo world file for FK sampling and non-collision-free benchmark.",
    )
    # Table III (DoF scalability): open-world IK on an arbitrary URDF (the chained Panda variants).
    parser.add_argument("--robot-urdf", type=str, default="",
                        help="Custom URDF for the open-world run (DoF variants). Default: built-in panda.")
    parser.add_argument("--base-link", type=str, default="panda_link0")
    parser.add_argument("--ee-link", type=str, default="panda_hand",
                        help="EE link for the custom-URDF run (shared-target frame).")

    args = parser.parse_args()

    if args.mode == "curobo" and not _HAS_CUROBO:
        raise SystemExit(
            f"--mode curobo requires cuRobo, which is not installed ({_CUROBO_IMPORT_ERR}).\n"
            "cuRobo is backlogged (may not build on newer CUDA); see docs/source/user_guide/benchmarks/results.rst. "
            "Use --mode pyroki, or install cuRobo on a compatible box."
        )
    if args.collision_free and args.mode == "pyroki" and not _HAS_CUROBO:
        print("[baseline_bench] WARNING: cuRobo absent -> PyRoki collision-free runs report IK "
              "timing/accuracy only; the 'collision_free' column will be blank (collision check needs cuRobo).")

    robot_file = args.robot_file
    seed_list = parse_int_list(args.seed_list)
    if args.print_seeds > 0:
        seed_list = [args.print_seeds]

    print("running...")

    # ---- solutions-dump path (debugging) ----
    if args.solutions_out:
        batched_ik_many = make_batched_pyroki_ik_many(args.solutions_seed, args.solutions_k)
        t = np.array([0.4142281711101532, -0.5743789076805115, 0.38658469915390015,
                      0.5558769702911377, 0.43919798731803894, 0.5561493039131165,
                      -0.43451568484306335], dtype=np.float32)
        target_pos  = jnp.asarray(t[0:3][None, :])
        target_wxyz = jnp.asarray(t[3:7][None, :])
        jax.block_until_ready(batched_ik_many(target_wxyz, target_pos))
        sols = batched_ik_many(target_wxyz, target_pos)
        jax.block_until_ready(sols)
        sols_np = np.array(sols[0])
        with open(args.solutions_out, "w", encoding="utf-8") as f:
            for q in sols_np:
                f.write(",".join(f"{float(v):.10f}" for v in q) + "\n")
        print(f"[OK] wrote {sols_np.shape[0]} solutions to {args.solutions_out}")
        raise SystemExit(0)

    # ---- MMD / Table IV dump: K best configs per target over the shared open-world goal set ----
    if args.mmd_dump:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from mmd import save_config_dump
        K, seeds = int(args.solutions_k), int(args.solutions_seed)

        if args.goal_file is not None:
            goal_dataset = load_goal_yaml(args.goal_file)
        else:
            world_dict0 = {}  # open-world: cuRobo v2 solver ignores the world when collision_free=False
            base_solver, _ = make_curobo_solver_from_world_dict(
                robot_file=robot_file, world_dict=world_dict0, collision_free=False,
                high_precision=args.high_precision, use_cuda_graph=False, num_seeds=12)
            goal_dataset = sample_fk_goals_with_curobo(base_solver, args.num_goals)
            del base_solver; torch.cuda.empty_cache()

        per_target = []
        if args.mode == "curobo":
            world_dict = {}  # open-world: cuRobo v2 solver ignores the world when collision_free=False
            ik_solver, tensor_args = make_curobo_solver_from_world_dict(
                robot_file=robot_file, world_dict=world_dict, collision_free=False,
                high_precision=args.high_precision, use_cuda_graph=args.use_cuda_graph, num_seeds=seeds)
            warmup_curobo_solver(ik_solver, tensor_args, goal_dataset[0], repeat=2)
            for goal7 in goal_dataset:
                _, _, _, _, sols, _, _ = run_curobo_on_goal_batch(ik_solver, goal7[None, :], tensor_args, print_k=K)
                per_target.append([[float(v) for v in q] for q in sols])
        else:
            # open-world frame = panda_hand (ik_beam_hand), matching the shared targets
            solve_many = partial(ik_beam_hand.solve_ik_many, num_seeds_init=seeds, k=K)
            batched_many = jax.jit(jax.vmap(solve_many))
            jax.block_until_ready(batched_many(jnp.asarray(goal_dataset[0, 3:7][None, :]),
                                               jnp.asarray(goal_dataset[0, 0:3][None, :])))
            for goal7 in goal_dataset:
                sols = batched_many(jnp.asarray(goal7[3:7][None, :]), jnp.asarray(goal7[0:3][None, :]))
                jax.block_until_ready(sols)
                per_target.append([[float(v) for v in q] for q in np.array(sols[0])])

        dof = len(per_target[0][0]) if (per_target and per_target[0]) else None
        save_config_dump(args.mmd_dump, args.mode, per_target, num_dof=dof)
        print(f"[OK] wrote MMD dump {args.mmd_dump} ({len(per_target)} targets x up to {K} configs)")
        raise SystemExit(0)

    # ---- benchmark ----
    from collections import defaultdict
    data = defaultdict(list)

    def record_row(*, problem_idx, num_seeds, solver_name, time_ms, pos_err_mm, ori_err_rad, succ_pct=None):
        # pos error stored in MILLIMETERS to match benchmark/hjcd_ik_bench.py (canonical CSV unit).
        data["problem_idx"].append(int(problem_idx))
        data["Batch-Size"].append(int(num_seeds))
        data["solver"].append(str(solver_name))
        if succ_pct is not None:
            data["succ(%)"].append(float(succ_pct))
        data["IK-time(ms)"].append(float(time_ms))
        data["Pos-Error(mm)"].append(float(pos_err_mm))
        data["Ori-Error"].append(float(ori_err_rad))
        data[f"{solver_name}-time(ms)"].append(float(time_ms))
        data[f"{solver_name}-ori-err(rad)"].append(float(ori_err_rad))

    if args.collision_free:
        # ---- collision-free: per-instance targets from mb_problems.json ----
        instances = load_mb_problem_set(MB_JSON_PATH, args.problem_set)[:args.num_instances]

        if args.mode == "curobo":
            # FAIR cuRobo collision timing: build the solver ONCE per seed count (mirroring cuRobo's own
            # benchmark/ik_benchmark.py) and swap the obstacle scene per instance via update_world(), which
            # writes in-place into the pre-allocated collision_cache buffers and therefore keeps the captured
            # CUDA graph valid (no per-scene re-capture). All bookshelf instances share one obstacle topology
            # (12 cuboids + 10 cylinders << the 64/64 cache), so update_world never reallocates. Rebuilding a
            # solver per scene (the old path) both lost graph amortization AND hit cudaErrorStreamCaptureInvalidated.
            for num_seeds in seed_list:
                print(f"  curobo num_seeds: {num_seeds}")
                ik_solver, tensor_args = make_curobo_solver_from_world_dict(
                    robot_file=robot_file, world_dict=mb_instance_to_world_dict(instances[0]),
                    collision_free=True, high_precision=args.high_precision,
                    use_cuda_graph=args.use_cuda_graph, num_seeds=num_seeds,
                    # MB problems are posed in the grasp frame; match it (else ~100 mm wrong-frame error).
                    robot_urdf=args.robot_urdf, base_link=args.base_link, ee_link=args.ee_link,
                )
                # Warm up (and capture the graph) once on the first instance's scene + goal.
                wx0, wp0 = mb_instance_to_cylinder_goal(instances[0], eps=1e-4, rot_sign=+1)
                warmup_curobo_solver(ik_solver, tensor_args, np.concatenate([wp0, wx0]), repeat=3)
                for idx, inst in enumerate(instances):
                    ik_solver.update_world(_world_dict_to_scene(mb_instance_to_world_dict(inst)))
                    target_wxyz, target_pos = mb_instance_to_cylinder_goal(inst, eps=1e-4, rot_sign=+1)
                    goal7 = np.concatenate([target_pos, target_wxyz])
                    dt_s, succ_pct, pos98, ori98, sols, pos_errs, ori_errs = run_curobo_on_goal_batch(ik_solver, goal7[None, :], tensor_args)
                    record_row(problem_idx=idx, num_seeds=num_seeds, solver_name="curobo",
                               time_ms=dt_s * 1000.0, pos_err_mm=pos98 * 1000.0, ori_err_rad=ori98, succ_pct=succ_pct)
                    if args.print_idx == idx:
                        print("\n==== cuRobo single-solution dump ====")
                        print(f"problem_idx: {idx}")
                        print(f"time_ms:     {dt_s * 1000.0:.4f}")
                        print(f"target_pos:  {target_pos.tolist()}")
                        print(f"target_wxyz: {target_wxyz.tolist()}")
                        print(f"q_solution:  {sols[0].tolist()}")
                        print(f"pos_err_m:   {pos_errs[0]:.6e}")
                        print(f"ori_err_rad: {ori_errs[0]:.6e}")
                        print(f"succ(%):     {succ_pct:.1f}")
                        print("=====================================\n")
                # Free this seed-count's collision solver (+ its captured graph/collision buffers) before
                # building the next. Without this, 5 accumulated collision solvers fragment the CUDA graph
                # memory pool and a later capture forces a disallowed cudaMalloc -> StreamCaptureInvalidated.
                del ik_solver
                gc.collect()
                torch.cuda.empty_cache()
        else:
            for num_seeds_init in seed_list:
                print(f"  pyroki num_seeds_init: {num_seeds_init}")
                batched_ik_fn = make_batched_pyroki_ik(num_seeds_init)
                wxyz0, pos0 = mb_instance_to_cylinder_goal(instances[0], eps=1e-4, rot_sign=+1)
                jax.block_until_ready(batched_ik_fn(jnp.asarray(wxyz0[None, :]), jnp.asarray(pos0[None, :])))
                if args.print_idx >= 0:
                    if args.print_idx >= len(instances):
                        raise ValueError(f"--print_idx {args.print_idx} out of range (0..{len(instances)-1})")
                    print_one_solution(robot_file, instances[args.print_idx], batched_ik_fn, args.print_idx)
                for idx, inst in enumerate(instances):
                    dt_ms, pe, oe, ps, cs = eval_one_mb_instance(robot_file, inst, batched_ik_fn)
                    record_row(problem_idx=idx, num_seeds=num_seeds_init, solver_name="pyroki",
                               time_ms=dt_ms, pos_err_mm=pe * 1000.0, ori_err_rad=oe, succ_pct=100.0 if ps else 0.0)

    else:
        # ---- non-collision-free: shared goal dataset (file or FK-sampled) ----
        # NOTE: world_dict / cuRobo world configs are only needed by the cuRobo code paths below;
        # load them lazily there so PyRoki + --goal_file runs need no cuRobo.
        if args.goal_file is not None:
            print(f"  loading goals from {args.goal_file}")
            goal_dataset = load_goal_yaml(args.goal_file)
        else:
            print(f"  sampling {args.num_goals} FK goals via cuRobo FK (panda_hand_tcp)")
            world_dict = {}  # open-world: cuRobo v2 solver ignores the world when collision_free=False
            base_solver, _ = make_curobo_solver_from_world_dict(
                robot_file=robot_file, world_dict=world_dict,
                collision_free=False, high_precision=args.high_precision,
                use_cuda_graph=False, num_seeds=12,
            )
            goal_dataset = sample_fk_goals_with_curobo(base_solver, args.num_goals)
            del base_solver
            torch.cuda.empty_cache()

        print(f"  {goal_dataset.shape[0]} goals loaded")

        if args.mode == "curobo":
            for num_seeds in seed_list:
                print(f"  curobo num_seeds: {num_seeds}")
                if args.robot_urdf:
                    ik_solver, tensor_args = make_curobo_solver_from_urdf(
                        args.robot_urdf, args.base_link, args.ee_link,
                        high_precision=args.high_precision, use_cuda_graph=args.use_cuda_graph, num_seeds=num_seeds)
                else:
                    world_dict = {}  # open-world: cuRobo v2 solver ignores the world when collision_free=False
                    ik_solver, tensor_args = make_curobo_solver_from_world_dict(
                        robot_file=robot_file, world_dict=world_dict,
                        collision_free=False, high_precision=args.high_precision,
                        use_cuda_graph=args.use_cuda_graph, num_seeds=num_seeds,
                    )
                warmup_curobo_solver(ik_solver, tensor_args, goal_dataset[0], repeat=2)
                for idx, goal7 in enumerate(goal_dataset):
                    dt_s, succ_pct, pos98, ori98, _, _, _ = run_curobo_on_goal_batch(ik_solver, goal7[None, :], tensor_args)
                    record_row(problem_idx=idx, num_seeds=num_seeds, solver_name="curobo",
                               time_ms=dt_s * 1000.0, pos_err_mm=pos98 * 1000.0, ori_err_rad=ori98, succ_pct=succ_pct)
        else:
            # custom URDF (DoF variants) -> build a matching PyRoki beam + FK; else the panda_hand default.
            if args.robot_urdf:
                ow_beam = PyrokiIkBeamHelper(args.ee_link, urdf_path=args.robot_urdf)
                ow_fk = jax.jit(jax.vmap(ow_beam.forward_kinematics))
            else:
                ow_beam, ow_fk = ik_beam_hand, batched_fk_hand
            for num_seeds_init in seed_list:
                print(f"  pyroki num_seeds_init: {num_seeds_init}")
                batched_ik_fn = make_batched_pyroki_ik(num_seeds_init, beam=ow_beam)
                jax.block_until_ready(batched_ik_fn(
                    jnp.asarray(goal_dataset[0, 3:7][None, :]),
                    jnp.asarray(goal_dataset[0, 0:3][None, :]),
                ))
                for idx, goal7 in enumerate(goal_dataset):
                    dt_ms, pe, oe, ps = eval_pyroki_on_goal7(goal7, batched_ik_fn, batched_fk_fn=ow_fk)
                    record_row(problem_idx=idx, num_seeds=num_seeds_init, solver_name="pyroki",
                               time_ms=dt_ms, pos_err_mm=pe * 1000.0, ori_err_rad=oe, succ_pct=100.0 if ps else 0.0)

    # ---- save outputs ---- (stdlib paths + pyyaml; no cuRobo dependency)
    out_prefix = os.path.join(args.save_path, args.file_name) if args.save_path is not None else args.file_name
    out_prefix = out_prefix + ("_curobo" if args.mode == "curobo" else "_pyroki")

    data_out = convert_numpy_scalars(dict(data))
    _write_yaml_safe(data_out, out_prefix + ".yml")

    try:
        import pandas as pd
        df = pd.DataFrame(data_out)
        df.to_csv(out_prefix + ".csv", index=False)

        try:
            from tabulate import tabulate
            print(tabulate(df.head(50), headers="keys", tablefmt="grid"))
            if len(df) > 50:
                print(f"... ({len(df)} rows total)")
        except ImportError:
            print(df.head())

        # Averages by batch size
        print("\n" + "=" * 60)
        print("Averages by Batch Size")
        print("=" * 60)
        avg_cols = [c for c in ["IK-time(ms)", "Pos-Error(mm)", "Ori-Error", "succ(%)"] if c in df.columns]
        group_cols = [c for c in ["Batch-Size", "solver"] if c in df.columns]
        avg_df = df.groupby(group_cols)[avg_cols].mean()
        try:
            print(tabulate(avg_df, headers="keys", tablefmt="grid", floatfmt=".6g"))
        except Exception:
            print(avg_df.to_string())
        print("=" * 60 + "\n")

    except ImportError:
        pass
