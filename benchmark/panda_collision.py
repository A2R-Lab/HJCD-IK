"""Place the shared 59-sphere Panda model in the world frame at a joint config q, and decide whether that
config is collision-free against a MotionBenchMaker world. Used to validate *any* solver's returned q for
the Table II collision column with one consistent geometry (the paper's own model).

The FK here replicates pRRTC's `fk<Panda>` (csrc/robots/panda.cuh) exactly: T_i = T_{i-1} @ fixed[i] @ R(q_{i-1})
for i=1..7, spheres placed by the transform of the joint they attach to. It is cross-validated against the
independent numpy URDF FK in gen_targets._fk (tests/... in benchmark/test_panda_collision.py).

Pure numpy (no GPU, no cuRobo, no jax).
"""
from __future__ import annotations

import numpy as np

from panda_model import SPHERES, SPHERE_TO_JOINT, FIXED_TRANSFORMS, JOINT_TYPES, N_JOINTS
from collision_check import config_is_collision_free

# pRRTC joint-type codes (csrc/robots/panda.cuh): 0/1/2 = prism x/y/z, 3/4/5 = rot x/y/z.
_X_PRISM, _Y_PRISM, _Z_PRISM, _X_ROT, _Y_ROT, _Z_ROT = 0, 1, 2, 3, 4, 5


def _joint_motion(jtype: int, q: float) -> np.ndarray:
    """4x4 motion of one joint (composed AFTER its fixed transform), matching pRRTC's *_fn helpers."""
    c, s = np.cos(q), np.sin(q)
    T = np.eye(4)
    if jtype == _Z_ROT:
        T[0, 0], T[0, 1], T[1, 0], T[1, 1] = c, -s, s, c
    elif jtype == _X_ROT:
        T[1, 1], T[1, 2], T[2, 1], T[2, 2] = c, -s, s, c
    elif jtype == _Y_ROT:
        T[0, 0], T[0, 2], T[2, 0], T[2, 2] = c, s, -s, c
    elif jtype in (_X_PRISM, _Y_PRISM, _Z_PRISM):
        T[jtype, 3] = q
    else:
        raise ValueError(f"unsupported pRRTC joint type {jtype}")
    return T


def panda_link_transforms(q) -> list[np.ndarray]:
    """World 4x4 transform accumulated after each joint i (i=0 base .. N_JOINTS-1). `q` has N_JOINTS-1
    actuated values (q[0] drives joint 1, ...), matching pRRTC's `q[i-1]` indexing."""
    q = np.asarray(q, dtype=float)
    Ts = [np.eye(4)]                                   # i = 0: base frame (identity)
    T = np.eye(4)
    for i in range(1, N_JOINTS):
        T = T @ FIXED_TRANSFORMS[i] @ _joint_motion(int(JOINT_TYPES[i]), q[i - 1])
        Ts.append(T.copy())
    return Ts


def panda_spheres_world(q) -> np.ndarray:
    """(N_SPHERES, 4) collision spheres [x, y, z, radius] in the world frame at config q."""
    Ts = panda_link_transforms(q)
    out = np.empty((len(SPHERES), 4))
    for s, (x, y, z, r) in enumerate(SPHERES):
        p = Ts[int(SPHERE_TO_JOINT[s])] @ np.array([x, y, z, 1.0])
        out[s, :3] = p[:3]
        out[s, 3] = r
    return out


def panda_config_collision_free(q, world_dict, exclude_base: bool = True, margin: float = 0.0) -> bool:
    """True iff the Panda at config q is collision-free against `world_dict`.
    `exclude_base=True` drops the base-link spheres (SPHERE_TO_JOINT==0), matching the HJCD kernel, which
    skips them because the base is bolted to the pedestal and 'always contacts' it."""
    spheres = panda_spheres_world(q)
    if exclude_base:
        spheres = spheres[SPHERE_TO_JOINT != 0]
    return config_is_collision_free(spheres, world_dict, margin)


def mb_instance_to_world_dict(inst: dict) -> dict:
    """A MotionBenchMaker problem instance -> the ``{"cuboid":{...}, "cylinder":{...}}`` world_dict that
    ``config_is_collision_free`` / ``panda_config_collision_free`` consume. Pure data reshape (numpy-free),
    so any solver's returned q can be validated against the same shared model without importing the heavy
    baseline harness. Keep in sync with the identical reshape in benchmark/baseline_bench.py."""
    obs = inst.get("obstacles", {})
    world = {"cuboid": {}, "cylinder": {}}
    for name, o in obs.get("cuboid", {}).items():
        world["cuboid"][name] = {"dims": o["dims"], "pose": o["pose"]}
    for name, o in obs.get("cylinder", {}).items():
        world["cylinder"][name] = {"radius": o["radius"], "height": o["height"], "pose": o["pose"]}
    return world
