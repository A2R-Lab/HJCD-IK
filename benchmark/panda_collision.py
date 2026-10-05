"""Place an explicitly selected Panda sphere model in the world at configuration q.

The default "paper" model preserves the shared historical Table II geometry.
Use model="hjcd" to validate the compiled default Panda geometry (different fixed
finger opening). These helpers check environment collisions, NOT self collisions.

The FK here replicates pRRTC's `fk<Panda>` (benchmark/reference/panda_collision_model.cuh) exactly: T_i = T_{i-1} @ fixed[i] @ R(q_{i-1})
for i=1..7, spheres placed by the transform of the joint they attach to. It is cross-validated against the
independent numpy URDF FK in gen_targets._fk (tests/... in benchmark/test_panda_collision.py).

Pure numpy (no GPU, no cuRobo, no jax).
"""
from __future__ import annotations

import numpy as np

from panda_model import panda_sphere_model, FIXED_TRANSFORMS, JOINT_TYPES, N_JOINTS
from collision_check import config_is_collision_free

# pRRTC joint-type codes (benchmark/reference/panda_collision_model.cuh): 0/1/2 = prism x/y/z, 3/4/5 = rot x/y/z.
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


def panda_spheres_world(q, *, model="paper") -> np.ndarray:
    """(N_SPHERES, 4) world spheres [x, y, z, radius] for the selected geometry."""
    spheres, anchors = panda_sphere_model(model)
    Ts = panda_link_transforms(q)
    out = np.empty((len(spheres), 4))
    for s, (x, y, z, r) in enumerate(spheres):
        p = Ts[int(anchors[s])] @ np.array([x, y, z, 1.0])
        out[s, :3] = p[:3]
        out[s, 3] = r
    return out


def panda_config_collision_free(q, world_dict, exclude_base: bool = True, margin: float = 0.0,
                                *, model="paper") -> bool:
    """Environment-only check for the selected geometry; touching is permitted.

    exclude_base=True matches HJCD's base-contact policy. The paper default is
    retained for cross-solver comparisons; implementation tests select "hjcd".
    """
    spheres = panda_spheres_world(q, model=model)
    if exclude_base:
        spheres = spheres[panda_sphere_model(model)[1] != 0]
    return config_is_collision_free(spheres, world_dict, margin)


def mb_instance_to_world_dict(inst: dict) -> dict:
    """Normalize supported obstacle collections; never silently discard geometry."""
    obs = inst.get("obstacles", {})
    if not isinstance(obs, dict) or set(obs) - {"cuboid", "cylinder", "sphere"}:
        raise ValueError("unsupported obstacle type")
    world = {"cuboid": {}, "cylinder": {}}
    for kind, shapes in obs.items():
        if isinstance(shapes, list):
            shapes = {str(i): s for i, s in enumerate(shapes)}
        world[kind] = {}
        for name, shape in shapes.items():
            o = dict(shape)
            if kind == "sphere" and "pose" not in o:
                o["pose"] = [*o["position"], 1, 0, 0, 0]
            if kind == "cylinder" and "height" not in o:
                o["height"] = o["length"]
            world[kind][name] = o
    return world
