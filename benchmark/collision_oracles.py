"""Solver-independent collision oracles for scoring returned Panda configurations.

Three judges of the same question ("is configuration q free of the scene's obstacles?"), so a solver's
collision-free rate can be reported under geometry it did not optimise against:

  hjcd   the URDF-derived sphere model HJCD-IK compiles (benchmark/panda_collision.py, 40 mm fingers)
  paper  the camera-ready paper's 59-sphere model (65 mm finger origins)
  mesh   the Panda collision meshes of `panda_description` (robot_descriptions), fingers fully open, checked
         against the obstacle primitives with FCL via trimesh. Needs `pip install python-fcl`.

All three ignore the base link (HJCD's base-contact policy), check environment obstacles only (no self
collision) and permit touching: `mesh` tolerates penetrations up to `tol_m`.
"""
from __future__ import annotations

import numpy as np

from panda_collision import panda_config_collision_free

MESH_FINGER_OPEN_M = 0.04      # fully open gripper, matching the sphere models' finger origins
MESH_TOUCH_TOL_M = 1e-3        # penetration depth treated as "touching"


def _quat_wxyz_to_T(pose7):
    x, y, z, qw, qx, qy, qz = [float(v) for v in pose7]
    n = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                 [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                 [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]]
    T[:3, 3] = [x, y, z]
    return T


class MeshOracle:
    """FCL mesh-vs-primitive check with the panda_description collision meshes. One instance per process;
    `collision_free(q, world_dict)` is ~ms per call."""

    def __init__(self, finger_open_m=MESH_FINGER_OPEN_M, tol_m=MESH_TOUCH_TOL_M, exclude_base=True):
        import trimesh
        import yourdfpy
        from robot_descriptions import panda_description
        import fcl  # noqa: F401  (trimesh.collision needs python-fcl)
        self._trimesh = trimesh
        self.urdf = yourdfpy.URDF.load(panda_description.URDF_PATH, build_collision_scene_graph=True,
                                       load_collision_meshes=True, build_scene_graph=False, load_meshes=False)
        self.tol_m = tol_m
        self.finger_open_m = finger_open_m
        base_geoms = set()
        if exclude_base:
            for node in self.urdf.collision_scene.graph.nodes_geometry:
                parent = self.urdf.collision_scene.graph.transforms.parents.get(node)
                if parent == "panda_link0" or node == "panda_link0":
                    base_geoms.add(self.urdf.collision_scene.graph[node][1])
        self._base_geoms = base_geoms
        self._world_cache = {}

    def _robot_manager(self, q7):
        cfg = np.zeros(len(self.urdf.actuated_joint_names))
        cfg[:7] = np.asarray(q7, float)[:7]
        if len(cfg) > 7:
            cfg[7:] = self.finger_open_m
        self.urdf.update_cfg(cfg)
        scene = self.urdf.collision_scene
        m = self._trimesh.collision.CollisionManager()
        for node in scene.graph.nodes_geometry:
            T, geom = scene.graph[node]
            if geom in self._base_geoms:
                continue
            m.add_object(node, scene.geometry[geom], transform=T)
        return m

    def _world_manager(self, world_dict):
        key = id(world_dict)
        if key in self._world_cache:
            return self._world_cache[key]
        tm = self._trimesh
        m = tm.collision.CollisionManager()
        for name, o in world_dict.get("cuboid", {}).items():
            m.add_object(f"cuboid:{name}", tm.creation.box(extents=[float(d) for d in o["dims"]]),
                         transform=_quat_wxyz_to_T(o["pose"]))
        for name, o in world_dict.get("cylinder", {}).items():
            m.add_object(f"cylinder:{name}", tm.creation.cylinder(radius=float(o["radius"]), height=float(o["height"])),
                         transform=_quat_wxyz_to_T(o["pose"]))
        self._world_cache[key] = m
        return m

    def max_penetration(self, q7, world_dict) -> float:
        """Deepest robot/obstacle penetration in metres (0.0 when free)."""
        robot = self._robot_manager(q7)
        world = self._world_manager(world_dict)
        if not world._objs:
            return 0.0
        hit, _, contacts = robot.in_collision_other(world, return_names=True, return_data=True)
        return float(max(c.depth for c in contacts)) if hit else 0.0

    def collision_free(self, q7, world_dict) -> bool:
        return self.max_penetration(q7, world_dict) <= self.tol_m


def make_oracles(names=("hjcd", "paper", "mesh")):
    """Name -> callable(q7, world_dict) -> bool. `mesh` is skipped (with a note) if python-fcl is missing."""
    oracles = {}
    for n in names:
        if n in ("hjcd", "paper"):
            oracles[n] = (lambda model: (lambda q, w: bool(panda_config_collision_free(q, w, model=model))))(n)
        elif n == "mesh":
            try:
                mo = MeshOracle()
                oracles[n] = mo.collision_free
            except ImportError as e:
                print(f"[collision_oracles] mesh oracle unavailable ({e}); pip install python-fcl")
    return oracles
