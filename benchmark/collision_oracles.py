"""Solver-independent collision oracles for scoring returned Panda configurations.

Three judges of the same question ("is configuration q free of the scene's obstacles?"), so a solver's
collision-free rate can be reported under geometry it did not optimise against:

  hjcd   the URDF-derived sphere model HJCD-IK compiles (benchmark/panda_collision.py, 40 mm fingers)
  paper  the camera-ready paper's 59-sphere model (65 mm finger origins)
  curobo cuRobo's bundled 61-sphere Panda model (franka.yml) on our kinematic chain; needs cuRobo installed
  hull   the Panda *collision* meshes of `panda_description` (franka_description's convex hulls — MoveIt's and
         MotionBenchMaker's own geometry), fingers fully open, FCL via trimesh. Needs `pip install python-fcl`.
  visual the Panda *visual* meshes (visual-link mesh approximation); an independent mesh approximation, not a safety certificate.

All of them ignore the base link (HJCD's base-contact policy), check environment obstacles only (no self
collision) and permit touching: the mesh judges tolerate `tol_m` by shrinking every obstacle by that much.
"""
from __future__ import annotations

import json

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
    """FCL mesh-vs-primitive check with the panda_description meshes. Two geometries of the same robot:

      geometry="hull"    the URDF's *collision* meshes — franka_description ships convex hulls (link5's is 50 %
                         larger than the link), i.e. what MoveIt and MotionBenchMaker validated the dataset with;
      geometry="visual"  the *visual* meshes (visual-link mesh approximation, ~50k vertices per link).

    `collision_free(q, world, tol_m)` is "no triangle intersection with every obstacle shrunk by tol_m" — the
    shrink is how touching is tolerated, since penetration depth is not reliable on non-convex meshes. One
    instance per process; a check is ~ms (hull) to ~10 ms (visual)."""

    def __init__(self, finger_open_m=MESH_FINGER_OPEN_M, tol_m=MESH_TOUCH_TOL_M, exclude_base=True,
                 geometry="hull"):
        import trimesh
        import yourdfpy
        from robot_descriptions import panda_description
        import fcl  # noqa: F401  (trimesh.collision needs python-fcl)
        if geometry not in ("hull", "visual"):
            raise ValueError("geometry must be 'hull' or 'visual'")
        self._trimesh = trimesh
        self.geometry = geometry
        visual = geometry == "visual"
        self.urdf = yourdfpy.URDF.load(panda_description.URDF_PATH,
                                       build_collision_scene_graph=not visual, load_collision_meshes=not visual,
                                       build_scene_graph=visual, load_meshes=visual, force_mesh=True)
        self.scene = self.urdf.scene if visual else self.urdf.collision_scene
        self.tol_m = tol_m
        self.finger_open_m = finger_open_m
        base_geoms = set()
        if exclude_base:
            for node in self.scene.graph.nodes_geometry:
                parent = self.scene.graph.transforms.parents.get(node)
                if parent == "panda_link0" or node == "panda_link0":
                    base_geoms.add(self.scene.graph[node][1])
        self._base_geoms = base_geoms
        self._world_cache = {}

    def _robot_manager(self, q7):
        """The robot's FCL manager posed at q7. Built once (BVHs of 50k-vertex visual meshes are costly) and
        re-posed with set_transform afterwards."""
        cfg = np.zeros(len(self.urdf.actuated_joint_names))
        cfg[:7] = np.asarray(q7, float)[:7]
        if len(cfg) > 7:
            cfg[7:] = self.finger_open_m
        self.urdf.update_cfg(cfg)
        scene = self.scene
        m = getattr(self, "_robot", None)
        if m is None:
            m = self._robot = self._trimesh.collision.CollisionManager()
            for node in scene.graph.nodes_geometry:
                T, geom = scene.graph[node]
                if geom in self._base_geoms:
                    continue
                m.add_object(node, scene.geometry[geom], transform=T)
        else:
            for node in scene.graph.nodes_geometry:
                T, geom = scene.graph[node]
                if geom in self._base_geoms:
                    continue
                m.set_transform(node, T)
        return m

    def _world_manager(self, world_dict, shrink_m=0.0):
        if set(world_dict) - {"sphere", "cuboid", "cylinder"}:
            raise ValueError("unsupported obstacle type")
        # Keyed by content, not id(): callers pass transient dicts whose ids get reused.
        key = (json.dumps(world_dict, sort_keys=True), round(shrink_m, 6))
        if key in self._world_cache:
            return self._world_cache[key]
        if len(self._world_cache) > 512:
            self._world_cache.clear()
        tm = self._trimesh
        m = tm.collision.CollisionManager()
        for name, o in world_dict.get("sphere", {}).items():
            pose = o.get("pose", [*o.get("position", []), 1, 0, 0, 0])
            m.add_object(f"sphere:{name}", tm.primitives.Sphere(radius=max(float(o["radius"]) - shrink_m, 1e-6)),
                         transform=_quat_wxyz_to_T(pose))
        for name, o in world_dict.get("cuboid", {}).items():
            dims = [max(float(d) - 2 * shrink_m, 1e-6) for d in o["dims"]]
            m.add_object(f"cuboid:{name}", tm.creation.box(extents=dims), transform=_quat_wxyz_to_T(o["pose"]))
        for name, o in world_dict.get("cylinder", {}).items():
            m.add_object(f"cylinder:{name}", tm.creation.cylinder(radius=max(float(o["radius"]) - shrink_m, 1e-6),
                                                                 height=max(float(o["height"]) - 2 * shrink_m, 1e-6)),
                         transform=_quat_wxyz_to_T(o["pose"]))
        self._world_cache[key] = m
        return m

    def in_collision(self, q7, world_dict, tol_m=0.0) -> bool:
        """Does the robot intersect any obstacle after shrinking every obstacle by tol_m?"""
        world = self._world_manager(world_dict, tol_m)
        if not world._objs:
            return False
        return bool(self._robot_manager(q7).in_collision_other(world))

    def max_penetration(self, q7, world_dict) -> float:
        """Deepest robot/obstacle penetration in metres (0.0 when free). Reliable for `hull` (convex pieces)
        only; prefer `in_collision(..., tol_m)` for tolerance decisions."""
        robot = self._robot_manager(q7)
        world = self._world_manager(world_dict)
        if not world._objs:
            return 0.0
        hit, _, contacts = robot.in_collision_other(world, return_names=True, return_data=True)
        return float(max(c.depth for c in contacts)) if hit else 0.0

    def collision_free(self, q7, world_dict, tol_m=None) -> bool:
        return not self.in_collision(q7, world_dict, self.tol_m if tol_m is None else tol_m)


def make_oracles(names=("hjcd", "paper", "curobo", "hull", "visual")):
    """Name -> callable(q7, world_dict) -> bool. `mesh` is skipped (with a note) if python-fcl is missing."""
    oracles = {}
    for n in names:
        if n in ("hjcd", "paper", "curobo"):
            if n == "curobo":
                try:
                    from panda_model import load_curobo_spheres
                    load_curobo_spheres()
                except ImportError as e:
                    print(f"[collision_oracles] curobo sphere oracle unavailable ({e})")
                    continue
            oracles[n] = (lambda model: (lambda q, w: bool(panda_config_collision_free(q, w, model=model))))(n)
        elif n in ("hull", "visual", "mesh"):          # "mesh" = the pre-10-03 name of "hull"
            try:
                mo = MeshOracle(geometry="hull" if n == "mesh" else n)
                oracles[n] = mo.collision_free
            except ImportError as e:
                print(f"[collision_oracles] mesh oracle unavailable ({e}); pip install python-fcl")
    return oracles
