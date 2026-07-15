"""Independent pure-numpy URDF forward kinematics — the trusted oracle for the CUDA FK.

Deliberately dumb and general: it does link-level FK over the WHOLE URDF tree (fixed joints
included), so it is branch-correct for a humanoid without any special casing. GRiD's world transform
of movable joint j is exactly the world pose of that joint's CHILD LINK, so the two line up directly.

Joint ordering must match GRiD's (DFS pre-order over movable joints, fixed joints folded out).
`joint_order` recomputes it independently; `assert_matches_grid` cross-checks it against GRiD's own
URDFParser so an ordering drift fails loudly instead of silently comparing the wrong joints.

numpy + stdlib only — no GPU, no GRiD, no pinocchio.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def _rpy_to_R(rpy):
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def _axis_angle_to_R(axis, theta):
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _xform(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


class UrdfFK:
    """Whole-tree URDF FK. `T[link_name]` is the link's world pose (4x4, row-major)."""

    def __init__(self, urdf_path):
        root = ET.parse(str(urdf_path)).getroot()
        self.joints = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            a = j.find("axis")
            xyz = [float(v) for v in (o.get("xyz") if o is not None and o.get("xyz") else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy") if o is not None and o.get("rpy") else "0 0 0").split()]
            axis = [float(v) for v in a.get("xyz").split()] if a is not None else [0.0, 0.0, 1.0]
            self.joints[j.get("name")] = dict(
                name=j.get("name"), type=j.get("type"),
                parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                xyz=np.array(xyz), rpy=np.array(rpy), axis=np.array(axis),
            )
        children = {jv["child"] for jv in self.joints.values()}
        roots = [jv["parent"] for jv in self.joints.values() if jv["parent"] not in children]
        self.root_link = roots[0]
        self._kids = {}
        for jv in self.joints.values():                      # URDF document order preserved
            self._kids.setdefault(jv["parent"], []).append(jv)

    def joint_order(self):
        """Movable joints in GRiD's order: DFS pre-order from the root, fixed joints folded out."""
        order = []

        def dfs(link):
            for jv in self._kids.get(link, []):
                if jv["type"] not in ("fixed",):
                    order.append(jv["name"])
                dfs(jv["child"])

        dfs(self.root_link)
        return order

    def fk(self, q):
        """q: (N,) over joint_order(). Returns {link_name: 4x4 world pose}."""
        names = self.joint_order()
        assert len(q) == len(names), f"q has {len(q)} entries, robot has {len(names)} movable joints"
        qmap = dict(zip(names, np.asarray(q, dtype=float)))
        T = {self.root_link: np.eye(4)}

        def dfs(link):
            for jv in self._kids.get(link, []):
                Tj = _xform(_rpy_to_R(jv["rpy"]), jv["xyz"])
                if jv["type"] in ("revolute", "continuous"):
                    Tj = Tj @ _xform(_axis_angle_to_R(jv["axis"], qmap[jv["name"]]), np.zeros(3))
                elif jv["type"] == "prismatic":
                    a = jv["axis"] / np.linalg.norm(jv["axis"])
                    Tj = Tj @ _xform(np.eye(3), a * qmap[jv["name"]])
                T[jv["child"]] = T[link] @ Tj
                dfs(jv["child"])

        dfs(self.root_link)
        return T

    def joint_world_transforms(self, q):
        """(N, 4, 4) world pose of each movable joint == the world pose of its child link."""
        T = self.fk(q)
        return np.stack([T[self.joints[n]["child"]] for n in self.joint_order()])

    def joint_world_axes(self, q):
        """(N, 3) each movable joint's rotation/translation axis in world frame."""
        Tj = self.joint_world_transforms(q)
        out = np.zeros((len(Tj), 3))
        for i, n in enumerate(self.joint_order()):
            a = self.joints[n]["axis"]
            out[i] = Tj[i][:3, :3] @ (a / np.linalg.norm(a))
        return out

    def ancestor_matrix(self):
        """(N, N) bool: A[k, j] is True iff movable joint j is an ancestor of (or is) joint k.

        Column j of a target-k geometric Jacobian is nonzero only where A[k, j] — moving a joint in
        another branch cannot move target k. This is the ground truth for the generated ancestor
        masks, and the reason a whole-body Jacobian cannot just fill all N columns on a humanoid.
        """
        names = self.joint_order()
        idx = {n: i for i, n in enumerate(names)}
        parent_link = {jv["child"]: jv["parent"] for jv in self.joints.values()}
        movable_by_child = {jv["child"]: jv["name"] for jv in self.joints.values()
                            if jv["type"] != "fixed"}
        n = len(names)
        A = np.zeros((n, n), dtype=bool)
        for k, name in enumerate(names):
            link = self.joints[name]["child"]
            while link is not None:
                if link in movable_by_child:
                    A[k, idx[movable_by_child[link]]] = True
                link = parent_link.get(link)
        return A

    def assert_matches_grid(self, urdf_path):
        """Cross-check our independent DFS ordering against GRiD's own URDFParser."""
        import sys
        grid_dir = Path(__file__).resolve().parents[1] / "external" / "GRiD"
        sys.path.insert(0, str(grid_dir))
        from URDFParser import URDFParser  # noqa: E402

        robot = URDFParser().parse(str(urdf_path))
        grid_names = [robot.get_joint_by_id(i).get_name() for i in range(robot.get_num_joints())]
        ours = self.joint_order()
        assert ours == grid_names, (
            "joint ordering drift between the numpy oracle and GRiD:\n"
            f"  oracle: {ours}\n  GRiD  : {grid_names}"
        )
        return grid_names
