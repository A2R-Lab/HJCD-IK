"""GRiD-independent G1 URDF model + generic fixed-base tree FK for the collision sidecar.

This module deliberately does NOT import GRiD, URDFParser, or any generated code. It parses
the canonical G1 URDF with the stdlib XML parser and builds:

    * the link tree (parent/child, fixed-vs-movable joints)
    * per-joint fixed parent->joint transform + axis
    * the HJCD actuated-joint order (29 joints), VERIFIED BY NAME against the build
    * generic FK: T_link(q) for every link, movable and fixed

FK convention (spec section 5):
    T_child(q) = T_parent(q) * T_parent_to_joint * T_joint(q_j)
where T_parent_to_joint is the URDF <origin> of the joint and T_joint(q_j) is the actuated
rotation about the joint axis (identity for fixed joints).

HJCD optimizes the 29 actuated joints only; there is no floating base here (fixed-base tree,
pelvis at identity). Self-collision is base-pose invariant, so this is exactly what the
sidecar needs.
"""
from __future__ import annotations

import hashlib
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

# The HJCD actuated-joint order (29), as the G1 build consumes q. This is the URDF's
# movable-joint depth-first order; it is ASSERTED by name against the build at load time
# (see verify_joint_order), never blindly trusted.
HJCD_JOINT_ORDER = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def _rpy_to_mat(rpy):
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _homog(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _axis_angle(axis, angle):
    a = np.asarray(axis, float)
    a = a / (np.linalg.norm(a) + 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


@dataclass
class Joint:
    name: str
    jtype: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray

    @property
    def movable(self):
        return self.jtype in ("revolute", "continuous", "prismatic")


@dataclass
class URDFModel:
    path: str
    links: list
    joints: list                       # in URDF document order
    joint_by_name: dict = field(default_factory=dict)
    parent_of_link: dict = field(default_factory=dict)   # link -> parent joint name
    children_joints: dict = field(default_factory=dict)  # link -> [joint names]
    root_link: str = ""

    def joint_origin(self, j: Joint):
        return _homog(_rpy_to_mat(j.origin_rpy), j.origin_xyz)

    def joint_motion(self, j: Joint, q_value: float):
        if not j.movable:
            return np.eye(4)
        if j.jtype == "prismatic":
            return _homog(np.eye(3), j.axis * q_value)
        return _homog(_axis_angle(j.axis, q_value), np.zeros(3))

    # -- generic fixed-base tree FK ---------------------------------------
    def fk(self, q_by_name: dict):
        """Return {link_name: 4x4 world transform} with the root link at identity."""
        T = {self.root_link: np.eye(4)}
        # iterate links in topological order (BFS from root)
        order, frontier = [], [self.root_link]
        while frontier:
            link = frontier.pop(0)
            order.append(link)
            for jn in self.children_joints.get(link, ()):
                frontier.append(self.joint_by_name[jn].child)
        for link in order:
            jn = self.parent_of_link.get(link)
            if jn is None:
                continue
            j = self.joint_by_name[jn]
            qv = float(q_by_name.get(j.name, 0.0)) if j.movable else 0.0
            T[link] = T[j.parent] @ self.joint_origin(j) @ self.joint_motion(j, qv)
        return T

    def q_vector_to_names(self, q):
        """Map an HJCD-ordered 29-vector to {joint_name: value}."""
        assert len(q) == len(HJCD_JOINT_ORDER), (len(q), len(HJCD_JOINT_ORDER))
        return {n: float(v) for n, v in zip(HJCD_JOINT_ORDER, q)}

    def movable_joint_names(self):
        return [j.name for j in self.joints if j.movable]

    def descendant_links(self, joint_name: str):
        """All links strictly below (and including the child of) a movable joint."""
        out, frontier = [], [self.joint_by_name[joint_name].child]
        while frontier:
            link = frontier.pop(0)
            out.append(link)
            for jn in self.children_joints.get(link, ()):
                frontier.append(self.joint_by_name[jn].child)
        return out

    def urdf_hash(self):
        return hashlib.sha1(open(self.path, "rb").read()).hexdigest()[:16]

    def joint_order_hash(self):
        return hashlib.sha1("\n".join(HJCD_JOINT_ORDER).encode()).hexdigest()[:16]


def parse_urdf(path) -> URDFModel:
    root = ET.parse(path).getroot()
    links = [l.get("name") for l in root.findall("link")]
    joints = []
    for je in root.findall("joint"):
        origin = je.find("origin")
        xyz = np.array([float(v) for v in (origin.get("xyz", "0 0 0").split())]) if origin is not None else np.zeros(3)
        rpy = np.array([float(v) for v in (origin.get("rpy", "0 0 0").split())]) if origin is not None else np.zeros(3)
        axis_e = je.find("axis")
        axis = np.array([float(v) for v in axis_e.get("xyz").split()]) if axis_e is not None else np.array([0, 0, 1.0])
        joints.append(Joint(
            name=je.get("name"), jtype=je.get("type"),
            parent=je.find("parent").get("link"), child=je.find("child").get("link"),
            origin_xyz=xyz, origin_rpy=rpy, axis=axis))
    m = URDFModel(path=os.path.abspath(path), links=links, joints=joints)
    m.joint_by_name = {j.name: j for j in joints}
    child_links = set()
    for j in joints:
        m.parent_of_link[j.child] = j.name
        m.children_joints.setdefault(j.parent, []).append(j.name)
        child_links.add(j.child)
    roots = [l for l in links if l not in child_links]
    assert len(roots) == 1, f"expected 1 root link, got {roots}"
    m.root_link = roots[0]
    return m


def verify_joint_order(model: URDFModel, expected_names):
    """Fail fast if the sidecar joint ordering disagrees with the HJCD build's (spec s5)."""
    got = list(HJCD_JOINT_ORDER)
    exp = list(expected_names)
    if got != exp:
        diffs = [(i, g, e) for i, (g, e) in enumerate(zip(got, exp)) if g != e]
        raise ValueError(f"HJCD/sidecar joint-order mismatch at {diffs[:5]} "
                         f"(len {len(got)} vs {len(exp)})")
    movable = model.movable_joint_names()
    missing = [n for n in got if n not in movable]
    if missing:
        raise ValueError(f"sidecar joints not movable in URDF: {missing}")
    return True
