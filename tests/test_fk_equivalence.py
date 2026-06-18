"""FK-equivalence tests for HJCD-IK.

Validates the GRiD-generated forward kinematics used by the solver against an *independent*
reference computed in pure numpy directly from the URDF (no GRiD codegen involved). This guards
the FK migration (bespoke ``X_warp`` -> stock ``grid::ee_pose_inner_warp`` + grasptarget tool
offset) against silent convention / tool-offset bugs.

``generate_solutions`` returns both the solved joint configuration and the EE pose the kernel
computes for it; we recompute the grasptarget world pose from the URDF and compare position +
orientation (quaternion, with sign disambiguation).

Skips cleanly when the built extension or the URDF is unavailable.
"""
import os
import xml.etree.ElementTree as ET

import pytest

np = pytest.importorskip("numpy")
hjcdik = pytest.importorskip("hjcdik")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = os.path.join(REPO, "include", "test_urdf", "panda.urdf")

POS_ATOL_M = 1e-4      # 0.1 mm
QUAT_ATOL = 1e-3


def _rpy_to_R(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


def _axis_R(axis, q):
    a = axis / np.linalg.norm(axis)
    c, s = np.cos(q), np.sin(q)
    x, y, z = a
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])


def _parse_joints():
    root = ET.parse(URDF).getroot()
    joints = {}
    for j in root.findall("joint"):
        o, a = j.find("origin"), j.find("axis")
        xyz = np.array([float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()])
        rpy = np.array([float(v) for v in (o.get("rpy") if o is not None else "0 0 0").split()])
        ax = np.array([float(v) for v in a.get("xyz").split()]) if a is not None else None
        joints[j.get("name")] = dict(type=j.get("type"), parent=j.find("parent").get("link"),
                                     child=j.find("child").get("link"), xyz=xyz, rpy=rpy, axis=ax)
    return joints


def _fk_grasptarget(q):
    """World 4x4 transform of panda_grasptarget_hand for actuated config q (independent numpy FK)."""
    joints = _parse_joints()
    by_child = {v["child"]: (k, v) for k, v in joints.items()}
    target_joint = next(k for k in joints if "grasptarget" in k)
    chain = []
    cur = joints[target_joint]["child"]
    while cur in by_child:
        name, v = by_child[cur]
        chain.append(v)
        cur = v["parent"]
    chain.reverse()  # base -> grasptarget
    T = np.eye(4)
    qi = 0
    for v in chain:
        Tj = np.eye(4)
        Tj[:3, :3] = _rpy_to_R(*v["rpy"])
        Tj[:3, 3] = v["xyz"]
        if v["type"] in ("revolute", "continuous"):
            Rj = np.eye(4); Rj[:3, :3] = _axis_R(v["axis"], q[qi]); qi += 1
            Tj = Tj @ Rj
        elif v["type"] == "prismatic":
            d = np.eye(4); d[:3, 3] = v["axis"] * q[qi]; qi += 1
            Tj = Tj @ d
        T = T @ Tj
    return T, qi


def _quat_from_R(R):
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def _kernel_qpose_pairs(num=12, seed=7, batch_size=4000):
    """(q, kernel_pose) pairs: solve sampled targets and read back the solution + its EE pose."""
    if not os.path.exists(URDF):
        pytest.skip(f"URDF not found at {URDF}")
    pairs = []
    for t in hjcdik.sample_targets(num_targets=num, seed=seed):
        out = hjcdik.generate_solutions(t, batch_size=batch_size, num_solutions=1)
        if out["count"] < 1:
            continue
        q = np.asarray(out["joint_config"])[0]
        pose = np.asarray(out["pose"])[0]  # [x,y,z, qw,qx,qy,qz]
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(pose)):
            continue
        pairs.append((q, pose))
    if not pairs:
        pytest.skip("solver returned no finite solutions to check")
    return pairs


def test_ee_position_matches_reference():
    worst = 0.0
    for q, pose in _kernel_qpose_pairs():
        T, used = _fk_grasptarget(q)
        assert used == hjcdik.num_joints(), f"actuated-joint count mismatch: {used}"
        worst = max(worst, float(np.linalg.norm(T[:3, 3] - pose[:3])))
    assert worst < POS_ATOL_M, f"max EE position error {worst*1e3:.4f} mm exceeds {POS_ATOL_M*1e3:.4f} mm"


def test_ee_orientation_matches_reference():
    worst = 0.0
    for q, pose in _kernel_qpose_pairs():
        T, _ = _fk_grasptarget(q)
        quat = _quat_from_R(T[:3, :3])
        kq = pose[3:7]
        if np.dot(quat, kq) < 0:
            quat = -quat
        worst = max(worst, float(np.linalg.norm(quat - kq)))
    assert worst < QUAT_ATOL, f"max EE quaternion error {worst:.2e} exceeds {QUAT_ATOL:.2e}"
