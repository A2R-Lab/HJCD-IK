"""CPU-only G1 tree-model contract (humanoid-v1 M2a): joint order, axes, contact frames, tree validator.

Three independent kinematic references are compared in fp64:
  * `urdf_fk.UrdfFK`  -- pure-numpy whole-tree URDF FK (fixed joints included);
  * GRiD's own parsed model -- per-joint `Xmat_hom` (child pose in parent) composed over GRiD parent ids,
    i.e. exactly the joint order, axes and fixed-frame folding the generated kernels will use;
  * MuJoCo (optional) on the MJCF CRAG verifies against, with the floating base at the identity.
No CUDA, no generated header. The MuJoCo comparisons skip when mujoco or the MJCF is unavailable;
set HJCD_G1_MJCF to point at the MJCF (default: the CRAG checkout this repo is a submodule of).
"""
import contextlib
import importlib.util
import io
import json
import os
import warnings
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from urdf_fk import UrdfFK

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "csrc/urdf/g1_29dof_contacts.urdf"
SPEC = json.loads((ROOT / "csrc/urdf/g1_29dof_contacts.targets.json").read_text())
TARGETS = [(t["name"], t["frame"]) for t in SPEC["targets"]]
MJCF = Path(os.environ.get("HJCD_G1_MJCF", ROOT.parents[1] / "assets/robot_models/g1/g1.xml"))

spec = importlib.util.spec_from_file_location("robot_model_g1_test", ROOT / "hjcdik/_robot_model.py")
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)

# The order the humanoid branch (bbc53a9) baked into hjcd_targets.json, and that CRAG's IK adapter
# concatenates into MuJoCo qpos[7:] without a permutation (src/crag/robot/ik.py).
G1_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
TARGET_ANCHOR = {"left_hand": "left_wrist_yaw_joint", "right_hand": "right_wrist_yaw_joint",
                 "left_foot": "left_ankle_roll_joint", "right_foot": "right_ankle_roll_joint"}
# Contact frame in its anchor link (= MuJoCo site in its body): identity rotation for all four.
TOOL_XYZ = {"left_hand": (0, 0, 0), "right_hand": (0, 0, 0),
            "left_foot": (0.035, 0, -0.035), "right_foot": (0.035, 0, -0.035)}
ANCESTOR_MASK = {"left_hand": 0x003FF000, "right_hand": 0x1FC07000, "left_foot": 0x3F, "right_foot": 0xFC0}
MJ_SITE = {name: f"{name}_contact" for name in TARGET_ANCHOR}


def parse(path=URDF, floating_base=False):
    import grid_rbd  # noqa: F401
    from URDFParser import URDFParser
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        return URDFParser().parse(str(path), floating_base=floating_base)


@pytest.fixture(scope="module")
def robot():
    return parse()


@pytest.fixture(scope="module")
def oracle():
    return UrdfFK(URDF)


@pytest.fixture(scope="module")
def described(robot):
    return model.describe_tree_robot(robot, URDF, TARGETS)


@pytest.fixture(scope="module")
def mj():
    mujoco = pytest.importorskip("mujoco")
    if not MJCF.exists():
        pytest.skip(f"G1 MJCF not found at {MJCF}; set HJCD_G1_MJCF")
    m = mujoco.MjModel.from_xml_path(str(MJCF))
    return mujoco, m, mujoco.MjData(m)


def limits(described):
    return np.array([j["limits"] for j in described["joints"]])


def configs(described):
    """zero, nominal, 200 uniform-in-limits, asymmetric left/right, waist-only and leg-only."""
    lo, hi = limits(described).T
    rng = np.random.default_rng(20261103)
    nominal = np.clip(np.array([-0.3, 0.1, 0.0, 0.6, -0.3, 0.05] + [-0.3, -0.1, 0.0, 0.6, -0.3, -0.05]
                               + [0.2, 0.1, 0.15] + [-0.8, 0.4, 0.2, 0.9, 0.3, -0.4, 0.2]
                               + [-0.8, -0.4, -0.2, 0.9, -0.3, -0.4, -0.2]), lo, hi)
    out = {"zero": [np.zeros(29)], "nominal": [nominal], "random": list(rng.uniform(lo, hi, (200, 29)))}
    asym = rng.uniform(lo, hi, (20, 29))
    asym[:, [6, 7, 8, 9, 10, 11, 22, 23, 24, 25, 26, 27, 28]] = 0.0     # right side held at zero
    out["asymmetric"] = list(asym)
    waist = np.zeros((10, 29)); waist[:, 12:15] = rng.uniform(lo[12:15], hi[12:15], (10, 3))
    legs = np.zeros((10, 29)); legs[:, 0:6] = rng.uniform(lo[0:6], hi[0:6], (10, 6))
    out["waist_only"], out["left_leg_only"] = list(waist), list(legs)
    return out


def grid_fk(robot, q):
    """World (=base_link) pose of each actuated joint's child link from GRiD's parsed model."""
    T = []
    for j in range(robot.get_num_joints()):
        X = np.array(robot.get_Xmat_hom_Func_by_id(j)(float(q[j])), dtype=float)
        p = robot.get_parent_id(j)
        T.append(X if p == -1 else T[p] @ X)
    return np.stack(T)


def grid_targets(robot, described, q):
    T = grid_fk(robot, q)
    return {t["name"]: T[t["anchor_jid"]] @ np.array(t["tool_row_major"]) for t in described["targets"]}


def mj_fk(mj, q):
    """MuJoCo body and site poses with the freejoint at the identity (base_link = world)."""
    mujoco, m, d = mj
    d.qpos[:] = 0
    d.qpos[3] = 1.0
    for i, name in enumerate(G1_JOINTS):
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]] = q[i]
    mujoco.mj_kinematics(m, d)
    def pose(pos, mat):
        T = np.eye(4); T[:3, :3] = np.asarray(mat).reshape(3, 3); T[:3, 3] = pos; return T
    bodies = {m.body(b).name: pose(d.xpos[b], d.xmat[b]) for b in range(m.nbody)}
    sites = {m.site(s).name: pose(d.site_xpos[s], d.site_xmat[s]) for s in range(m.nsite)}
    return bodies, sites


def pose_err(A, B):
    dp = np.linalg.norm(A[:3, 3] - B[:3, 3])
    c = np.clip((np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2, -1, 1)
    return dp, float(np.arccos(c))


# --- joint order ---------------------------------------------------------------------------------

def test_joint_order_grid_equals_urdf_dfs_and_pinned(robot, oracle):
    assert oracle.assert_matches_grid(URDF) == G1_JOINTS
    assert [robot.get_joint_by_id(i).get_name() for i in range(29)] == G1_JOINTS


def test_joint_order_matches_mujoco_qpos_by_name(mj):
    mujoco, m, _ = mj
    free = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "floating_base")
    assert m.jnt_type[free] == mujoco.mjtJoint.mjJNT_FREE and m.jnt_qposadr[free] == 0
    assert m.nq == 7 + 29 and m.nv == 6 + 29
    for i, name in enumerate(G1_JOINTS):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert jid >= 0, name
        assert m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE, name
        assert m.jnt_qposadr[jid] == 7 + i, f"{name}: qpos {m.jnt_qposadr[jid]} != {7 + i}"
        assert m.jnt_dofadr[jid] == 6 + i, name


def test_joint_limits_match_mujoco(described, mj):
    mujoco, m, _ = mj
    for i, name in enumerate(G1_JOINTS):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        np.testing.assert_allclose(m.jnt_range[jid], described["joints"][i]["limits"], atol=1e-12, err_msg=name)


# --- axes ----------------------------------------------------------------------------------------

def test_joint_axes_grid_urdf_mujoco(robot, oracle, described, mj):
    mujoco, m, _ = mj
    for i, name in enumerate(G1_JOINTS):
        a = oracle.joints[name]["axis"] / np.linalg.norm(oracle.joints[name]["axis"])
        np.testing.assert_array_equal(described["joints"][i]["axis"], a, err_msg=name)
        np.testing.assert_allclose(m.jnt_axis[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)], a,
                                   atol=1e-12, err_msg=name)
    # world axes under motion, and the +theta sense: a small step rotates the child about +axis
    for q in configs(described)["random"][:20]:
        Tg = grid_fk(robot, q)
        world_axes = oracle.joint_world_axes(q)
        for j in range(29):
            a_local = np.asarray(described["joints"][j]["axis"])
            np.testing.assert_allclose(Tg[j][:3, :3] @ a_local, world_axes[j], atol=1e-12)
            dq = np.zeros(29); dq[j] = 1e-6
            dR = grid_fk(robot, q + dq)[j][:3, :3] @ Tg[j][:3, :3].T
            omega = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / 2e-6
            np.testing.assert_allclose(omega, world_axes[j], atol=1e-5)


# --- contact frames ------------------------------------------------------------------------------

def test_target_tools_are_the_mujoco_sites(described, mj):
    mujoco, m, _ = mj
    for t in described["targets"]:
        tool = np.array(t["tool_row_major"])
        np.testing.assert_array_equal(tool[:3, :3], np.eye(3))
        np.testing.assert_allclose(tool[:3, 3], TOOL_XYZ[t["name"]], atol=0)
        assert t["anchor_joint"] == TARGET_ANCHOR[t["name"]]
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, MJ_SITE[t["name"]])
        body = m.body(m.site_bodyid[sid]).name
        assert body == t["anchor_joint"].replace("_joint", "_link")
        np.testing.assert_allclose(m.site_pos[sid], tool[:3, 3], atol=1e-12)
        np.testing.assert_allclose(m.site_quat[sid], [1, 0, 0, 0], atol=1e-12)


def test_fk_three_way(robot, oracle, described, mj):
    worst = {"grid_vs_oracle_joint": 0.0, "grid_vs_oracle_target": 0.0,
             "mujoco_vs_oracle_body": 0.0, "mujoco_vs_grid_site": 0.0, "rot": 0.0}
    for group, qs in configs(described).items():
        for q in qs:
            links = oracle.fk(q)
            Tg = grid_fk(robot, q)
            To = oracle.joint_world_transforms(q)
            tg = grid_targets(robot, described, q)
            bodies, sites = mj_fk(mj, q)
            for j, name in enumerate(G1_JOINTS):
                dp, dr = pose_err(Tg[j], To[j])
                worst["grid_vs_oracle_joint"] = max(worst["grid_vs_oracle_joint"], dp); worst["rot"] = max(worst["rot"], dr)
                dp, dr = pose_err(bodies[oracle.joints[name]["child"]], To[j])
                worst["mujoco_vs_oracle_body"] = max(worst["mujoco_vs_oracle_body"], dp)
            for name, T in tg.items():
                child = oracle.joints[dict(TARGETS)[name]]["child"]
                dp, dr = pose_err(T, links[child])
                worst["grid_vs_oracle_target"] = max(worst["grid_vs_oracle_target"], dp); worst["rot"] = max(worst["rot"], dr)
                dp, dr = pose_err(T, sites[MJ_SITE[name]])
                worst["mujoco_vs_grid_site"] = max(worst["mujoco_vs_grid_site"], dp)
    # GRiD vs the numpy oracle: same URDF, same math -> fp64 round-off only.
    assert worst["grid_vs_oracle_joint"] < 1e-12 and worst["grid_vs_oracle_target"] < 1e-12 and worst["rot"] < 1e-7
    # MuJoCo's MJCF stores body orientations as rounded quaternions (6 significant digits), so it is
    # an independent model, not a byte copy: sub-micrometre agreement is the contract, cm is a bug.
    assert worst["mujoco_vs_oracle_body"] < 5e-6 and worst["mujoco_vs_grid_site"] < 5e-6, worst


# --- tree topology -------------------------------------------------------------------------------

def test_ancestor_masks(described, oracle):
    A = oracle.ancestor_matrix()
    for t in described["targets"]:
        assert t["ancestor_mask"] == ANCESTOR_MASK[t["name"]], t["name"]
        assert t["ancestor_mask"] == sum(1 << j for j in range(29) if A[t["anchor_jid"], j])
    for j in range(29):
        expected = sum(1 << k for k, t in enumerate(described["targets"]) if t["ancestor_mask"] >> j & 1)
        assert described["joint_target_mask"][j] == expected


@pytest.mark.parametrize("block,moved", [(range(12, 15), {"left_hand", "right_hand"}),
                                          (range(0, 6), {"left_foot"}), (range(6, 12), {"right_foot"}),
                                          (range(15, 22), {"left_hand"}), (range(22, 29), {"right_hand"})])
def test_branch_isolation(robot, described, block, moved):
    rng = np.random.default_rng(7)
    lo, hi = limits(described).T
    for _ in range(10):
        q0 = rng.uniform(lo, hi)
        q1 = q0.copy(); q1[list(block)] = rng.uniform(lo[list(block)], hi[list(block)])
        a, b = grid_targets(robot, described, q0), grid_targets(robot, described, q1)
        changed = {k for k in a if not np.allclose(a[k], b[k], atol=1e-12, rtol=0)}
        assert changed == moved


# --- validator -----------------------------------------------------------------------------------

def test_tree_validator_accepts_g1(described):
    assert described["kind"] == "fixed_base_tree" and described["base"] == "external"
    assert [j["name"] for j in described["joints"]] == G1_JOINTS
    assert len(described["joints"]) == 29        # no virtual base joints
    assert [t["name"] for t in described["targets"]] == ["left_hand", "right_hand", "left_foot", "right_foot"]
    assert [j["parent"] for j in described["joints"]] == [
        -1, 0, 1, 2, 3, 4, -1, 6, 7, 8, 9, 10, -1, 12, 13, 14, 15, 16, 17, 18, 19, 20, 14, 22, 23, 24, 25, 26, 27]


def test_serial_validator_still_rejects_g1(robot):
    with pytest.raises(ValueError, match="fixed-base serial chain"):
        model.validate_solver_robot(robot, "left_hand_contact_joint")


@pytest.mark.parametrize("targets,match", [
    ([], "1 to 32 targets"),
    ([("a", "left_hand_contact_joint"), ("a", "right_hand_contact_joint")], "unique"),
    ([("a", "left_hand_contact_joint"), ("b", "left_hand_contact_joint")], "unique"),
    ([("a", "no_such_joint")], "no fixed joint"),
    ([("a", "imu_in_pelvis_joint")], "no actuated ancestor"),
    ([("a", "left_hip_pitch_joint")], "no fixed joint"),          # an actuated joint is not a frame
])
def test_tree_validator_rejects_targets(robot, targets, match):
    with pytest.raises(ValueError, match=match):
        model.validate_tree_robot(robot, targets)


def _variant(tmp_path, edit):
    tree = ET.parse(URDF)
    edit(tree.getroot())
    path = tmp_path / "variant.urdf"
    tree.write(path)
    return path


def test_tree_validator_rejects_floating_base():
    with pytest.raises(ValueError, match="fixed-base"):
        model.validate_tree_robot(parse(floating_base=True), TARGETS)


def test_tree_validator_rejects_mimic_and_bad_tool(tmp_path):
    def mimic(root):
        ET.SubElement(root.find("joint[@name='right_wrist_yaw_joint']"), "mimic", joint="left_wrist_yaw_joint")
    with pytest.raises(ValueError, match="independent"):
        model.validate_tree_robot(parse(_variant(tmp_path, mimic)), TARGETS)

    def spin(root):
        root.find("joint[@name='left_foot_contact_joint']/origin").set("rpy", "0.3 0 0")
    rotated = model.describe_tree_robot(parse(_variant(tmp_path, spin)), URDF, TARGETS)
    R = np.array(rotated["targets"][2]["tool_row_major"])[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)     # a rotated tool is kept, not dropped
    assert not np.allclose(R, np.eye(3))
