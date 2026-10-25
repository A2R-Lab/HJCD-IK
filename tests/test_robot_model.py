"""CPU-only checks for the shared gate, including the runtime-build boundary."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("robot_model_test", ROOT / "hjcdik/_robot_model.py")
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


def fake_robot(n=7, kind="revolute", axis=(0, 0, 1), limits=(-1, 1), mimic=False,
               serial=True, floating=False, target_parent=None):
    joints = [SimpleNamespace(name=f"joint{i}", jid=i, jtype=kind, is_mimic=mimic,
                              get_joint_limits=lambda: limits) for i in range(n)]
    return SimpleNamespace(
        floating_base=floating, is_serial_chain=lambda: serial, get_num_joints=lambda: n,
        get_joints_ordered_by_id=lambda: joints, get_joint_by_id=lambda i: joints[i],
        get_S_by_id=lambda i: np.array([0, 0, 0, *axis] if kind == "prismatic" else [*axis, 0, 0, 0]),
        get_fixed_joint_by_name=lambda name: SimpleNamespace(
            parent_name=target_parent or f"joint{n-1}") if name == "tool" else None)


@pytest.mark.parametrize("n", [1, 7, 32])
def test_supported_joint_count(n):
    model.validate_solver_robot(fake_robot(n=n), "tool")


@pytest.mark.parametrize("kwargs,match", [
    ({"n": 0}, "1 to 32"), ({"n": 33}, "1 to 32"),
    ({"serial": False}, "serial"), ({"floating": True}, "fixed-base"),
    ({"mimic": True}, "independent"), ({"kind": "spherical"}, "independent"),
    ({"axis": (0, 0, 0)}, "unit axis"),
    ({"limits": (1, 1)}, "limits"), ({"limits": (2, 1)}, "limits"),
    ({"limits": (-np.inf, 1)}, "limits"), ({"limits": (0, np.nan)}, "limits"),
    ({"target_parent": "joint0"}, "final actuated"),
])
def test_reject_unsupported_model(kwargs, match):
    with pytest.raises(ValueError, match=match):
        model.validate_solver_robot(fake_robot(**kwargs), "tool")


def test_continuous_sampling_contract_and_missing_target():
    model.validate_solver_robot(fake_robot(kind="continuous", limits=(-np.inf, np.inf)), "tool")
    with pytest.raises(ValueError, match="fixed joint"):
        model.validate_solver_robot(fake_robot(), "missing")


@pytest.mark.parametrize("change,match", [("limit", "limits"),
                                           ("target", "final actuated")])
def test_runtime_plan_rejects_before_codegen(tmp_path, monkeypatch, change, match):
    import hjcdik
    from grid_codegen import GRiDCodeGenerator
    tree = ET.parse(ROOT / "csrc/urdf/panda.urdf")
    first = tree.getroot().find("joint[@name='panda_joint1']")
    if change == "limit":
        first.find("limit").set("upper", first.find("limit").get("lower"))
    else:
        ET.SubElement(tree.getroot(), "link", name="early_tool")
        fixed = ET.SubElement(tree.getroot(), "joint", name="early_tool", type="fixed")
        ET.SubElement(fixed, "parent", link="panda_link1")
        ET.SubElement(fixed, "child", link="early_tool")
    path = tmp_path / "invalid.urdf"
    tree.write(path)
    monkeypatch.setattr(GRiDCodeGenerator, "gen_all_code", lambda *a, **k: pytest.fail("generated invalid robot"))
    with pytest.raises(hjcdik.BuildError, match=match):
        hjcdik.build(path, ee_link="early_tool" if change == "target" else "panda_grasptarget_hand",
                     plan_only=True)


@pytest.mark.parametrize("kind,axis", [("revolute", (0, 0, -1)), ("prismatic", (0.6, 0.8, 0))])
def test_general_axes_and_types(kind, axis):
    model.validate_solver_robot(fake_robot(kind=kind, axis=axis), "tool")


def test_side_header_identity_is_stable_and_detects_corruption(tmp_path):
    r = fake_robot(kind="prismatic", axis=(0.6, 0.8, 0))
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("test model")
    description = model.describe_robot(r, urdf, "tool")
    original_axis = description["joints"][0]["axis"][:]
    path = model.stage_joint_header(description, tmp_path)
    assert model.stage_joint_header(description, tmp_path) == path
    assert "T(0.59999999999999998)" in path.read_text()
    assert "ALL_REVOLUTE_Z = false" in path.read_text()
    description["joints"][0]["axis"] = [0, 0, 1]
    assert model.stage_joint_header(description, tmp_path) != path
    path.write_text("corrupt")
    description["joints"][0]["axis"] = original_axis
    with pytest.raises(ValueError, match="corrupt"):
        model.stage_joint_header(description, tmp_path)
