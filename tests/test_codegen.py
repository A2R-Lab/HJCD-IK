"""Codegen must reject unsupported robots without damaging the build's last good header."""
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

pytest.importorskip("sympy")
pytest.importorskip("bs4")
pytest.importorskip("lxml")

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "scripts/codegen/generate_grid.py"
PANDA = ROOT / "csrc/urdf/panda.urdf"


@pytest.mark.parametrize("flags,match", [
    (["--floating-base"], "fixed-base serial chain"),
    (["--namespace", "custom"], "grid namespace"),
    (["-t", "missing_end_effector"], None),
])
def test_failed_codegen_preserves_previous_header(tmp_path, flags, match):
    output = tmp_path / "grid.cuh"
    output.write_text("previous buildable header\n")
    run = subprocess.run(
        [sys.executable, str(GEN), str(PANDA), "-o", str(output), *flags],
        capture_output=True, text=True, timeout=60,
    )
    assert run.returncode != 0
    if match:
        assert match in run.stderr
    assert output.read_text() == "previous buildable header\n"
    assert not list(tmp_path.glob(".hjcd-codegen-*"))


@pytest.mark.parametrize("kind,match", [
    ("prismatic", "independent revolute"),
    ("negative_axis", "local +Z"),
])
def test_unsupported_joint_geometry_is_rejected(tmp_path, kind, match):
    tree = ET.parse(PANDA)
    joint = tree.getroot().find("joint[@name='panda_joint1']")
    assert joint is not None
    if kind == "prismatic":
        joint.set("type", "prismatic")
    else:
        joint.find("axis").set("xyz", "0 0 -1")
    urdf = tmp_path / "unsupported.urdf"
    tree.write(urdf)
    output = tmp_path / "grid.cuh"
    run = subprocess.run(
        [sys.executable, str(GEN), str(urdf), "-o", str(output)],
        capture_output=True, text=True, timeout=60,
    )
    assert run.returncode != 0
    assert match in run.stderr
    assert not output.exists()
