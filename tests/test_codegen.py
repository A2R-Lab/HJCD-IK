"""Codegen must reject unsupported robots without damaging the build's last good header."""
import importlib.util
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


def test_generation_failure_after_partial_write_preserves_header(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "external/GRiD"))
    monkeypatch.syspath_prepend(str(ROOT / "external/GRiD/external"))
    from grid_codegen import GRiDCodeGenerator

    def fail_during_generation(self, **kwargs):
        Path(kwargs["output_path"]).write_text("incomplete header")
        raise RuntimeError("injected generation failure")

    monkeypatch.setattr(GRiDCodeGenerator, "gen_all_code", fail_during_generation)
    output = tmp_path / "grid.cuh"
    output.write_text("previous buildable header\n")
    monkeypatch.setattr(sys, "argv", [str(GEN), str(PANDA), "-o", str(output)])
    spec = importlib.util.spec_from_file_location("hjcd_codegen_test", GEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(RuntimeError, match="injected generation failure"):
        module.main()
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
