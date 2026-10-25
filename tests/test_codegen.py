"""Codegen must reject unsupported robots without damaging the build's last good header."""
import importlib.util
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

pytest.importorskip("sympy")
pytest.importorskip("bs4")
pytest.importorskip("lxml")


@pytest.fixture(autouse=True)
def submodule_codegen(monkeypatch):
    """These tests exercise the submodule's codegen; a pip-installed grid_codegen (the [build]
    extra) imported earlier in the session would otherwise shadow it through sys.modules."""
    saved = {name: module for name, module in sys.modules.items()
             if name == "grid_codegen" or name.startswith("grid_codegen.")
             or name == "URDFParser" or name.startswith("URDFParser.")}
    for name in saved:
        monkeypatch.delitem(sys.modules, name)
    yield
    for name in [n for n in sys.modules if n == "grid_codegen" or n.startswith("grid_codegen.")
                 or n == "URDFParser" or n.startswith("URDFParser.")]:
        sys.modules.pop(name, None)
    sys.modules.update(saved)

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


@pytest.mark.parametrize("kind", ["prismatic", "negative_axis"])
def test_general_joint_geometry_is_accepted(tmp_path, kind):
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
    assert run.returncode == 0, run.stderr
    assert output.exists()
    assert "ALL_REVOLUTE_Z = false" in output.with_name("hjcd_joint_model.h").read_text()


def test_explicit_glass_provenance_is_archive_reproducible(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "external/GRiD"))
    monkeypatch.syspath_prepend(str(ROOT / "external/GRiD/external"))
    from grid_codegen.helpers import _lin_alg_helpers as glass
    # Model Git discovery explicitly so this test also runs from a source archive.
    revision = "1" * 40
    monkeypatch.setattr(glass, "_glass_git_head", lambda: revision)
    spec = importlib.util.spec_from_file_location("hjcd_codegen_provenance_test", GEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "grid.cuh"
    monkeypatch.setattr(sys, "argv", [
        str(GEN), str(PANDA), "-o", str(output), "--glass-revision", revision,
    ])
    module.main()
    checkout_bytes = output.read_bytes()
    monkeypatch.setattr(glass, "_glass_git_head", lambda: None)
    module.main()
    assert output.read_bytes() == checkout_bytes


def test_wrong_glass_provenance_preserves_previous_header(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "external/GRiD"))
    monkeypatch.syspath_prepend(str(ROOT / "external/GRiD/external"))
    from grid_codegen.helpers import _lin_alg_helpers as glass
    monkeypatch.setattr(glass, "_glass_git_head", lambda: "1" * 40)
    output = tmp_path / "grid.cuh"
    output.write_text("previous header\n")
    spec = importlib.util.spec_from_file_location("hjcd_codegen_wrong_provenance_test", GEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [
        str(GEN), str(PANDA), "-o", str(output), "--glass-revision", "0" * 40,
    ])
    with pytest.raises(ValueError, match="disagrees with the GLASS checkout"):
        module.main()
    assert output.read_text() == "previous header\n"
    assert not list(tmp_path.glob(".hjcd-codegen-*"))


def test_cmake_rejects_mismatched_same_dof_headers(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    subprocess.run([sys.executable,str(GEN),str(ROOT/"tests/robots/panda_negz.urdf"),
                    "-o",str(model/"grid.cuh")],check=True,capture_output=True,text=True)
    # Both are seven-joint robots; a joint-count assertion cannot catch this mix-up.
    shutil.copy(ROOT/"csrc/generated/hjcd_joint_model.h",model/"hjcd_joint_model.h")
    run = subprocess.run(["cmake","-S",str(ROOT),"-B",str(tmp_path/"build"),
                          "-DBUILD_PYTHON=OFF","-DCMAKE_CUDA_ARCHITECTURES=120",
                          "-DHJCDIK_GRID_HEADER="+str(model/"grid.cuh")],capture_output=True,text=True)
    assert run.returncode != 0
    assert "Mismatched generated robot headers" in run.stdout+run.stderr


def test_punctuated_tool_name_in_static_and_runtime_builds(tmp_path):
    import hjcdik
    source = tmp_path/"punctuated.urdf"
    tree = ET.parse(PANDA)
    tree.getroot().find("joint[@name='panda_grasptarget_hand']").set("name","flange-tool0")
    tree.write(source)
    subprocess.run([sys.executable,str(GEN),str(source),"-t","flange-tool0","-o",str(tmp_path/"grid.cuh")],
                   check=True,capture_output=True,text=True)
    assert "EE_FIXED_FRAME_IDX = 10" in (tmp_path/"grid.cuh").read_text()
    plan = hjcdik.build(source,ee_link="flange-tool0",name="test-punctuated",plan_only=True)
    assert "HJCDIK_EE_FIXED_FRAME_IDX=10" in plan["defines"]
