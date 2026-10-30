"""Actionable build/load failures without corrupting caches or provoking GPU faults."""
import sys
from types import SimpleNamespace

import pytest
import hjcdik
from hjcdik import _build


@pytest.mark.parametrize("robot", [None, "panda"])
@pytest.mark.parametrize("spheres", ["panda_foam", "missing-spheres.urdf", ""])
def test_packaged_solver_rejects_sphere_override_before_gpu(robot, spheres, monkeypatch):
    from hjcdik import _solver
    def unexpected(*args, **kwargs):
        pytest.fail("invalid packaged override reached a build or CUDA")
    monkeypatch.setattr(_solver, "_build_artifact", unexpected)
    monkeypatch.setattr(_solver, "_load_artifact", unexpected)
    monkeypatch.setattr(_solver._native, "current_device", unexpected)
    with pytest.raises(ValueError, match="spheres.*packaged.*URDF"):
        hjcdik.Solver(robot, spheres=spheres)


def test_missing_runtime_dependency_names_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "grid_rbd", None)
    with pytest.raises(hjcdik.BuildError, match=r"pip install 'hjcdik\[build\]'"):
        hjcdik.load("missing-dependency-fixture")


def test_missing_pybind_names_first_build_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "pybind11", None)
    with pytest.raises(hjcdik.BuildError, match=r"pybind11.*hjcdik\[build\]"):
        hjcdik.build(hjcdik.presets()["panda"]["urdf"], ee_link="panda_grasptarget_hand", plan_only=True)


def test_missing_compiler_names_build_and_cause(monkeypatch):
    import grid_rbd
    monkeypatch.setattr(_build, "_build_inputs", lambda *a, **k: {})
    def missing(*args, **kwargs):
        raise FileNotFoundError("nvcc not found; install the CUDA toolkit")
    monkeypatch.setattr(grid_rbd, "build_artifact", missing)
    with pytest.raises(hjcdik.BuildError, match="build of 'no-compiler'.*nvcc"):
        hjcdik.build(hjcdik.presets()["panda"]["urdf"], ee_link="panda_grasptarget_hand", name="no-compiler")


@pytest.mark.parametrize("failure", ["missing", "damaged", "abi"])
def test_prepared_import_failure_is_actionable(tmp_path, monkeypatch, failure):
    artifact = _build.Artifact("broken-fixture", "panda", "tool", False,
                               tmp_path / "broken.so", tmp_path / "grid.cuh", "fixture-key", 120,
                               "_hjcdik_failure_fixture", False)
    monkeypatch.setattr(_build, "_LOADED", {})
    if failure == "damaged":
        artifact.path.write_bytes(b"not an ELF extension")
    elif failure == "abi":
        monkeypatch.setattr(_build.importlib.util, "module_from_spec", lambda spec: SimpleNamespace(BINDING_ABI=-1))
        monkeypatch.setattr(_build.importlib.machinery.ExtensionFileLoader, "exec_module", lambda *a: None)
    with pytest.raises(hjcdik.BuildError, match="broken-fixture.*force_rebuild=True"):
        _build.import_artifact(artifact)
    assert not _build._LOADED and artifact.module_name not in sys.modules
