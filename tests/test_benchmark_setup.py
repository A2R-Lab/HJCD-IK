"""Benchmark setup contracts only: never time or invoke a solver here."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bench():
    spec = importlib.util.spec_from_file_location("hjcd_bench_setup", ROOT / "benchmark/hjcd_ik_bench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("collision", [False, True])
def test_benchmark_delegates_to_canonical_codegen(bench, monkeypatch, collision):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kw: calls.append((command, kw)))
    assert bench.run_grid_codegen(Path("csrc/urdf/panda.urdf"), False, collision=collision)
    command, options = calls.pop()
    assert command[:3] == [sys.executable, str(ROOT / "scripts/codegen/generate_grid.py"),
                           str(ROOT / "csrc/urdf/panda.urdf")]
    assert options == {"cwd": ROOT, "check": True}
    assert ("--collision" in command) == collision
    assert ("--spherized-urdf" in command) == collision


def test_benchmark_codegen_skip_and_custom_target(bench, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kw: calls.append(command))
    assert not bench.run_grid_codegen(Path("missing.urdf"), True)
    assert not calls
    with pytest.raises(ValueError, match="grid-target"):
        bench.run_grid_codegen(Path("csrc/urdf/fetch.urdf"), False)
    assert not calls
    assert bench.run_grid_codegen(Path("csrc/urdf/fetch.urdf"), False, "ee_fixed")
    assert calls[0][-2:] == ["-t", "ee_fixed"]


def test_benchmark_codegen_failure_propagates(bench, monkeypatch):
    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(2, command)
    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        bench.run_grid_codegen(Path("csrc/urdf/panda.urdf"), False)


@pytest.mark.parametrize("regenerate", [False, True])
def test_paper_workflow_selects_collision_builds_without_running_benchmarks(tmp_path, regenerate):
    script = tmp_path / "scripts/bench/run_paper_experiments.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts/bench/run_paper_experiments.sh", script)
    # All child Python/build commands are recorders. The shell control flow is real,
    # but no CUDA, compilation, benchmark or external solver is executed.
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    recorder = "#!" + sys.executable + "\n" + (
        "import json, os, sys\n"
        "with open(os.environ['SETUP_COMMAND_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    for name in ("python", "bash"):
        executable = fake_bin / name
        executable.write_text(recorder)
        executable.chmod(0o755)
    log = tmp_path / "commands.jsonl"
    env = {**os.environ, "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
           "PYTHON": str(fake_bin / "python"), "SETUP_COMMAND_LOG": str(log),
           "HJCD_REGEN": "1" if regenerate else "0", "SKIP_HJCD": "0",
           "SKIP_PYROKI": "1", "SKIP_CUROBO": "1", "SKIP_IKFLOW": "1",
           "RUN_FETCH": "0", "RUN_DOF": "0", "RUN_MMD": "0",
           "OUT_DIR": str(tmp_path / "results")}
    run = subprocess.run(["/bin/bash", str(script)], env=env,
                         capture_output=True, text=True, timeout=30)
    if not regenerate:
        assert run.returncode == 2
        assert "HJCD_REGEN=1" in run.stderr
        assert not log.exists()
        return
    assert run.returncode == 0, run.stdout + run.stderr
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    generated = [cmd for cmd in commands if cmd[0] == "scripts/codegen/generate_grid.py"]
    assert len(generated) == 3  # open-world, collision scene, restored default
    assert generated[0][-1] == "panda_hand_joint"
    for cmd in generated[1:]:
        assert "panda_grasptarget_hand" in cmd and "--collision" in cmd
        assert cmd[-2:] == ["--spherized-urdf",
                           "external/foam/assets/panda/smaller_panda_spherized.urdf"]
