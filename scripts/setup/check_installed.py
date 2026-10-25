#!/usr/bin/env python3
"""Explicit sdist -> wheel -> isolated public-API integration check (no timing).

Run with the development venv and initialized submodules:
  .venv/bin/python scripts/setup/check_installed.py --out /tmp/hjcd-installed-UNIQUE
Needs pip access (or a populated pip cache), nvcc and a CUDA GPU. Never modifies
the development environment. Keeps artifacts/logs for inspection; use a new output
directory after interruption. Ctrl-C stops the foreground command; no background jobs.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[2]


def installed_check(work, phase, expected_key=None):
    """Runs in the isolated interpreter, with only copied example assets beside it."""
    import runpy
    import numpy as np
    import hjcdik
    package = Path(hjcdik.__file__).resolve()
    assert package.is_relative_to(work / "venv"), package
    report = dict(package=str(package), python=sys.version, phase=phase)
    if phase == "panda":
        import importlib.util
        assert importlib.util.find_spec("grid_rbd") is None
        assert importlib.util.find_spec("pybind11") is None
        for robot in (None, "panda"):
            try:
                hjcdik.Solver(robot, spheres="missing-spheres.urdf")
            except ValueError as error:
                assert "packaged" in str(error) and "URDF" in str(error)
            else:
                raise AssertionError("packaged robot silently ignored a sphere override")
        solver = hjcdik.Solver()
        result = solver.solve(solver.sample_targets(4, seed=5), collision=None, batch_size=2000)
        assert result.ok(5, .001).any(axis=1).all()
        report["accurate"] = 4
    else:
        if phase == "load":
            import grid_rbd
            def no_build(*args, **kwargs):
                raise AssertionError("prepared load attempted to build")
            grid_rbd.build_artifact = no_build
            hjcdik.build = no_build
        example = runpy.run_path(str(work / "rail_ur5e.py"))
        cells = {}
        for precision in ("fp32", "fp64"):
            artifact, solver, results = example["run"](work / "cache", precision, load_only=phase == "load")
            assert solver.info.joint_units == ("m",) + ("rad",)*6
            if expected_key is not None:
                assert artifact.content_key == expected_key and not artifact.compiled
            for scene, rows in results.items():
                assert rows["warm"] is not None
                cold = int(rows["out"].ok(5, .001).any(axis=1).sum())
                warm = int(rows["warm"].ok(5, .001).any(axis=1).sum())
                assert cold >= 31 and warm >= 31
                assert rows["blocked"].count.tolist() == [0]
                assert not rows["blocked"].valid.any() and np.isnan(rows["blocked"].q).all()
                assert rows["restored"].ok(5, .001).any()
                cells[f"{scene}-{precision}"] = dict(cold=cold, warm=warm, blocked=0)
        report.update(content_key=artifact.content_key, artifact=str(artifact.path), cells=cells)
    report["versions"] = {name: metadata.version(name) for name in ("hjcdik", "numpy")}
    if phase != "panda":
        report["versions"]["grid-rbd"] = metadata.version("grid-rbd")
    (work / f"{phase}.json").write_text(json.dumps(report, indent=2) + "\n")


def run(out):
    out = out.resolve()
    if out.exists():
        raise SystemExit("--out must be a new directory; previous evidence is never overwritten")
    if out.is_relative_to(ROOT):
        raise SystemExit("--out must be outside the checkout to test import isolation")
    out.mkdir(parents=True)
    env = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    env["PYTHONNOUSERSITE"] = "1"
    commands = []
    def command(label, args, cwd=out):
        args = list(map(str, args))
        commands.append(dict(label=label, argv=args, cwd=str(cwd)))
        (out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        print(label, flush=True)
        with (out / f"{label}.log").open("w") as log:
            subprocess.run(args, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    dist = out / "dist"; dist.mkdir()
    command("sdist", [sys.executable, "-c", "from scikit_build_core.build import build_sdist; import sys; build_sdist(sys.argv[1])", dist], ROOT)
    archives = list(dist.glob("*.tar.gz")); assert len(archives) == 1
    command("wheel", [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", archives[0], "-w", dist])
    wheels = list(dist.glob("*.whl")); assert len(wheels) == 1
    venv.EnvBuilder(with_pip=True).create(out / "venv")
    python = out / "venv/bin/python"
    command("install-base", [python, "-m", "pip", "install", wheels[0], "numpy=="+metadata.version("numpy")])
    for source in (Path(__file__), ROOT / "examples/rail_ur5e.py", ROOT / "examples/rail_ur5e_targets.json"):
        shutil.copy2(source, out / source.name)
    child = [python, "-I", out / Path(__file__).name, "--out", out, "--child"]
    command("panda", [*child, "panda"])
    command("install-build", [python, "-m", "pip", "install", "grid-rbd=="+metadata.version("grid-rbd"),
                               "pybind11=="+metadata.version("pybind11")])
    command("build", [*child, "build"])
    key = json.loads((out / "build.json").read_text())["content_key"]
    command("load", [*child, "load", "--expected-key", key])
    files = [*archives, *wheels, out / "rail_ur5e_targets.json", *(out / "cache/example-inputs").iterdir()]
    report = dict(status="passed", git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  git_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
                  hashes={str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                  phases={name: json.loads((out / f"{name}.json").read_text()) for name in ("panda", "build", "load")})
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Installed-package checks passed: {out / 'report.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--child", choices=("panda", "build", "load"), help=argparse.SUPPRESS)
    parser.add_argument("--expected-key", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        installed_check(args.out.resolve(), args.child, args.expected_key)
    else:
        run(args.out)
