"""Model identity and CLI boundaries: failures precede import, codegen and GPU work."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest
import hjcdik
from hjcdik import _build, _solver

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("change", ["none", "relocated", "urdf", "spheres", "absent", "tool", "missing-meta"])
def test_no_build_matches_requested_model(tmp_path, monkeypatch, change):
    import grid_rbd
    urdf = tmp_path / "arm.urdf"
    spheres = tmp_path / "spheres.urdf"
    urdf.write_text('<robot name="arm"/>')
    spheres.write_text('<robot name="arm"><!-- spheres --></robot>')
    meta = dict(urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
                code_options=dict(ee_joint_names=["tool"], spherized_urdf=str(spheres),
                                  spherized_urdf_sha256=hashlib.sha256(spheres.read_bytes()).hexdigest()),
                defines=['HJCDIK_EE_TARGET="tool"', 'HJCDIK_ROBOT_NAME="arm"', 'HJCD_HAS_COLLISION=1'])
    record = SimpleNamespace(meta=meta, path=tmp_path/'module.so', header=tmp_path/'grid.cuh',
                             content_key='key', cuda_arch=120)
    monkeypatch.setattr(grid_rbd, "load_artifact", lambda *a, **k: record)
    def unexpected(*args, **kwargs):
        pytest.fail("prepared load invoked compilation, codegen or CUDA")
    monkeypatch.setattr(grid_rbd, "build_artifact", unexpected)
    monkeypatch.setattr(_build, "_prepare_robot", unexpected)
    monkeypatch.setattr(_solver._native, "current_device", unexpected)
    imported = []
    monkeypatch.setattr(_solver, "import_artifact", lambda a: imported.append(a) or 'module')
    if change == 'urdf':
        urdf.write_text('<robot name="arm"><!-- changed limits --></robot>')
    elif change == 'spheres':
        spheres.write_text('<robot name="arm"><!-- changed geometry --></robot>')
    elif change == 'relocated':
        other = tmp_path/'moved'
        other.mkdir()
        moved_urdf, moved_spheres = other/urdf.name, other/spheres.name
        moved_urdf.write_bytes(urdf.read_bytes())
        moved_spheres.write_bytes(spheres.read_bytes())
        urdf, spheres = moved_urdf, moved_spheres
    elif change == 'missing-meta':
        meta.pop('urdf_sha256')
    solver = object.__new__(hjcdik.Solver)
    args = (urdf, 'other-tool' if change == 'tool' else 'tool',
            None if change == 'absent' else spheres, False, tmp_path/'cache')
    if change in ('none', 'relocated'):
        assert solver._resolve(*args) == 'module'
        assert len(imported) == 1
    else:
        with pytest.raises(hjcdik.BuildError, match='requested model.*rebuild'):
            solver._resolve(*args)
        assert not imported
    # Explicit name loading is intentionally independent of the original input files.
    assert hjcdik.load('prepared-name', cache_dir=tmp_path/'cache').content_key == 'key'


@pytest.mark.parametrize("sphere_path", ['csrc/collision/ur5e_spheres.urdf', 'missing.urdf'])
def test_codegen_requires_collision_switch(tmp_path, sphere_path):
    header = tmp_path/'grid.cuh'
    side = tmp_path/'hjcd_joint_model.h'
    header.write_text('old header')
    side.write_text('old joint model')
    run = subprocess.run([sys.executable, str(ROOT/'scripts/codegen/generate_grid.py'),
                          str(ROOT/'csrc/urdf/ur5e.urdf'), '-t', 'flange-tool0',
                          '--spherized-urdf', str(ROOT/sphere_path), '-o', str(header)],
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 2
    assert '--spherized-urdf requires --collision' in run.stderr
    assert header.read_text() == 'old header'
    assert side.read_text() == 'old joint model'


@pytest.mark.parametrize('collision', [False, True])
def test_real_prepared_model_load(tmp_path, monkeypatch, collision):
    import grid_rbd
    source = tmp_path/'arm.urdf'
    source.write_bytes((ROOT/'examples/mixed_chain.urdf').read_bytes())
    spheres = tmp_path/'spheres.urdf'
    spheres.write_bytes(source.read_bytes())
    sphere_arg = spheres if collision else None
    artifact = hjcdik.build(source, ee_link='tool', spheres=sphere_arg, cache_dir=tmp_path/'cache')
    def no_build(*a, **k):
        pytest.fail('prepared loading requires no compiler or generator')
    monkeypatch.setattr(grid_rbd, 'build_artifact', no_build)
    monkeypatch.setattr(_build, '_prepare_robot', no_build)
    relocated = tmp_path/'relocated.urdf'
    relocated.write_bytes(source.read_bytes())
    solver = hjcdik.Solver(relocated, ee_link='tool', spheres=sphere_arg,
                           allow_build=False, cache_dir=tmp_path/'cache')
    assert solver.info.artifact == str(artifact.path)
    assert solver.info.joint_limits[1][1] == .5
    relocated.write_text(relocated.read_text().replace('upper="0.5"', 'upper="1.5"'))
    with pytest.raises(hjcdik.BuildError, match='requested model'):
        hjcdik.Solver(relocated, ee_link='tool', spheres=sphere_arg, allow_build=False, cache_dir=tmp_path/'cache')
    assert hjcdik.load(artifact.name, cache_dir=tmp_path/'cache').content_key == artifact.content_key


def test_codegen_collision_positive(tmp_path):
    header = tmp_path/'grid.cuh'
    run = subprocess.run([sys.executable, str(ROOT/'scripts/codegen/generate_grid.py'),
                          str(ROOT/'csrc/urdf/ur5e.urdf'), '-t', 'flange-tool0', '--collision',
                          '--spherized-urdf', str(ROOT/'csrc/collision/ur5e_spheres.urdf'),
                          '-o', str(header)], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout+run.stderr
    assert '#define HJCD_HAS_COLLISION 1' in header.read_text()
    assert 'namespace grid_collision' in header.read_text()
