"""Runtime robot builds over GRiD's consumer-artifact store (Stage 2).

The first run compiles two artifacts (a second Panda and the Fetch arm) with nvcc, a few minutes;
later runs reuse the store under ~/.cache (or HJCDIK_TEST_CACHE). GRiD's store rules (identity,
object reuse, atomic publish) are GRiD's to test; here we test what HJCD owns: the inputs we pass,
the load path, isolation between modules, and the error contract.
"""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

hjcdik = pytest.importorskip("hjcdik")
grid_rbd = pytest.importorskip("grid_rbd")

ROOT = Path(__file__).resolve().parents[1]
PANDA = ROOT / "csrc/urdf/panda.urdf"
FETCH = ROOT / "csrc/urdf/fetch.urdf"
CACHE = os.environ.get("HJCDIK_TEST_CACHE") or None


@pytest.fixture(scope="module")
def panda_rt():
    assert shutil.which("nvcc") or Path("/usr/local/cuda/bin/nvcc").exists(), "runtime builds need nvcc"
    return hjcdik.build(PANDA, ee_link="panda_grasptarget_hand", spheres="panda_foam", name="test-panda-cc",
                        cache_dir=CACHE)


@pytest.fixture(scope="module")
def fetch_rt():
    return hjcdik.build(FETCH, ee_link="ee_fixed", name="test-fetch", cache_dir=CACHE)


def test_plan_reports_identity_without_building():
    plan = hjcdik.build(PANDA, ee_link="panda_grasptarget_hand", spheres="panda_foam",
                        name="test-plan-only", cache_dir=CACHE, plan_only=True)
    assert plan["consumer"] == "hjcdik" and plan["name"] == "test-plan-only"
    assert any(d.startswith("HJCDIK_EE_FIXED_FRAME_IDX=10") for d in plan["defines"])
    assert any(d.startswith("HJCDIK_MODULE_NAME=_hjcdik_") for d in plan["defines"])
    assert "HJCD_HAS_COLLISION=1" in plan["defines"]
    assert plan["code_options"]["spherized_urdf"].endswith("panda_foam_rebound.urdf")
    assert {Path(s).name for s in plan["sources"]} == {"hjcd_kernel.cu", "pybind_module.cpp"}
    assert "hjcd_settings.h" in {Path(h).name for h in plan["headers"]}


def test_identity_changes_with_every_input_class(tmp_path):
    def key(**kwargs):
        args = dict(urdf=PANDA, ee_link="panda_grasptarget_hand", spheres="panda_foam", name="test-keys",
                    cache_dir=CACHE, plan_only=True)
        args.update(kwargs)
        return hjcdik.build(args.pop("urdf"), **args)["input_key"]

    base = key()
    assert key() == base
    urdf = tmp_path / "panda.urdf"
    urdf.write_text(PANDA.read_text().replace("<robot name=\"panda\"", "<robot name=\"panda_edit\"", 1))
    assert key(urdf=urdf) != base                                   # URDF bytes
    spheres = tmp_path / "spheres.urdf"
    spheres.write_text(hjcdik.sphere_presets()["panda_foam"].read_text().replace("0.08", "0.081", 1))
    assert key(spheres=spheres) != base                             # sphere model bytes
    assert key(spheres=None) != base                                # collision option and define
    assert key(ee_link="panda_hand_joint") != base                  # target frame define and codegen option
    assert key(name="test-keys-2") != base                          # module name define


def test_runtime_panda_matches_the_packaged_build(panda_rt):
    assert panda_rt.robot == "panda" and panda_rt.ee_frame == "panda_grasptarget_hand" and panda_rt.collision_enabled
    packaged = hjcdik.Solver()
    runtime = hjcdik.Solver("test-panda-cc", cache_dir=CACHE)
    assert runtime.info.robot == "panda" and runtime.info.dof == 7 and runtime.info.collision_enabled
    assert runtime.info.artifact == str(panda_rt.path) and packaged.info.artifact is None
    assert runtime.info.grid_header_sha256 == panda_rt.grid_header_sha256
    assert runtime.info.ee_frame == packaged.info.ee_frame
    targets = packaged.sample_targets(4, seed=21)
    np.testing.assert_allclose(runtime.sample_targets(4, seed=21), targets, atol=1e-9)
    a = packaged.solve(targets, batch_size=512, num_solutions=1, collision=None)
    b = runtime.solve(targets, batch_size=512, num_solutions=1, collision=None)
    assert a.count.tolist() == b.count.tolist() == [1, 1, 1, 1]
    assert (a.pos_err_mm < 0.01).all() and (b.pos_err_mm < 0.01).all()
    runtime.set_world(spheres=[[0, 0, 0, 100]])
    assert runtime.solve(targets[0], batch_size=128).count.tolist() == [0]


def test_load_needs_no_toolchain_and_names_the_build_command(panda_rt, monkeypatch):
    monkeypatch.setattr(grid_rbd, "build_artifact", lambda *a, **k: pytest.fail("load compiled"))
    loaded = hjcdik.load("test-panda-cc", cache_dir=CACHE)
    assert loaded.content_key == panda_rt.content_key and not loaded.compiled
    assert loaded.collision_enabled and loaded.ee_frame == "panda_grasptarget_hand"
    with pytest.raises(hjcdik.BuildError, match="hjcdik build"):
        hjcdik.load("no-such-build", cache_dir=CACHE)
    with pytest.raises(hjcdik.BuildError, match="hjcdik build"):
        hjcdik.Solver("no-such-build", cache_dir=CACHE)
    with pytest.raises(hjcdik.BuildError):
        hjcdik.Solver(str(PANDA), ee_link="panda_hand_joint", spheres=None, allow_build=False, cache_dir=CACHE)
    assert any(e["name"] == "test-panda-cc" for e in hjcdik.list_builds(cache_dir=CACHE))


def test_solver_argument_contract():
    with pytest.raises(ValueError, match="ee_link"):
        hjcdik.Solver(str(PANDA))
    with pytest.raises(ValueError, match="build name"):
        hjcdik.Solver("test-panda-cc", ee_link="x")
    with pytest.raises(ValueError, match="packaged"):
        hjcdik.Solver("panda", ee_link="panda_hand_joint")
    with pytest.raises(TypeError):
        hjcdik.Solver(42)
    with pytest.raises(hjcdik.BuildError, match="preset"):
        hjcdik.build(PANDA, ee_link="panda_grasptarget_hand", spheres="no-such-preset", plan_only=True)
    with pytest.raises(hjcdik.BuildError, match="fixed joint"):
        hjcdik.build(PANDA, ee_link="not_a_joint", plan_only=True)


def test_two_robots_in_one_process_are_isolated(panda_rt, fetch_rt):
    assert fetch_rt.robot == "fetch_arm_no_helpers" and not fetch_rt.collision_enabled
    fetch = hjcdik.Solver("test-fetch", cache_dir=CACHE)
    assert fetch.info.robot == "fetch_arm_no_helpers" and fetch.info.ee_frame == "ee_fixed"
    assert not fetch.info.collision_enabled
    with pytest.raises(ValueError, match="no collision geometry"):
        fetch.set_world(spheres=[[0, 0, 0, 1]])
    assert fetch.solve(fetch.sample_targets(2, seed=3), batch_size=512).count.tolist() == [1, 1]   # default None
    packaged = hjcdik.Solver()
    runtime = hjcdik.Solver("test-panda-cc", cache_dir=CACHE)
    # Each module keeps its own kinematics: a second robot must not solve with the first robot's cached
    # model (template statics bound process-wide once; hidden visibility and -fno-gnu-unique prevent it).
    fetch_targets = fetch.sample_targets(4, seed=3)
    assert not np.allclose(fetch_targets[:, :3], packaged.sample_targets(4, seed=3)[:, :3])
    assert (fetch.solve(fetch_targets, batch_size=512, collision=None).pos_err_mm[:, 0] < 0.01).all()
    target = packaged.sample_targets(1, seed=5)[0]
    packaged.set_world(spheres=[[0, 0, 0, 100]])
    runtime.set_world(spheres=[[10, 10, 10, 0.1]])
    for _ in range(2):
        assert packaged.solve(target, batch_size=128).count.tolist() == [0]
        assert runtime.solve(target, batch_size=128).count.tolist() == [1]
        assert fetch.solve(fetch.sample_targets(1, seed=4)[0], batch_size=128).count.tolist() == [1]


def test_damaged_entry_is_rebuilt(panda_rt):
    artifact = Path(panda_rt.path)
    assert artifact.exists()
    artifact.rename(artifact.with_suffix(".so.moved"))
    try:
        with pytest.raises(hjcdik.BuildError, match="cannot be used here"):
            hjcdik.load("test-panda-cc", cache_dir=CACHE)
        rebuilt = hjcdik.build(PANDA, ee_link="panda_grasptarget_hand", spheres="panda_foam",
                               name="test-panda-cc", cache_dir=CACHE)
        assert rebuilt.content_key == panda_rt.content_key and Path(rebuilt.path).exists() and rebuilt.compiled
    finally:
        moved = artifact.with_suffix(".so.moved")
        if moved.exists() and not artifact.exists():
            moved.rename(artifact)
        elif moved.exists():
            moved.unlink()
    assert hjcdik.Solver("test-panda-cc", cache_dir=CACHE).solve(
        hjcdik.Solver().sample_targets(1, seed=1)[0], batch_size=128, collision=None).count.tolist() == [1]
