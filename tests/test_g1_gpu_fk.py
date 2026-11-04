"""G1 tree-model GPU target FK and prepared-artifact identity (humanoid-v1 M2b). Needs a CUDA GPU and nvcc.

The artifact is built through hjcdik._tree_build into HJCD_TREE_CACHE (default: a pytest temp dir), never
into the packaged Panda module. References: the M2a numpy URDF oracle (tests/urdf_fk.py), GRiD's parsed
model composed on the CPU (test_g1_model.grid_fk) and MuJoCo on CRAG's MJCF (optional).
Tolerances: fp64 1e-9 m / 1e-9 Frobenius; fp32 2e-5 m / 2e-5 Frobenius; MuJoCo 5e-6 (MJCF rounds quaternions).
"""
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

import test_g1_model as g1
from test_g1_model import mj  # noqa: F401  (shared MuJoCo fixture)

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "csrc/urdf/g1_29dof_contacts.urdf"
SPEC = ROOT / "csrc/urdf/g1_29dof_contacts.targets.json"
NAME = "g1-contacts-tree"


def _need_toolchain():
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc not on PATH")
    try:
        import pybind11  # noqa: F401
        import grid_rbd  # noqa: F401
    except ImportError:
        pytest.skip("runtime builds need grid_rbd and pybind11")


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    _need_toolchain()
    return os.environ.get("HJCD_TREE_CACHE") or str(tmp_path_factory.mktemp("tree-store"))


@pytest.fixture(scope="module")
def tb():
    from hjcdik import _tree_build
    return _tree_build


@pytest.fixture(scope="module")
def artifact(tb, store):
    return tb.build_tree(URDF, SPEC, name=NAME, cache_dir=store)


@pytest.fixture(scope="module")
def fk(tb, artifact):
    return tb.import_tree(artifact)


@pytest.fixture(scope="module")
def described():
    return g1.model.describe_tree_robot(g1.parse(), URDF, g1.TARGETS)


@pytest.fixture(scope="module")
def batch(described):
    groups = g1.configs(described)
    lo, hi = g1.limits(described).T
    rng = np.random.default_rng(20261103)
    arms = []
    for block in (range(15, 22), range(22, 29), range(0, 6), range(6, 12)):
        q = np.zeros((5, 29)); q[:, list(block)] = rng.uniform(lo[list(block)], hi[list(block)], (5, len(block)))
        arms.append(q)
    alternating = np.where(np.arange(29) % 2 == 0, lo, hi)
    groups["limits"] = [lo, hi, alternating, np.where(np.arange(29) % 2 == 0, hi, lo)]
    groups["single_limbs"] = list(np.concatenate(arms))
    names = [n for n, qs in groups.items() for _ in qs]
    return np.ascontiguousarray(np.concatenate([np.asarray(qs) for qs in groups.values()])), names


def reference(robot, described, Q):
    return np.stack([[g1.grid_targets(robot, described, q)[n] for n in ("left_hand", "right_hand", "left_foot", "right_foot")]
                     for q in Q])


def errors(A, B):
    return (float(np.max(np.linalg.norm(A[..., :3, 3] - B[..., :3, 3], axis=-1))),
            float(np.max(np.linalg.norm(A[..., :3, :3] - B[..., :3, :3], axis=(-2, -1)))))


# --- metadata ------------------------------------------------------------------------------------

def test_metadata_matches_validated_model(fk, described):
    assert fk.ROBOT == "g1_29dof_rev_1_0" and fk.NUM_JOINTS == 29 and fk.NUM_TARGETS == 4
    assert list(fk.TARGET_NAMES) == ["left_hand", "right_hand", "left_foot", "right_foot"]
    assert list(fk.TARGET_FRAMES) == [f for _, f in g1.TARGETS]
    assert list(fk.TARGET_ANCHOR_JID) == [21, 28, 5, 11]
    assert [fk.TARGET_ANCESTOR_MASK[k] for k in range(4)] == [g1.ANCESTOR_MASK[n] for n in fk.TARGET_NAMES]
    assert list(fk.JOINT_TARGET_MASK) == described["joint_target_mask"]
    assert list(fk.JOINT_PARENT) == [j["parent"] for j in described["joints"]]
    assert fk.MODEL_SHA256 == g1.model.model_sha256(described)
    tools = np.asarray(fk.TARGET_TOOL_COLUMN_MAJOR).reshape(4, 4, 4).transpose(0, 2, 1)   # column-major -> row-major
    np.testing.assert_array_equal(tools, [t["tool_row_major"] for t in described["targets"]])
    assert list(fk.TARGET_FIXED_FRAME_IDX) == [32, 29, 38, 36]


def test_generated_header_is_the_g1_tree(artifact):
    text = artifact.header.read_text()
    assert "const int NUM_JOINTS = 29;" in text and "const int XHOM_T_COUNT = 656;" in text
    assert g1.model.tree_fixed_frame_index(text, "left_hand_contact_joint") == 32   # the build's named target


# --- numerics ------------------------------------------------------------------------------------

def test_gpu_fk_fp64_matches_cpu_references(fk, described, batch, mj):
    Q, names = batch
    assert len(Q) >= 200
    gpu, _ = fk.target_poses_f64(Q)
    assert gpu.shape == (len(Q), 4, 4, 4) and gpu.dtype == np.float64
    np.testing.assert_array_equal(gpu[..., 3, :], np.broadcast_to([0, 0, 0, 1], gpu[..., 3, :].shape))
    ref = reference(g1.parse(), described, Q)
    dp, dR = errors(gpu, ref)
    assert dp <= 1e-9 and dR <= 1e-9, (dp, dR)
    oracle = g1.UrdfFK(URDF)
    frames = {n: oracle.joints[f]["child"] for n, f in g1.TARGETS}
    for i in range(0, len(Q), 7):
        links = oracle.fk(Q[i])
        o = np.stack([links[frames[n]] for n in fk.TARGET_NAMES])
        dp, dR = errors(gpu[i], o)
        assert dp <= 1e-9 and dR <= 1e-9, (names[i], dp, dR)
    for i in range(0, len(Q), 5):
        _, sites = g1.mj_fk(mj, Q[i])
        m = np.stack([sites[g1.MJ_SITE[n]] for n in fk.TARGET_NAMES])
        dp, dR = errors(gpu[i], m)
        assert dp <= 5e-6 and dR <= 5e-6, (names[i], dp, dR)


def test_gpu_fk_fp32_matches_fp64(fk, batch):
    Q, _ = batch
    g64, _ = fk.target_poses_f64(Q)
    g32, _ = fk.target_poses_f32(Q.astype(np.float32))
    assert g32.dtype == np.float32
    dp, dR = errors(g32.astype(np.float64), g64)
    assert dp <= 2e-5 and dR <= 2e-5, (dp, dR)


def test_tool_table_equals_grid_fixed_frames(fk, batch):
    Q, _ = batch
    a, _ = fk.target_poses_f64(Q, source="metadata")
    b, _ = fk.target_poses_f64(Q, source="grid")
    dp, dR = errors(a, b)
    assert dp <= 1e-12 and dR <= 1e-12


def test_batch_order_and_launch_shape_invariance(fk, batch):
    Q, _ = batch
    full, _ = fk.target_poses_f64(Q)
    perm = np.random.default_rng(3).permutation(len(Q))
    shuffled, _ = fk.target_poses_f64(np.ascontiguousarray(Q[perm]))
    np.testing.assert_array_equal(shuffled, full[perm])
    for warps in (1, 3, 8):
        np.testing.assert_array_equal(fk.target_poses_f64(Q, warps_per_block=warps)[0], full)
    single, _ = fk.target_poses_f64(Q[7:8])
    np.testing.assert_array_equal(single[0], full[7])


@pytest.mark.parametrize("block,moved", [(range(12, 15), {0, 1}), (range(15, 22), {0}), (range(22, 29), {1}),
                                          (range(0, 6), {2}), (range(6, 12), {3})])
def test_branch_isolation_on_gpu(fk, described, block, moved):
    lo, hi = g1.limits(described).T
    rng = np.random.default_rng(11)
    q0 = rng.uniform(lo, hi, (16, 29))
    q1 = q0.copy(); q1[:, list(block)] = rng.uniform(lo[list(block)], hi[list(block)], (16, len(block)))
    a, _ = fk.target_poses_f64(q0)
    b, _ = fk.target_poses_f64(q1)
    changed = {k for k in range(4) if np.any(np.abs(a[:, k] - b[:, k]) > 1e-12)}
    assert changed == moved


def test_input_validation(fk):
    with pytest.raises(ValueError, match="shape"):
        fk.target_poses_f64(np.zeros((2, 28)))
    with pytest.raises(ValueError, match="source"):
        fk.target_poses_f64(np.zeros((1, 29)), source="other")


# --- prepared-artifact identity --------------------------------------------------------------------

def test_identical_inputs_reuse_the_artifact(tb, artifact, store):
    again = tb.build_tree(URDF, SPEC, name=NAME, cache_dir=store)
    assert not again.compiled and again.content_key == artifact.content_key
    loaded = tb.load_tree(NAME, urdf=URDF, targets_spec=SPEC, cache_dir=store)
    assert loaded.content_key == artifact.content_key and loaded.model_sha256 == artifact.model_sha256


def _spec_copy(tmp_path, mutate):
    spec = json.loads(SPEC.read_text())
    mutate(spec)
    path = tmp_path / "targets.json"
    path.write_text(json.dumps(spec))
    return path


def test_changed_targets_rebuild_and_reject_stale(tb, artifact, store, tmp_path):
    reordered = _spec_copy(tmp_path, lambda s: s["targets"].reverse())
    other = tb.build_tree(URDF, reordered, name="g1-tree-mutated-targets", cache_dir=store)
    assert other.content_key != artifact.content_key and other.model_sha256 != artifact.model_sha256
    module = tb.import_tree(other)
    assert list(module.TARGET_NAMES) == ["right_foot", "left_foot", "right_hand", "left_hand"]
    with pytest.raises(tb.BuildError, match="does not match"):
        tb.load_tree("g1-tree-mutated-targets", urdf=URDF, targets_spec=SPEC, cache_dir=store)
    with pytest.raises(tb.BuildError, match="does not match"):
        tb.load_tree(NAME, urdf=URDF, targets_spec=reordered, cache_dir=store)


def test_changed_urdf_tool_rebuilds_and_rejects_stale(tb, artifact, store, tmp_path, fk):
    tree = ET.parse(URDF)
    tree.getroot().find("joint[@name='left_foot_contact_joint']/origin").set("rpy", "0 0 0.5")
    urdf = tmp_path / "g1_29dof_contacts.urdf"
    tree.write(urdf)
    spec = _spec_copy(tmp_path, lambda s: s.pop("urdf_sha256"))
    rotated = tb.build_tree(urdf, spec, name="g1-tree-mutated-urdf", cache_dir=store)
    assert rotated.content_key != artifact.content_key and rotated.urdf_sha256 != artifact.urdf_sha256
    with pytest.raises(tb.BuildError, match="does not match"):
        tb.load_tree("g1-tree-mutated-urdf", urdf=URDF, targets_spec=SPEC, cache_dir=store)
    module = tb.import_tree(rotated)
    q = np.zeros((1, 29))
    a, b = fk.target_poses_f64(q)[0], module.target_poses_f64(q)[0]
    np.testing.assert_allclose(a[0, [0, 1, 3]], b[0, [0, 1, 3]], atol=0)        # other targets unchanged
    R = a[0, 2, :3, :3].T @ b[0, 2, :3, :3]                                       # left foot: +0.5 rad yaw kept
    np.testing.assert_allclose(R[:2, :2], [[np.cos(0.5), -np.sin(0.5)], [np.sin(0.5), np.cos(0.5)]], atol=1e-12)
    with pytest.raises(tb.BuildError, match="pinned"):                           # canonical spec refuses other URDFs
        tb.build_tree(urdf, SPEC, name="g1-tree-wrong-pin", cache_dir=store, plan_only=True)


def test_rebuilt_name_cannot_replace_an_imported_module(tb, store, tmp_path, fk):
    reordered = _spec_copy(tmp_path, lambda s: s["targets"].reverse())
    changed = tb.build_tree(URDF, reordered, name=NAME, cache_dir=store)      # same name, new identity
    try:
        with pytest.raises(tb.BuildError, match="changed after it was imported"):
            tb.import_tree(changed)
    finally:
        tb.build_tree(URDF, SPEC, name=NAME, cache_dir=store)                 # restore the canonical entry


def test_tree_and_panda_artifacts_are_not_confused(tb, artifact, store, fk):
    from hjcdik import _build
    with pytest.raises(_build.BuildError):
        _build.import_artifact(_build.load(NAME, cache_dir=store))           # Panda loader refuses a tree module
    record = _build.load(NAME, cache_dir=store)
    assert record.ee_frame == ""                                              # carries no single-target EE define
    with pytest.raises(tb.BuildError, match="no prepared tree build"):
        tb.load_tree("panda-panda_grasptarget_hand", cache_dir=store)


def test_packaged_panda_solver_unaffected_in_the_same_process(fk):
    import hjcdik
    assert hjcdik.build_info()["robot"] == "panda" and hjcdik.build_info()["num_joints"] == 7
    solver = hjcdik.Solver()
    targets = solver.sample_targets(4, seed=0)
    fk.target_poses_f32(np.zeros((8, 29), np.float32))
    out = solver.solve(targets, batch_size=256, num_solutions=1, collision=None)
    assert out.count.shape == (4,) and np.all(out.count >= 1)
    assert hjcdik.build_info()["grid_header_sha256"] == "47632efb92f1d5a9de49d02804606417e85d7f750a4094ca70fc9e47a671b031"
