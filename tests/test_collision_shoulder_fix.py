"""Checkpoint 3C.1 tests: shoulder_yaw<->torso exact-GJK fix + fused final path.

Run: env PYTHONPATH= python3 -m pytest tests/test_collision_shoulder_fix.py -q
"""
import json, os, sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HJCD = os.path.dirname(HERE)
GEN = os.path.join(HJCD, "generated")
SC = os.path.join(HJCD, "collision_sidecar")
sys.path.insert(0, SC)

hjcdik = pytest.importorskip("hjcdik")
from collision_cpu import SidecarCPU  # noqa: E402
from urdf_model import HJCD_JOINT_ORDER  # noqa: E402

JI = {n: i for i, n in enumerate(HJCD_JOINT_ORDER)}


@pytest.fixture(scope="module")
def cpu():
    return SidecarCPU()


@pytest.fixture(scope="module")
def hn():
    return json.load(open(os.path.join(GEN, "g1_hard_negatives.json")))["hard_negatives"]


@pytest.fixture(scope="module")
def gpu_ready():
    hjcdik._ensure_self_collision_sidecar()
    return hjcdik._hjcdik


def _verdict(m, Q):
    return np.asarray(m.sidecar_full_check(np.ascontiguousarray(Q.astype(np.float32)), 0.0))


# 1/2. left/right shoulder_yaw<->torso known misses are detected (GPU)
#      hard-negative mujoco_pairs entry = [[linkA, linkB], depth_mm]
def test_left_shoulder_miss_detected(hn, gpu_ready):
    q = [np.asarray(h["q"]) for h in hn if "left_shoulder_yaw_link" in h["mujoco_pairs"][0][0]]
    V = _verdict(gpu_ready, np.stack(q))
    assert V.any(axis=1).all()

def test_right_shoulder_miss_detected(hn, gpu_ready):
    q = [np.asarray(h["q"]) for h in hn if "right_shoulder_yaw_link" in h["mujoco_pairs"][0][0]]
    V = _verdict(gpu_ready, np.stack(q))
    assert V.any(axis=1).all()

# 3. neutral shoulder nesting remains free
def test_neutral_shoulder_free(gpu_ready):
    assert not _verdict(gpu_ready, np.zeros((1, 29)))[0].any()

# 4. torso is a MULTI-piece convex link routed to GJK (3 pieces)
def test_torso_multipiece_gjk():
    art = json.load(open(os.path.join(GEN, "g1_collision_sidecar.json")))
    gjk = {frozenset(p) for p in art["gjk_pairs"]}
    for s in ("left", "right"):
        assert frozenset((f"{s}_shoulder_yaw_link", "torso_link")) in gjk
    cj = json.load(open(os.path.join(GEN, "g1_convex_pieces.json")))
    assert len([p for p in cj["links"]["torso_link"]["pieces"] if p["type"] == "hull"]) == 3

# 5. torso remains a cluster SDF for OTHER limbs (not globally replaced)
def test_torso_still_cluster():
    art = json.load(open(os.path.join(GEN, "g1_collision_sidecar.json")))
    assert "TORSO" in art["clusters"] and len(art["clusters"]["TORSO"]["limb_prims"]) > 0

# 6. all 11 hard negatives detected (CPU and GPU agree)
def test_all_hard_negatives_detected(hn, cpu, gpu_ready):
    Q = np.stack([np.asarray(h["q"]) for h in hn])
    V = _verdict(gpu_ready, Q)
    assert V.any(axis=1).all()                                    # GPU
    assert all(not cpu.collision_free(np.asarray(h["q"]), 0.0) for h in hn)   # CPU

# 7. held-out deep shoulder_yaw<->torso collisions are all detected (0 shoulder FN)
def test_heldout_shoulder_no_fn():
    R = json.load(open(os.path.join(GEN, "g1_heldout_corpus.json")))["rows"]
    sh_fn = [r for r in R if r["mujoco_colliding"] and not r["native_colliding"]
             and any("shoulder_yaw" in p for p in r["mujoco_pairs"]) and r["mujoco_min_depth_mm"] < -5]
    assert len(sh_fn) == 0

# 8. GPU full == incremental for the new GJK pairs (affected-pair CSR includes them).
#    _hjcdik binds only the final-mode full-check; incremental is exercised via the standalone module.
def test_gpu_full_equals_incremental_shoulder():
    from sidecar import Sidecar
    sc = Sidecar()
    rng = np.random.default_rng(5)
    Q = np.ascontiguousarray((rng.normal(0, 0.4, (64, 29))).astype(np.float32))
    base = sc.full_check(Q, 0.0)
    j = JI["left_shoulder_yaw_joint"]
    nv = (Q[:, j] + 0.3).astype(np.float32)
    Vi = sc.incr_check(Q, base, np.full(64, j, np.int32), nv, 0.0)
    Qn = Q.copy(); Qn[:, j] = nv
    Vf = sc.full_check(np.ascontiguousarray(Qn), 0.0)
    assert np.array_equal(Vi, Vf)

# 9. existing corpus verdicts do not regress (176/176, FN 0)
def test_corpus_no_regression(cpu, gpu_ready):
    corpus = json.load(open(os.path.join(GEN, "g1_collision_corpus.json")))
    Q = np.stack([np.asarray(s["q"], np.float32) for s in corpus["samples"]])
    V = _verdict(gpu_ready, Q)
    label = np.array([s["label"]["colliding"] for s in corpus["samples"]])
    assert int((label & ~V.any(axis=1)).sum()) == 0                      # FN 0
    assert int((label & V.any(axis=1)).sum()) == 176                     # recall

# 10. existing pelvis-SDF + wrist-thigh GJK pairs verdict-match the CPU oracle (unchanged)
def test_existing_pairs_verdict_parity(cpu, gpu_ready):
    corpus = json.load(open(os.path.join(GEN, "g1_collision_corpus.json")))
    Q = np.stack([np.asarray(s["q"], np.float32) for s in corpus["samples"]])[:60]
    V = _verdict(gpu_ready, Q)
    for b in range(0, 60, 3):
        assert set(np.nonzero(V[b])[0].tolist()) == cpu.full_linkpair_verdict(Q[b], 0.0)

# 11. fused final path performs no extra allocation after warm-up
def test_fused_no_extra_alloc(gpu_ready):
    m = gpu_ready
    Q = np.zeros((512, 29), np.float32)
    m.sidecar_full_check(Q, 0.0)                 # warm to >=512
    n0 = m.sidecar_ws_nalloc()
    for _ in range(5):
        m.sidecar_full_check(Q, 0.0)
    assert m.sidecar_ws_nalloc() == n0           # no growth for same-or-smaller batches

# 12. final mode returns no native collision; info advertises the fix
#     hard_enabled flipped to True in Checkpoint 3D/3E; the 3C.1 shoulder/GJK facts are unchanged.
def test_final_no_native_collision_and_info():
    info = hjcdik.self_collision_info()
    assert info["shoulder_torso_gjk"] is True and info["fused_final_path"] is True
    assert info["hard_enabled"] is True and info["n_gjk_pairs"] == 53

# 13. hard is available (Checkpoint 3D/3E) and still validates its arguments up front
def test_hard_available_and_validates():
    assert "hard" in hjcdik.self_collision_info()["supported_modes"]
    with pytest.raises(ValueError):        # collision_top_k is checked before any compute
        hjcdik.solve(np.zeros((4, 29), np.float32), np.zeros((4, 4, 3)), np.zeros((4, 4, 4)),
                     self_collision_mode="hard", collision_top_k=0)
