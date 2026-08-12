"""Checkpoint 3C.2 tests: exact-GJK for the cross-body leg-leg pairs (knee/hip_yaw).

Run: env PYTHONPATH= python3 -m pytest tests/test_collision_leg_fix.py -q
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
LEG_PAIRS = [("left_knee_link", "right_knee_link"), ("left_hip_yaw_link", "right_hip_yaw_link"),
             ("left_knee_link", "right_hip_yaw_link"), ("left_hip_yaw_link", "right_knee_link")]


@pytest.fixture(scope="module")
def cpu():
    return SidecarCPU()


@pytest.fixture(scope="module")
def m():
    hjcdik._ensure_self_collision_sidecar()
    return hjcdik._hjcdik


def _V(m, Q):
    return np.asarray(m.sidecar_full_check(np.ascontiguousarray(Q.astype(np.float32)), 0.0))


@pytest.fixture(scope="module")
def leg_hn():
    return json.load(open(os.path.join(GEN, "g1_leg_hard_negatives.json")))["candidates"]


# 1. every known deep leg-leg hard negative is detected
def test_all_leg_hard_negatives_detected(leg_hn, m, cpu):
    Q = np.stack([np.asarray(c["q"]) for c in leg_hn])
    assert _V(m, Q).any(axis=1).all()
    assert all(not cpu.collision_free(np.asarray(c["q"]), 0.0) for c in leg_hn)

# 2/3. neutral & crouch remain free
def test_neutral_free(m):
    assert not _V(m, np.zeros((1, 29)))[0].any()

def test_crouch_free(m):
    cro = np.zeros(29)
    for s in ("left", "right"):
        cro[JI[f"{s}_hip_pitch_joint"]] = -0.6; cro[JI[f"{s}_knee_joint"]] = 1.2
        cro[JI[f"{s}_ankle_pitch_joint"]] = -0.6
    assert not _V(m, cro[None])[0].any()

# 4/5. left-over-right and right-over-left deep collisions are detected (from the new leg held-out)
def _heldout_rows(kind):
    R = json.load(open(os.path.join(GEN, "g1_heldout_leg_corpus.json")))["rows"]
    return [r for r in R if r["kind"] == kind and r["mujoco_colliding"] and r["mujoco_min_depth_mm"] < -5]

def test_left_over_right_detected(m):
    rows = _heldout_rows("left_over_right")[:64]
    Q = np.stack([np.asarray(r["q"]) for r in rows])
    assert _V(m, Q).any(axis=1).all()

def test_right_over_left_detected(m):
    rows = _heldout_rows("right_over_left")[:64]
    Q = np.stack([np.asarray(r["q"]) for r in rows])
    assert _V(m, Q).any(axis=1).all()

# 6/7. hip_yaw and knee cross-body pairs are routed to GJK; knees are convex links
def test_leg_pairs_routed_to_gjk():
    art = json.load(open(os.path.join(GEN, "g1_collision_sidecar.json")))
    gjk = {frozenset(p) for p in art["gjk_pairs"]}
    for a, b in LEG_PAIRS:
        assert frozenset((a, b)) in gjk, (a, b)

def test_knee_is_convex_link():
    cj = json.load(open(os.path.join(GEN, "g1_convex_pieces.json")))
    for lk in ("left_knee_link", "right_knee_link"):
        assert lk in cj["links"] and len([p for p in cj["links"][lk]["pieces"] if p["type"] == "hull"]) >= 1

# 8. new leg-focused held-out: zero deep FN (global)
def test_new_heldout_zero_deep_fn():
    R = json.load(open(os.path.join(GEN, "g1_heldout_leg_corpus.json")))["rows"]
    deep_fn = [r for r in R if r["mujoco_colliding"] and not r["native_colliding"] and r["mujoco_min_depth_mm"] < -5]
    assert len(deep_fn) == 0

# 9. GPU full == incremental for a leg (hip_yaw) joint move
def test_gpu_full_equals_incremental_leg():
    from sidecar import Sidecar
    sc = Sidecar()
    rng = np.random.default_rng(8)
    Q = np.ascontiguousarray((rng.normal(0, 0.5, (64, 29))).astype(np.float32))
    base = sc.full_check(Q, 0.0)
    j = JI["left_hip_yaw_joint"]
    nv = (Q[:, j] + 0.4).astype(np.float32)
    Vi = sc.incr_check(Q, base, np.full(64, j, np.int32), nv, 0.0)
    Qn = Q.copy(); Qn[:, j] = nv
    Vf = sc.full_check(np.ascontiguousarray(Qn), 0.0)
    assert np.array_equal(Vi, Vf)

# 10. corpus + shoulder fix do not regress
def test_corpus_and_shoulder_preserved(m, cpu):
    corpus = json.load(open(os.path.join(GEN, "g1_collision_corpus.json")))
    Q = np.stack([np.asarray(s["q"], np.float32) for s in corpus["samples"]])
    V = _V(m, Q)
    label = np.array([s["label"]["colliding"] for s in corpus["samples"]])
    assert int((label & ~V.any(axis=1)).sum()) == 0 and int((label & V.any(axis=1)).sum()) == 176
    shn = json.load(open(os.path.join(GEN, "g1_hard_negatives.json")))["hard_negatives"]
    Qs = np.stack([np.asarray(h["q"]) for h in shn])
    assert _V(m, Qs).any(axis=1).all()

# 11. info advertises 20 GJK pairs; hard is enabled as of Checkpoint 3D/3E
def test_info_and_hard():
    info = hjcdik.self_collision_info()
    assert info["n_gjk_pairs"] == 53
    assert info["hard_enabled"] is True and info["supported_modes"] == ["off", "final", "hard"]
