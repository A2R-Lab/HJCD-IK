"""Native-completion checkpoint, Stage A: the broad audit's deep self-collision false-negative pair
classes are now detected by exact GJK (spec §16).

Enumerated from data (collision_sidecar/mine_all_selfcoll_fn.py -> g1_selfcoll_fn_corpus.json), NOT
from names. All 25 pair classes are now routed to exact FP64 GJK -- including the ankle_roll FOOT,
whose 4-sphere geometry is handled by the TYPED-piece path (sphere + hull) added in Task A. Every
deep-FN example, foot included, is detected; CPU and GPU agree exactly.

Run: env PYTHONPATH= python3 -m pytest tests/test_selfcoll_native_completion.py -q
"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HJCD = os.path.dirname(HERE)
GEN = os.path.join(HJCD, "generated")
sys.path.insert(0, os.path.join(HJCD, "collision_sidecar"))

hjcdik = pytest.importorskip("hjcdik")

FOOT_RESIDUAL = {  # the 4 ankle_roll cross-body classes that still need sphere-aware GJK
    frozenset(("left_ankle_roll_link", "right_hip_yaw_link")),
    frozenset(("left_ankle_roll_link", "right_knee_link")),
    frozenset(("left_hip_yaw_link", "right_ankle_roll_link")),
    frozenset(("left_knee_link", "right_ankle_roll_link")),
}


@pytest.fixture(scope="module", autouse=True)
def uploaded():
    hjcdik._ensure_self_collision_sidecar()


def _nat(Q):
    q = np.ascontiguousarray(np.asarray(Q, np.float32))
    return np.asarray(hjcdik._hjcdik.sidecar_full_check(q, 0.0))


@pytest.fixture(scope="module")
def fn_corpus():
    path = os.path.join(GEN, "g1_selfcoll_fn_corpus.json")
    if not os.path.exists(path):
        pytest.skip("run collision_sidecar/mine_all_selfcoll_fn.py to build the FN corpus")
    return json.load(open(path))


# 1/5. Every deep-FN example -- foot included -- is now detected (Task A foot closure).
def test_all_deep_fn_examples_detected(fn_corpus):
    Q = np.stack([np.asarray(e["q"]) for e in fn_corpus["examples"]])
    det = _nat(Q).any(axis=1)
    assert det.all(), f"{int((~det).sum())} deep-FN examples still missed after typed-piece GJK"


# 11/12. The fixed pair classes are on the exact GJK narrow phase (non-conservative proxy replaced).
def test_fixed_pair_classes_are_routed_to_gjk():
    art = json.load(open(os.path.join(GEN, "g1_collision_sidecar.json")))
    gjk = {frozenset(p) for p in art["gjk_pairs"]}
    # a representative sample of the classes the audit flagged and Stage A fixed
    must_be_gjk = [
        ("left_hip_yaw_link", "torso_link"), ("right_hip_yaw_link", "torso_link"),
        ("left_hip_roll_link", "torso_link"), ("right_hip_roll_link", "torso_link"),
        ("left_hip_yaw_link", "left_shoulder_yaw_link"),
        ("right_elbow_link", "right_knee_link"),
        ("left_knee_link", "left_wrist_pitch_link"),
        ("base_link", "torso_link"),                        # un-disabled + routed
    ]
    for a, b in must_be_gjk:
        assert frozenset((a, b)) in gjk, f"{a}<->{b} is not on exact GJK"


def test_base_torso_is_checked_not_disabled():
    """base<->torso was auto-disabled as 'cluster<->cluster, never collides'; the audit refuted that
    (waist flexion) so it must now be a checked, GJK-routed pair."""
    art = json.load(open(os.path.join(GEN, "g1_collision_sidecar.json")))
    checked = {frozenset(p) for p in art["checked_link_pairs"]}
    assert frozenset(("base_link", "torso_link")) in checked


# 13. gjk pair count matches the regenerated artifact (hash/count guard).
def test_gjk_pair_count():
    assert hjcdik.self_collision_info()["n_gjk_pairs"] == 53


# The foot pairs are now on exact typed GJK, and the foot links carry SPHERE convex pieces.
def test_foot_pairs_routed_and_typed_sphere():
    art = json.load(open(os.path.join(GEN, "g1_collision_sidecar.json")))
    gjk = {frozenset(p) for p in art["gjk_pairs"]}
    for pair in FOOT_RESIDUAL:
        assert pair in gjk, f"foot pair {pair} not routed to GJK"
    cj = json.load(open(os.path.join(GEN, "g1_convex_pieces.json")))
    for foot in ("left_ankle_roll_link", "right_ankle_roll_link"):
        pieces = cj["links"][foot]["pieces"]
        assert pieces and all(p["type"] == "sphere" for p in pieces), \
            f"{foot} is not a union of sphere pieces"


def test_typed_piece_hash_exposed():
    assert hjcdik.self_collision_info()["hashes"].get("typed_piece")
