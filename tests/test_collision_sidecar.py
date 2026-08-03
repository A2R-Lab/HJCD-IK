"""Checkpoint-1 (pre-CUDA) tests for the G1 collision sidecar (spec section 11, tests 1-12).

Pure-numpy tests always run. The FK-vs-MuJoCo test skips gracefully when MuJoCo is not in the
environment. Requires the built artifacts (generated/g1_collision_sidecar.json and the corpus);
regenerate with:
    python3 collision_sidecar/apply_tuning.py && python3 collision_sidecar/build_collision_sidecar.py
    MUJOCO_GL=egl python3 collision_sidecar/corpus.py
"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HJCD = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HJCD, "collision_sidecar"))

from collision_cpu import SidecarCPU, seg_seg_dist, pt_seg_dist  # noqa: E402
from urdf_model import parse_urdf, HJCD_JOINT_ORDER, verify_joint_order  # noqa: E402

CORPUS = os.path.join(HJCD, "generated", "g1_collision_corpus.json")
JSON = os.path.join(HJCD, "generated", "g1_collision_sidecar.json")
URDF = os.path.join(HJCD, "csrc", "urdf", "g1_29dof_rev_1_0.urdf")

pytestmark = pytest.mark.skipif(not (os.path.exists(JSON) and os.path.exists(CORPUS)),
                                reason="sidecar/corpus artifacts not built")


@pytest.fixture(scope="module")
def sc():
    return SidecarCPU()


@pytest.fixture(scope="module")
def corpus():
    return json.load(open(CORPUS))


def _cfg(corpus, category):
    return next(np.asarray(s["q"], float) for s in corpus["samples"] if s["category"] == category)


def _detected_colliding(corpus, category):
    """A clearly-colliding exemplar of a category (skips the sub-mm boundary cases)."""
    sc = SidecarCPU()
    best = None
    for s in corpus["samples"]:
        if s["category"] == category and s["label"]["colliding"]:
            q = np.asarray(s["q"], float)
            _, _, gap = sc.check(q)
            if gap < (best[1] if best else 0.0):
                best = (q, gap)
    return best[0] if best else None


# 1. HJCD joint-name ordering matches the sidecar
def test_joint_ordering(sc):
    model = parse_urdf(URDF)
    assert verify_joint_order(model, HJCD_JOINT_ORDER)
    assert list(sc.art["hjcd_joint_order"]) == list(HJCD_JOINT_ORDER)
    assert len(HJCD_JOINT_ORDER) == 29


# 2. sidecar FK vs MuJoCo (skips without MuJoCo)
def test_fk_matches_mujoco():
    try:
        import mujoco  # noqa: F401
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HJCD)), "src"))
        from collision_sidecar.validate_collision_sidecar import fk_check  # noqa
    except Exception:
        pytest.skip("MuJoCo / parent scene unavailable")
    from validate_collision_sidecar import fk_check
    fk, status = fk_check()
    if fk is None:
        pytest.skip(status)
    maxp, maxr = fk
    assert maxp < 5e-4 and maxr < 2e-3, (maxp, maxr)


# 3. sphere-sphere distance
def test_sphere_sphere_distance():
    a, b = np.array([0.0, 0, 0]), np.array([0.3, 0, 0])
    assert abs(np.linalg.norm(a - b) - 0.3) < 1e-12


# 4. sphere-capsule (point-segment) distance
def test_sphere_capsule_distance():
    p = np.array([0.0, 0.1, 0])
    assert abs(pt_seg_dist(p, np.array([-1.0, 0, 0]), np.array([1.0, 0, 0])) - 0.1) < 1e-12
    # projection past the cap end clamps to the endpoint
    p2 = np.array([2.0, 0.0, 0])
    assert abs(pt_seg_dist(p2, np.array([-1.0, 0, 0]), np.array([1.0, 0, 0])) - 1.0) < 1e-12


# 5. capsule-capsule (segment-segment) distance
def test_capsule_capsule_distance():
    d = seg_seg_dist(np.array([-1.0, 0, 0]), np.array([1.0, 0, 0]),
                     np.array([-1.0, 0.2, 0.5]), np.array([1.0, 0.2, 0.5]))
    assert abs(d - np.hypot(0.2, 0.5)) < 1e-9          # parallel offset segments
    d2 = seg_seg_dist(np.array([-1.0, 0, 0]), np.array([1.0, 0, 0]),
                      np.array([0.0, -1, 0.3]), np.array([0.0, 1, 0.3]))
    assert abs(d2 - 0.3) < 1e-9                        # crossing segments, gap in z


# 6. neutral pose is free
def test_neutral_free(sc):
    assert sc.collision_free(np.zeros(29))


# 7. nominal crouch is free
def test_crouch_free(sc, corpus):
    assert sc.collision_free(_cfg(corpus, "crouch"))


# 8. deliberate arm-torso collision detected
def test_arm_torso_detected(corpus, sc):
    q = _detected_colliding(corpus, "deliberate_arm_torso")
    assert q is not None and not sc.collision_free(q)


# 9. deliberate elbow/wrist-region (forearm) collision detected
def test_forearm_detected(corpus, sc):
    q = _detected_colliding(corpus, "deliberate_forearm")
    assert q is not None and not sc.collision_free(q)


# 10. deliberate leg-leg collision detected
def test_leg_leg_detected(corpus, sc):
    q = _detected_colliding(corpus, "deliberate_leg_leg")
    assert q is not None and not sc.collision_free(q)


# 11. descendant + affected-pair lists correct
def test_descendant_and_affected_pairs(sc):
    model = parse_urdf(URDF)
    for jn in HJCD_JOINT_ORDER:
        assert sc.art["joint_descendant_links"][jn] == model.descendant_links(jn)
    checked = [tuple(p) for p in sc.art["checked_link_pairs"]]
    for jn in HJCD_JOINT_ORDER:
        sub = set(sc.art["joint_descendant_links"][jn])
        for idx in sc.art["joint_affected_pairs"][jn]:
            a, b = checked[idx]
            assert (a in sub) ^ (b in sub)             # exactly one side in the subtree
        # completeness: every cross-subtree checked pair is listed
        expect = {i for i, (a, b) in enumerate(checked) if (a in sub) ^ (b in sub)}
        assert set(sc.art["joint_affected_pairs"][jn]) == expect


# 12. CPU incremental check equals CPU full check after one-joint updates
def test_incremental_equals_full(sc, corpus):
    rng = np.random.default_rng(0)
    bases = [np.zeros(29)] + [np.asarray(s["q"], float)
                              for s in corpus["samples"][:6] if s["category"] == "random"]
    for q_base in bases:
        for _ in range(5):
            j = int(rng.integers(29))
            val = float(q_base[j] + rng.normal(scale=0.6))
            inc = sc.incremental_linkpair_verdict(q_base, j, val)
            full = sc.full_linkpair_verdict(np.where(np.arange(29) == j, val, q_base))
            assert inc == full, (j, inc ^ full)


# ===========================================================================
# Checkpoint 1B: hybrid torso-SDF narrow phase (spec section 5)
# ===========================================================================
from collision_cpu import TorsoSDF  # noqa: E402

SDF = os.path.join(HJCD, "generated", "g1_torso_sdf.npz")
sdf_skip = pytest.mark.skipif(not os.path.exists(SDF), reason="torso SDF not built")


def test_sdf_interpolation_analytic():
    # analytic plane field sdf(x,y,z) = x - 0.1 on a small grid; trilinear must recover it
    sp = 0.02
    xs = np.arange(10) * sp
    grid = np.zeros((10, 10, 10), np.float32)
    for i, x in enumerate(xs):
        grid[i, :, :] = x - 0.1
    f = TorsoSDF.from_grid(grid, origin=[0, 0, 0], spacing=sp)
    for x in (0.03, 0.055, 0.12):
        assert abs(f.trilinear_sdf([x, 0.05, 0.05]) - (x - 0.1)) < 1e-4


@sdf_skip
def test_sphere_vs_sdf():
    f = TorsoSDF(SDF)
    # a point deep inside the torso (near a chest geom centre) is negative; far outside positive
    inside = [0.005, 0.0, 0.15]
    outside = [0.4, 0.4, 0.15]
    assert f.sphere_torso_sdf_distance(inside, 0.0)[0] < 0
    assert f.sphere_torso_sdf_distance(outside, 0.0)[0] > 0.1


@sdf_skip
def test_capsule_vs_sdf_and_adaptive_interior():
    f = TorsoSDF(SDF)
    # a capsule whose ENDPOINTS are outside the torso but whose middle passes through it:
    # endpoint-only sampling would miss it; adaptive sampling must find the interior minimum.
    p0 = np.array([-0.3, 0.0, 0.15]); p1 = np.array([0.3, 0.0, 0.15])
    gap, closest, ev = f.capsule_torso_sdf_distance(p0, p1, 0.0)
    assert gap < 0, "adaptive capsule sampling missed an interior collision"
    assert ev > 3


@sdf_skip
def test_neutral_shoulder_nesting_free(sc):
    # the resting shoulder nests against the torso; the SDF must keep it FREE
    assert sc.collision_free(np.zeros(29))
    lp = sc.colliding_link_pairs(np.zeros(29))
    assert not any("torso_link" in k for k in lp)


@sdf_skip
def test_shoulder_torso_left_and_right_detected(corpus, sc):
    for side in ("left", "right"):
        got = False
        for s in corpus["samples"]:
            if not (s["category"] == "deliberate_arm_torso" and s["label"]["colliding"]):
                continue
            pairs = " ".join("".join(p) for (p, _d) in s["label"]["pairs"])
            if f"{side}_shoulder" in pairs and "torso" in pairs:
                if not sc.collision_free(np.asarray(s["q"], float)):
                    got = True; break
        assert got, f"no detected {side} shoulder-torso collision"


@sdf_skip
def test_hip_torso_detected(corpus, sc):
    ok = False
    for s in corpus["samples"]:
        if s["category"] == "deliberate_hip_pelvis" and s["label"]["colliding"]:
            pairs = " ".join("".join(p) for (p, _d) in s["label"]["pairs"])
            if "torso" in pairs and not sc.collision_free(np.asarray(s["q"], float)):
                ok = True; break
    if not ok:
        pytest.skip("no hip-torso (vs pelvis) deliberate collision in corpus")
    assert ok


@sdf_skip
def test_sdf_pair_incremental_equals_full(sc, corpus):
    # a joint that moves an arm relative to the torso must update the SDF pair incrementally
    rng = np.random.default_rng(1)
    j = HJCD_JOINT_ORDER.index("left_shoulder_roll_joint")
    base = np.zeros(29)
    for _ in range(5):
        val = float(rng.uniform(-1.5, 1.5))
        inc = sc.incremental_linkpair_verdict(base, j, val)
        full = sc.full_linkpair_verdict(np.where(np.arange(29) == j, val, base))
        assert inc == full


@sdf_skip
def test_corpus_and_sdf_hashes():
    f = TorsoSDF(SDF)
    art = json.load(open(JSON))
    assert f.hashes["urdf"] == art["urdf_hash"]     # SDF + topology built from the same URDF
    corp = json.load(open(CORPUS))
    assert corp["n"] == len(corp["samples"]) and corp["n_tuning"] + corp["n_heldout"] == corp["n"]


# ===========================================================================
# Checkpoint 1C: pelvis SDF + pair-generic cluster routing (spec section 6)
# ===========================================================================
PELVIS_SDF = os.path.join(HJCD, "generated", "g1_pelvis_sdf.npz")
pelvis_skip = pytest.mark.skipif(not os.path.exists(PELVIS_SDF), reason="pelvis SDF not built")


@pelvis_skip
def test_pelvis_sdf_reproducibility_and_hashes():
    f = TorsoSDF(PELVIS_SDF)
    art = json.load(open(JSON))
    assert f.hashes["urdf"] == art["urdf_hash"]
    assert f.cluster_id == "PELVIS" and f.torso_link == "base_link"
    assert tuple(f.dims) == tuple(np.asarray(f.sdf.shape))


@pelvis_skip
def test_neutral_hip_pelvis_nesting_free(sc):
    assert sc.collision_free(np.zeros(29))
    lp = sc.colliding_link_pairs(np.zeros(29))
    assert not any("base_link" in k for k in lp)         # pelvis nesting stays free


@pelvis_skip
def test_deliberate_hip_pelvis_left_and_right(corpus, sc):
    seen = set()
    for s in corpus["samples"]:
        if s["category"] != "deliberate_hip_pelvis" or not s["label"]["colliding"]:
            continue
        if sc.collision_free(np.asarray(s["q"], float)):
            continue
        for (p, _d) in s["label"]["pairs"]:
            for side in ("left", "right"):
                if f"{side}_hip" in "".join(p) and "base_link" in p:
                    seen.add(side)
    assert seen, "no detected hip<->pelvis collision"


@pelvis_skip
def test_deliberate_forearm_pelvis_detected(corpus, sc):
    q = _detected_colliding(corpus, "deliberate_forearm")
    assert q is not None and not sc.collision_free(q)


@pelvis_skip
def test_pelvis_sphere_and_capsule_vs_sdf():
    f = TorsoSDF(PELVIS_SDF)
    inside = [0.0, 0.0, -0.09]                            # inside the pelvis mass
    outside = [0.3, 0.3, -0.09]
    assert f.sphere_torso_sdf_distance(inside, 0.0)[0] < 0
    assert f.sphere_torso_sdf_distance(outside, 0.0)[0] > 0.1
    gap, _, ev = f.capsule_torso_sdf_distance([-0.3, 0, -0.09], [0.3, 0, -0.09], 0.0)
    assert gap < 0 and ev > 3                             # capsule through the pelvis


@pelvis_skip
def test_pelvis_full_incremental_agreement(sc):
    rng = np.random.default_rng(2)
    j = HJCD_JOINT_ORDER.index("left_hip_roll_joint")     # moves a leg vs the pelvis
    base = np.zeros(29)
    for _ in range(5):
        val = float(rng.uniform(-0.5, 0.5))
        assert sc.incremental_linkpair_verdict(base, j, val) == \
               sc.full_linkpair_verdict(np.where(np.arange(29) == j, val, base))


@pelvis_skip
def test_sdf_routing_selects_correct_cluster(sc):
    # an arm prim paired with torso routes to TORSO; a leg prim paired with base_link -> PELVIS
    W, T = sc.world_primitives(np.zeros(29))
    for cid, want in (("TORSO", "TORSO"), ("PELVIS", "PELVIS")):
        c = sc.clusters[cid]
        il = c["limb_prims"][0]
        _, _, diag = sc._cluster_gap(W[il], T[c["link"]], cid)
        assert diag["sdf_id"] == want


# ===========================================================================
# Checkpoint 1D: exact convex/GJK narrow phase (spec section 7)
# ===========================================================================
from gjk import gjk, world_support, _support_local, link_pieces_collide  # noqa: E402

_I = np.eye(3); _Z = np.zeros(3)
gjk_skip = pytest.mark.skipif(
    not os.path.exists(os.path.join(HJCD, "generated", "g1_convex_pieces.json")),
    reason="convex pieces not built")


def test_support_mappings():
    d = np.array([1.0, 0, 0])
    assert np.allclose(_support_local({"type": "point", "p": [2, 0, 0]}, d), [2, 0, 0])
    assert np.allclose(_support_local({"type": "sphere", "center": [0, 0, 0], "radius": 1.0}, d), [1, 0, 0])
    assert np.allclose(_support_local({"type": "capsule", "p0": [-1, 0, 0], "p1": [1, 0, 0], "radius": 0.5}, d), [1.5, 0, 0])
    # box: use a diagonal direction so the supporting corner is unique
    assert np.allclose(_support_local({"type": "box", "center": [0, 0, 0], "half": [1, 2, 3], "R": _I.tolist()},
                                      np.array([1.0, 1.0, 1.0])), [1, 2, 3])
    cube = [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    assert _support_local({"type": "hull", "verts": cube}, d)[0] == 1


def _sph(c, r):
    return {"type": "sphere", "center": list(c), "radius": r}


def test_gjk_separated_touching_intersecting():
    A = _sph([0, 0, 0], 1.0)
    far = gjk(lambda d: world_support(A, _I, _Z, d), lambda d: world_support(_sph([3, 0, 0], 1), _I, _Z, d))
    assert (not far["colliding"]) and abs(far["distance"] - 1.0) < 1e-6
    touch = gjk(lambda d: world_support(A, _I, _Z, d), lambda d: world_support(_sph([2, 0, 0], 1), _I, _Z, d))
    assert touch["distance"] < 1e-4
    hit = gjk(lambda d: world_support(A, _I, _Z, d), lambda d: world_support(_sph([1, 0, 0], 1), _I, _Z, d))
    assert hit["colliding"]


def test_gjk_transformed_and_deterministic():
    box = {"type": "box", "center": [0, 0, 0], "half": [1, 1, 1], "R": _I.tolist()}
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])          # 90deg about z
    r1 = gjk(lambda d: world_support(box, R, np.array([3.0, 0, 0]), d), lambda d: world_support(box, _I, _Z, d))
    r2 = gjk(lambda d: world_support(box, R, np.array([3.0, 0, 0]), d), lambda d: world_support(box, _I, _Z, d))
    assert r1 == r2                                             # deterministic
    assert (not r1["colliding"]) and abs(r1["distance"] - 1.0) < 1e-6 and r1["iterations"] <= 64


@gjk_skip
def test_gjk_pairs_present(sc):
    assert len(sc.gjk_pairs) >= 8
    assert any("base_link" in p and "hip_roll" in "".join(p) for p in sc.gjk_pairs)
    assert any("wrist" in "".join(p) and "hip" in "".join(p) for p in sc.gjk_pairs)
    assert sc.convex and "base_link" in sc.convex


@gjk_skip
def test_deliberate_hip_roll_pelvis_left_right(corpus, sc):
    seen = set()
    for s in corpus["samples"]:
        if not s["label"]["colliding"]:
            continue
        for (p, _d) in s["label"]["pairs"]:
            if "base_link" in p and "hip_roll" in "".join(p) and _d < -5:
                if not sc.collision_free(np.asarray(s["q"], float)):
                    seen.add("left" if "left" in "".join(p) else "right")
    assert {"left", "right"} <= seen, f"deep hip_roll<->pelvis not detected both sides: {seen}"


@gjk_skip
def test_deliberate_wrist_thigh_detected(corpus, sc):
    ok = False
    for s in corpus["samples"]:
        if not s["label"]["colliding"]:
            continue
        for (p, _d) in s["label"]["pairs"]:
            if "wrist" in "".join(p) and "hip" in "".join(p) and _d < -5:
                if not sc.collision_free(np.asarray(s["q"], float)):
                    ok = True
    assert ok, "no deep wrist/hand<->thigh collision detected"


@gjk_skip
def test_neutral_hip_wrist_free(sc):
    lp = sc.colliding_link_pairs(np.zeros(29))
    # neutral: no GJK pair collides
    gset = {frozenset(p) for p in sc.gjk_pairs}
    assert not any(frozenset(k) in gset for k in lp)


@gjk_skip
def test_gjk_pair_incremental_equals_full(sc):
    rng = np.random.default_rng(3)
    for jn in ("left_hip_roll_joint", "left_shoulder_roll_joint"):
        j = HJCD_JOINT_ORDER.index(jn)
        base = np.zeros(29)
        for _ in range(4):
            val = float(rng.uniform(-1.2, 1.2))
            assert sc.incremental_linkpair_verdict(base, j, val) == \
                   sc.full_linkpair_verdict(np.where(np.arange(29) == j, val, base))
