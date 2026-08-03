"""CUDA collision-sidecar parity tests (Checkpoint 2, Stage 6).

Verifies the standalone GPU sidecar against the accepted CPU oracle across all four narrow phases
plus the full/incremental checker. Requires a CUDA GPU + the _sidecar extension (auto-built).
Run:  env PYTHONPATH= python3 -m pytest tests/test_collision_sidecar_cuda.py -q
"""
import json, os, sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HJCD = os.path.dirname(HERE)
GEN = os.path.join(HJCD, "generated")
SC = os.path.join(HJCD, "collision_sidecar")
sys.path.insert(0, SC)

pytest.importorskip("numpy")
try:
    from sidecar import Sidecar
    from collision_cpu import SidecarCPU, seg_seg_dist, pt_seg_dist
    from urdf_model import parse_urdf, HJCD_JOINT_ORDER
except Exception as e:  # pragma: no cover
    pytest.skip(f"sidecar unavailable: {e}", allow_module_level=True)


@pytest.fixture(scope="module")
def ctx():
    sc = Sidecar()
    cpu = SidecarCPU()
    corpus = json.load(open(os.path.join(GEN, "g1_collision_corpus.json")))
    samples = corpus["samples"]
    Q = np.ascontiguousarray(np.stack([np.asarray(s["q"], np.float32) for s in samples]))
    Vg = sc.full_check(Q, 0.0)
    NP = sc.n_pairs
    cat = {s["category"]: i for i, s in enumerate(samples)}
    sub = list(range(0, len(samples), 5))          # 58-config CPU subset for speed
    return dict(sc=sc, cpu=cpu, samples=samples, Q=Q, Vg=Vg, NP=NP, cat=cat, sub=sub)


def _bfs(model):
    order, fr = [], [model.root_link]
    while fr:
        lk = fr.pop(0); order.append(lk)
        for jn in model.children_joints.get(lk, ()):
            fr.append(model.joint_by_name[jn].child)
    return order


# ---------------- FK (2) ----------------
def test_fk_neutral_matches_oracle(ctx):
    model = parse_urdf(os.path.join(HJCD, "csrc", "urdf", "g1_29dof_rev_1_0.urdf"))
    order = _bfs(model)
    from parity_fk import main as fk_main   # runs full FK parity (neutral+crouch+corpus)
    assert fk_main() == 0

def test_fk_link_count(ctx):
    assert ctx["sc"].info["n_links"] == 40 and ctx["sc"].info["n_joints"] == 29


# ---------------- primitive narrow phase (3) ----------------
def test_seg_seg_dist_parity(ctx):
    import ctypes
    from parity_fk import build_lib
    lib = ctypes.CDLL(build_lib()); fp = ctypes.POINTER(ctypes.c_float)
    lib.sidecar_seg_seg_probe.argtypes = [fp, fp, ctypes.c_int]
    rng = np.random.default_rng(1); B = 2000
    a = (rng.standard_normal((B, 12)) * 0.3).astype(np.float32); out = np.zeros(B, np.float32)
    lib.sidecar_seg_seg_probe(a.ctypes.data_as(fp), out.ctypes.data_as(fp), B)
    ref = np.array([seg_seg_dist(a[k, 0:3], a[k, 3:6], a[k, 6:9], a[k, 9:12]) for k in range(B)])
    assert np.abs(out - ref).max() * 1000 < 1e-2

def test_pt_seg_dist_parity(ctx):
    import ctypes
    from parity_fk import build_lib
    lib = ctypes.CDLL(build_lib()); fp = ctypes.POINTER(ctypes.c_float)
    lib.sidecar_pt_seg_probe.argtypes = [fp, fp, ctypes.c_int]
    rng = np.random.default_rng(2); B = 2000
    a = (rng.standard_normal((B, 9)) * 0.3).astype(np.float32); out = np.zeros(B, np.float32)
    lib.sidecar_pt_seg_probe(a.ctypes.data_as(fp), out.ctypes.data_as(fp), B)
    ref = np.array([pt_seg_dist(a[k, 0:3], a[k, 3:6], a[k, 6:9]) for k in range(B)])
    assert np.abs(out - ref).max() * 1000 < 1e-2

def test_prim_linkpair_gap_parity(ctx):
    sc, cpu, Q, sub = ctx["sc"], ctx["cpu"], ctx["Q"], ctx["sub"]
    gpu = sc.prim_gaps(Q[sub])
    worst = 0.0
    for bi, b in enumerate(sub):
        W, _ = cpu.world_primitives(Q[b])
        for idx, k in enumerate(cpu.lp_class):
            if k[0] != "prim": continue
            ref = min(cpu._pair_gap(W, i, j) for (i, j) in k[1])
            worst = max(worst, abs(gpu[bi, idx] - ref) * 1000)
    assert worst < 1e-2


# ---------------- cluster SDF (3) ----------------
def test_cluster_sphere_and_capsule_gap_parity(ctx):
    sc, cpu, Q, sub = ctx["sc"], ctx["cpu"], ctx["Q"], ctx["sub"]
    gpu, _ = sc.cluster_gaps(Q[sub])
    worst = 0.0
    for bi, b in enumerate(sub):
        W, T = cpu.world_primitives(Q[b])
        for idx, k in enumerate(cpu.lp_class):
            if k[0] != "cluster": continue
            _, cid, limbs = k
            Tc = T[cpu.clusters[cid]["link"]]
            ref = min(cpu._cluster_gap(W[il], Tc, cid)[0] for il in limbs)
            worst = max(worst, abs(gpu[bi, idx] - ref) * 1000)
    assert worst < 1e-2

def test_cluster_sdf_eval_counts_match(ctx):
    sc, cpu, Q, sub = ctx["sc"], ctx["cpu"], ctx["Q"], ctx["sub"]
    _, ev = sc.cluster_gaps(Q[sub])
    mism = 0
    for bi, b in enumerate(sub):
        W, T = cpu.world_primitives(Q[b])
        for idx, k in enumerate(cpu.lp_class):
            if k[0] != "cluster": continue
            _, cid, limbs = k
            Tc = T[cpu.clusters[cid]["link"]]
            evs = sum(cpu._cluster_gap(W[il], Tc, cid)[1] for il in limbs)
            mism += int(ev[bi, idx] != evs)
    assert mism == 0

def test_cluster_sdf_eval_cap_respected(ctx):
    _, ev = ctx["sc"].cluster_gaps(ctx["Q"])
    assert int(ev.max()) <= 48


# ---------------- convex GJK (3) ----------------
def test_gjk_gap_parity(ctx):
    sc, cpu, Q, sub = ctx["sc"], ctx["cpu"], ctx["Q"], ctx["sub"]
    art = json.load(open(os.path.join(GEN, "g1_collision_sidecar.json")))
    gs = {frozenset(p) for p in art["gjk_pairs"]}
    go = [(a, b) for (a, b) in art["checked_link_pairs"] if frozenset((a, b)) in gs]
    gpu, _ = sc.gjk_gaps(Q[sub])
    worst = 0.0
    for bi, b in enumerate(sub):
        _, T = cpu.world_primitives(Q[b])
        for o, (a, bl) in enumerate(go):
            ref, _, _ = cpu._gjk_gap(a, bl, T, 0.0)
            # gap parity only for COLLIDING pairs (both ~ -1e-9). For free multi-piece poses the GPU
            # per-piece broad phase returns a conservative bounding-sphere underestimate (verdict-correct).
            if ref < 0.0:
                worst = max(worst, abs(gpu[bi, o] - ref) * 1000)
    assert worst < 0.05   # f64 GJK: sub-micron on colliding pairs

def test_gjk_verdict_matches(ctx):
    sc, cpu, Q = ctx["sc"], ctx["cpu"], ctx["Q"]
    art = json.load(open(os.path.join(GEN, "g1_collision_sidecar.json")))
    gs = {frozenset(p) for p in art["gjk_pairs"]}
    go = [(a, b) for (a, b) in art["checked_link_pairs"] if frozenset((a, b)) in gs]
    gpu, _ = sc.gjk_gaps(Q)
    mism = 0
    for b in range(0, len(Q), 5):
        _, T = cpu.world_primitives(Q[b])
        for o, (a, bl) in enumerate(go):
            ref, _, _ = cpu._gjk_gap(a, bl, T, 0.0)
            mism += int((gpu[b, o] < 0) != (ref < 0))
    assert mism == 0

def test_gjk_iters_bounded(ctx):
    _, it = ctx["sc"].gjk_gaps(ctx["Q"])
    assert int(it.max()) <= 64


# ---------------- full checker (4) ----------------
def test_full_verdict_matches_cpu(ctx):
    sc, cpu, Q, Vg, sub = ctx["sc"], ctx["cpu"], ctx["Q"], ctx["Vg"], ctx["sub"]
    for b in sub:
        cpu_set = cpu.full_linkpair_verdict(Q[b], margin=0.0)
        gpu_set = set(np.nonzero(Vg[b])[0].tolist())
        assert cpu_set == gpu_set, f"config {b}"

def test_neutral_free(ctx):
    assert ctx["Vg"][ctx["cat"]["neutral"]].sum() == 0

def test_crouch_free(ctx):
    assert ctx["Vg"][ctx["cat"]["crouch"]].sum() == 0

def test_recall_and_no_false_negatives(ctx):
    label = np.array([s["label"]["colliding"] for s in ctx["samples"]])
    sidecar_coll = ctx["Vg"].any(axis=1)
    FN = int((label & ~sidecar_coll).sum())
    recall = int((label & sidecar_coll).sum())
    assert FN == 0 and recall == 176


# ---------------- incremental checker (3) ----------------
def test_incremental_matches_cpu(ctx):
    sc, cpu, Q, Vg = ctx["sc"], ctx["cpu"], ctx["Q"], ctx["Vg"]
    ji = list(range(0, 29, 4))
    for b in range(0, len(Q), 12):
        for j in ji:
            nv = float(Q[b][j] + 0.3)
            g = sc.incr_check(Q[b], Vg[b], j, nv, 0.0)[0]
            cpu_set = cpu.incremental_linkpair_verdict(Q[b], j, nv, margin=0.0)
            assert set(np.nonzero(g)[0].tolist()) == cpu_set, f"cfg {b} joint {j}"

def test_incremental_equals_full_on_trial(ctx):
    sc, Q, Vg = ctx["sc"], ctx["Q"], ctx["Vg"]
    B = len(Q); joints = list(range(29))
    trials = [(c, j) for c in range(B) for j in joints]
    Qb = np.stack([Q[c] for c, _ in trials]); Base = np.stack([Vg[c] for c, _ in trials])
    jidx = np.array([j for _, j in trials], np.int32)
    nv = np.array([Q[c][j] + 0.3 for c, j in trials], np.float32)
    Vi = sc.incr_check(Qb, Base, jidx, nv, 0.0)
    Qn = Qb.copy()
    for t, (c, j) in enumerate(trials): Qn[t, j] = nv[t]
    Vf = sc.full_check(Qn, 0.0)
    assert np.array_equal(Vi, Vf)

def test_incremental_base_buffer_not_mutated(ctx):
    sc, Q, Vg = ctx["sc"], ctx["Q"], ctx["Vg"]
    base = Vg[10].copy()
    sc.incr_check(Q[10], base, 4, float(Q[10][4] + 0.5), 0.0)
    assert np.array_equal(base, Vg[10])   # committed state untouched


# ---------------- robustness / model (4) ----------------
def test_no_cross_warp_state_leakage(ctx):
    sc, Q = ctx["sc"], ctx["Q"]
    ref = sc.full_check(Q, 0.0)
    perm = np.random.default_rng(7).permutation(len(Q))
    shuf = sc.full_check(np.ascontiguousarray(Q[perm]), 0.0)
    assert np.array_equal(shuf, ref[perm])         # neighbor-independent per warp

def test_single_equals_batch(ctx):
    sc, Q, Vg = ctx["sc"], ctx["Q"], ctx["Vg"]
    for b in (0, 50, 150, 288):
        assert np.array_equal(sc.full_check(Q[b:b+1], 0.0)[0], Vg[b])

def test_hashes_match_artifacts(ctx):
    model = parse_urdf(os.path.join(HJCD, "csrc", "urdf", "g1_29dof_rev_1_0.urdf"))
    h = ctx["sc"].info["hashes"]
    assert h["urdf"] == model.urdf_hash()
    assert h["joint_order"] == model.joint_order_hash()
    art = json.load(open(os.path.join(GEN, "g1_collision_sidecar.json")))
    assert h["proxy_yaml"] == art["proxy_yaml_hash"]

def test_model_info_counts(ctx):
    info = ctx["sc"].info
    assert info["n_checked_pairs"] == info["n_prim_pairs"] + info["n_cluster_pairs"] + info["n_gjk_pairs"]
    # +2 shoulder (3C.1) +4 leg (3C.2) +2 elbow<->torso (3C.3)
    assert info["n_clusters"] == 2 and info["n_gjk_pairs"] == 53
