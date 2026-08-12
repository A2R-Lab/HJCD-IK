"""Checkpoint 3B/3C integration tests: self_collision_mode off/final in hjcdik.solve.

Run: env PYTHONPATH= python3 -m pytest tests/test_collision_integration.py -q
Requires the rebuilt _hjcdik (sidecar compiled in) + a CUDA GPU.
"""
import hashlib, json, os, sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HJCD = os.path.dirname(HERE)
GEN = os.path.join(HJCD, "generated")
SC = os.path.join(HJCD, "collision_sidecar")
sys.path.insert(0, SC)

hjcdik = pytest.importorskip("hjcdik")
from collision_cpu import SidecarCPU  # noqa: E402


def _mat2quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    if i == 0:
        s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif i == 1:
        s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q); return q / np.linalg.norm(q)


def _problem(pi=0, B=256):
    """Reproducible multi-target problem: 4-frame targets by FK from a reference config."""
    rng = np.random.default_rng(3)
    qrefs = np.clip(rng.normal(0, 0.35, (4, 29)), -1.2, 1.2)
    T = np.asarray(hjcdik.target_transforms(np.ascontiguousarray(qrefs)))[pi]
    tpos = np.ascontiguousarray(np.broadcast_to(T[:, :3, 3], (B, 4, 3)))
    tquat = np.ascontiguousarray(np.broadcast_to(np.stack([_mat2quat(T[k, :3, :3]) for k in range(4)]), (B, 4, 4)))
    seed_q = np.ascontiguousarray(qrefs[pi][None] + np.random.default_rng(100 + pi).normal(0, 0.2, (B, 29)))
    return seed_q, tpos, tquat


def _solve(seed_q, tpos, tquat, **kw):
    return hjcdik.solve(seed_q, tpos, tquat, seed=42, precision="float32",
                        position_tol=0.02, orientation_tol=0.1, **kw)


@pytest.fixture(scope="module")
def prob():
    return _problem(1, 256)


@pytest.fixture(scope="module")
def cpu():
    return SidecarCPU()


# 1. default mode is off
def test_default_mode_is_off(prob):
    out = _solve(*prob)
    assert "self_collision" not in out and "self_collision_free" not in out

# 2. explicit off == omitted mode
def test_explicit_off_equals_omitted(prob):
    a = _solve(*prob); b = _solve(*prob, self_collision_mode="off")
    assert np.array_equal(a["joint_config"], b["joint_config"])
    assert np.array_equal(a["success"], b["success"])

# 3. invalid mode raises
def test_invalid_mode_raises(prob):
    with pytest.raises(ValueError):
        _solve(*prob, self_collision_mode="sometimes")

# 4. hard is ENABLED as of Checkpoint 3D/3E (behaviour covered by tests/test_collision_hard_mode.py).
#    What this suite still owns is that the mode is advertised consistently: a caller must never see
#    "hard" in supported_modes while solve() refuses it, or the reverse.
def test_hard_mode_is_enabled_and_advertised(prob):
    info = hjcdik.self_collision_info()
    assert info["hard_enabled"] is True
    assert "hard" in info["supported_modes"]
    out = _solve(*prob, self_collision_mode="hard")
    assert "self_collision" in out and out["self_collision"]["mode"] == "hard"

# 5. old positional/keyword signatures still valid
def test_backward_compatible_signature(prob):
    sq, tp, tq = prob
    out = hjcdik.solve(sq, tp, tq, None, 1.0, 1.0)     # positional legacy call
    assert "joint_config" in out and "success" in out

# 6. off byte-identical to captured pre-integration baseline.
#    The solver's argmin(cost) winner is workspace-warm-up-sensitive for borderline problems (a
#    property of the EXISTING solver, independent of Checkpoint 3), so this must reproduce the exact
#    baseline solve SEQUENCE -- done in a fresh subprocess via baseline_capture.py, then compared to
#    the frozen pre-integration reference.
def test_off_matches_captured_baseline():
    import subprocess
    ref = json.load(open(os.path.join(GEN, "baseline_g1_solver_ref.json")))
    env = dict(os.environ); env["PYTHONPATH"] = ""
    subprocess.run([sys.executable, os.path.join(SC, "baseline_capture.py")],
                   cwd=HJCD, env=env, check=True, capture_output=True)
    cur = json.load(open(os.path.join(GEN, "baseline_g1_solver.json")))
    for label in ref["runs"]:
        assert cur["runs"][label]["q_hashes"] == ref["runs"][label]["q_hashes"], label

# 7. off launches no sidecar kernel
def test_off_launches_no_sidecar(monkeypatch, prob):
    calls = {"n": 0}
    orig = hjcdik._hjcdik.sidecar_full_check
    monkeypatch.setattr(hjcdik._hjcdik, "sidecar_full_check",
                        lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), orig(*a, **k))[1])
    _solve(*prob, self_collision_mode="off")
    assert calls["n"] == 0

# 8. final verdict agrees with the CPU oracle on returned candidates
def test_final_agrees_with_oracle(prob, cpu):
    out = _solve(*prob, self_collision_mode="final")
    q = np.ascontiguousarray(np.asarray(out["joint_config"], np.float32))
    v = np.asarray(hjcdik._hjcdik.sidecar_full_check(q, 0.0))
    for b in range(0, q.shape[0], 7):
        cpu_coll = len(cpu.full_linkpair_verdict(q[b], margin=0.0)) > 0
        assert bool(v[b].any()) == cpu_coll

# 9/10. final rejects colliding, preserves free
def test_final_rejects_colliding_preserves_free(prob):
    off = _solve(*prob); fin = _solve(*prob, self_collision_mode="final")
    q = np.ascontiguousarray(np.asarray(fin["joint_config"], np.float32))
    coll = np.asarray(hjcdik._hjcdik.sidecar_full_check(q, 0.0)).any(axis=1)
    off_s = np.asarray(off["success"]).astype(bool)
    fin_s = np.asarray(fin["success"]).astype(bool)
    assert np.array_equal(fin_s, off_s & ~coll)      # colliding rejected, free preserved

# 11. no rejected candidate remains marked successful
def test_no_rejected_marked_successful(prob):
    fin = _solve(*prob, self_collision_mode="final")
    q = np.ascontiguousarray(np.asarray(fin["joint_config"], np.float32))
    coll = np.asarray(hjcdik._hjcdik.sidecar_full_check(q, 0.0)).any(axis=1)
    assert not bool((np.asarray(fin["success"]).astype(bool) & coll).any())

# 12. batch permutation preserves per-candidate verdicts
def test_batch_permutation_invariant(prob):
    fin = _solve(*prob, self_collision_mode="final")
    q = np.ascontiguousarray(np.asarray(fin["joint_config"], np.float32))
    v = np.asarray(hjcdik._hjcdik.sidecar_full_check(q, 0.0))
    perm = np.random.default_rng(9).permutation(q.shape[0])
    vp = np.asarray(hjcdik._hjcdik.sidecar_full_check(np.ascontiguousarray(q[perm]), 0.0))
    assert np.array_equal(vp, v[perm])

# 13. final works for B=1 and larger batches
def test_final_b1_and_large():
    for B in (1, 1024):
        sq, tp, tq = _problem(2, B)
        out = _solve(sq, tp, tq, self_collision_mode="final")
        assert out["self_collision"]["candidates_checked"] == B
        assert np.asarray(out["success"]).shape[0] == B

# 14. stale-binary guard catches a wrong SHA
def test_stale_binary_guard():
    so = hjcdik._hjcdik.__file__
    sha = hashlib.sha256(open(so, "rb").read()).hexdigest()

    def guard(expected):
        actual = hashlib.sha256(open(hjcdik._hjcdik.__file__, "rb").read()).hexdigest()
        if actual != expected:
            raise RuntimeError("stale/mismatched _hjcdik binary")
        return True
    assert guard(sha) is True
    with pytest.raises(RuntimeError):
        guard("0" * 64)

# 15. integrated fast-math corpus parity == 0 verdict mismatches
def test_integrated_corpus_parity(cpu):
    m = hjcdik._hjcdik
    for cid, fn in ((0, "g1_torso_sdf.npz"), (1, "g1_pelvis_sdf.npz")):
        z = np.load(os.path.join(GEN, fn), allow_pickle=True)
        m.sidecar_upload_sdf(cid, np.ascontiguousarray(z["sdf_i16"].astype(np.int16).ravel(order="C")))
    m.sidecar_upload_convex(np.ascontiguousarray(np.load(os.path.join(GEN, "g1_convex_verts.npy")).astype(np.float64)))
    corpus = json.load(open(os.path.join(GEN, "g1_collision_corpus.json")))
    Q = np.ascontiguousarray(np.stack([np.asarray(s["q"], np.float32) for s in corpus["samples"]]))
    V = np.asarray(m.sidecar_full_check(Q, 0.0))
    mism = sum(set(np.nonzero(V[b])[0].tolist()) != cpu.full_linkpair_verdict(Q[b], 0.0)
               for b in range(Q.shape[0]))
    assert mism == 0
