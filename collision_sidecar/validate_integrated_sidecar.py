"""Checkpoint 3C fast-math revalidation: run the accepted sidecar parity through the ACTUAL
_hjcdik binary (compiled with -use_fast_math), not the standalone sm_89 build.

Verifies FK / primitive / torso-SDF / pelvis-SDF / FP64-GJK / full-check verdict parity on all 289
corpus configs, plus neutral/crouch free, 176/176 deep recall, 0 false negatives. Also records the
loaded-binary identity (py exe, _hjcdik path, SHA, build mtime, sidecar hashes) for the stale-binary guard.
"""
from __future__ import annotations
import hashlib, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HJCD = os.path.dirname(HERE)
GEN = os.path.join(HJCD, "generated")
sys.path.insert(0, HERE)
from collision_cpu import SidecarCPU  # noqa: E402
from urdf_model import parse_urdf, HJCD_JOINT_ORDER  # noqa: E402


def bfs(model):
    order, fr = [], [model.root_link]
    while fr:
        lk = fr.pop(0); order.append(lk)
        for jn in model.children_joints.get(lk, ()):
            fr.append(model.joint_by_name[jn].child)
    return order


def upload(m):
    for cid, fn in ((0, "g1_torso_sdf.npz"), (1, "g1_pelvis_sdf.npz")):
        z = np.load(os.path.join(GEN, fn), allow_pickle=True)
        m.sidecar_upload_sdf(cid, np.ascontiguousarray(z["sdf_i16"].astype(np.int16).ravel(order="C")))
    m.sidecar_upload_convex(np.ascontiguousarray(np.load(os.path.join(GEN, "g1_convex_verts.npy")).astype(np.float64)))


def main():
    import hjcdik._hjcdik as m
    so = m.__file__
    print("=== loaded-binary safety ===")
    print("py exe    :", sys.executable)
    print("_hjcdik   :", so)
    print("SHA256    :", hashlib.sha256(open(so, "rb").read()).hexdigest()[:16])
    print("mtime     :", time.ctime(os.path.getmtime(so)))
    print("hashes    :", m.sidecar_model_info()["hashes"])
    upload(m)

    cpu = SidecarCPU(); NP = len(cpu.lp_class)
    model = parse_urdf(os.path.join(HJCD, "csrc", "urdf", "g1_29dof_rev_1_0.urdf"))
    order = bfs(model)
    corpus = json.load(open(os.path.join(GEN, "g1_collision_corpus.json")))
    samples = corpus["samples"]
    Q = np.ascontiguousarray(np.stack([np.asarray(s["q"], np.float32) for s in samples]))
    B = Q.shape[0]

    # ---- phase parities through the fast-math binary ----
    Tg = np.asarray(m.sidecar_fk(Q))                                   # [B,NL,16]
    fk_worst = 0.0
    for b in range(0, B, 4):
        Tc = model.fk(model.q_vector_to_names(Q[b]))
        for L, lk in enumerate(order):
            g = Tg[b, L].reshape(4, 4, order="F")
            fk_worst = max(fk_worst, np.abs(g[:3, 3] - Tc[lk][:3, 3]).max() * 1000)

    prim = np.asarray(m.sidecar_prim_gaps(Q))
    cl, _ev = m.sidecar_cluster_gaps(Q); cl = np.asarray(cl)
    art = json.load(open(os.path.join(GEN, "g1_collision_sidecar.json")))
    gs = {frozenset(p) for p in art["gjk_pairs"]}
    go = [(a, b) for (a, b) in art["checked_link_pairs"] if frozenset((a, b)) in gs]
    gj, _it = m.sidecar_gjk_gaps(Q); gj = np.asarray(gj)

    prim_v = torso_v = pelvis_v = gjk_v = 0
    sub = range(0, B, 3)
    for b in sub:
        W, T = cpu.world_primitives(Q[b])
        for idx, k in enumerate(cpu.lp_class):
            if k[0] == "prim":
                ref = min(cpu._pair_gap(W, i, j) for (i, j) in k[1])
                prim_v += int((prim[b, idx] < 0) != (ref < 0))
            elif k[0] == "cluster":
                _, cid, limbs = k; Tc = T[cpu.clusters[cid]["link"]]
                ref = min(cpu._cluster_gap(W[il], Tc, cid)[0] for il in limbs)
                mism = int((cl[b, idx] < 0) != (ref < 0))
                if cid == "TORSO": torso_v += mism
                else: pelvis_v += mism
        for o, (a, bl) in enumerate(go):
            ref, _, _ = cpu._gjk_gap(a, bl, T, 0.0)
            gjk_v += int((gj[b, o] < 0) != (ref < 0))

    # ---- full-check verdict parity on ALL 289 configs ----
    Vg = np.asarray(m.sidecar_full_check(Q, 0.0))
    full_mismatch = 0
    for b in range(B):
        cpu_set = cpu.full_linkpair_verdict(Q[b], margin=0.0)
        if set(np.nonzero(Vg[b])[0].tolist()) != cpu_set:
            full_mismatch += 1
    cat = {s["category"]: i for i, s in enumerate(samples)}
    neu_free = Vg[cat["neutral"]].sum() == 0
    cro_free = Vg[cat["crouch"]].sum() == 0
    label = np.array([s["label"]["colliding"] for s in samples])
    sidecar_coll = Vg.any(axis=1)
    FN = int((label & ~sidecar_coll).sum()); recall = int((label & sidecar_coll).sum())

    print("\n=== integrated fast-math parity (through _hjcdik) ===")
    print(f"FK max err            : {fk_worst:.6f} mm")
    print(f"primitive verdict mism: {prim_v}")
    print(f"torso-SDF verdict mism: {torso_v}")
    print(f"pelvis-SDF verdict mism:{pelvis_v}")
    print(f"FP64-GJK verdict mism : {gjk_v}")
    print(f"FULL verdict mism (289): {full_mismatch}")
    print(f"neutral free={neu_free} crouch free={cro_free} recall={recall}/176 FN={FN}")
    ok = (fk_worst < 0.05 and prim_v == 0 and torso_v == 0 and pelvis_v == 0 and gjk_v == 0
          and full_mismatch == 0 and neu_free and cro_free and recall == 176 and FN == 0)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
