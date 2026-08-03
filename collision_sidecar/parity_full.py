"""Stage 5 parity: CUDA full + incremental checker vs CPU oracle (Checkpoint 2).

Gates (all at margin=0, the sidecar operating point):
  * GPU full verdict SET == CPU full_linkpair_verdict, every corpus config
  * GPU incremental == CPU incremental_linkpair_verdict, deterministic joint trials
  * GPU incremental == GPU full(q_new) on the resulting trial config (affected-pair completeness)
  * neutral & crouch are collision-free
  * recall vs MuJoCo labels: all 176 colliding caught (FN == 0)
"""
from __future__ import annotations
import ctypes, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HJCD = os.path.dirname(HERE)
GEN = os.path.join(HJCD, "generated")
SCR = os.environ.get("SIDECAR_SCRATCH", "/tmp/sidecar_build")
sys.path.insert(0, HERE)
from urdf_model import HJCD_JOINT_ORDER  # noqa: E402
from collision_cpu import SidecarCPU  # noqa: E402
from parity_fk import build_lib  # noqa: E402
from parity_sdf import upload_sdfs  # noqa: E402
from parity_gjk import upload_convex  # noqa: E402

U8 = ctypes.POINTER(ctypes.c_uint8); FP = ctypes.POINTER(ctypes.c_float); IP = ctypes.POINTER(ctypes.c_int32)


def main():
    lib = ctypes.CDLL(build_lib()); upload_sdfs(lib); upload_convex(lib)
    lib.sidecar_full_check.argtypes = [FP, U8, ctypes.c_int, ctypes.c_float]
    lib.sidecar_incr_check.argtypes = [FP, U8, IP, FP, U8, ctypes.c_int, ctypes.c_float]

    cpu = SidecarCPU(); NP = len(cpu.lp_class)
    corpus = json.load(open(os.path.join(GEN, "g1_collision_corpus.json")))
    samples = corpus["samples"]
    Q = np.ascontiguousarray(np.stack([np.asarray(s["q"], np.float32) for s in samples]))
    B = Q.shape[0]

    # ---- full check ----
    Vg = np.zeros((B, NP), np.uint8)
    lib.sidecar_full_check(Q.ctypes.data_as(FP), Vg.ctypes.data_as(U8), B, 0.0)
    full_mismatch = 0; first = None
    for b in range(B):
        cpu_set = cpu.full_linkpair_verdict(Q[b], margin=0.0)
        gpu_set = set(np.nonzero(Vg[b])[0].tolist())
        if cpu_set != gpu_set:
            full_mismatch += 1
            if first is None: first = (b, sorted(cpu_set ^ gpu_set))
    # neutral / crouch free
    cat = {s["category"]: i for i, s in enumerate(samples)}
    neu_free = Vg[cat["neutral"]].sum() == 0
    cro_free = Vg[cat["crouch"]].sum() == 0
    # recall / FN vs MuJoCo labels
    label_coll = np.array([s["label"]["colliding"] for s in samples])
    sidecar_coll = Vg.any(axis=1)
    FN = int((label_coll & ~sidecar_coll).sum())
    recall = int((label_coll & sidecar_coll).sum())

    # ---- incremental: ALL configs x ALL 29 joints, deterministic +0.3 rad ----
    joints = list(range(29))
    trials = [(c, j) for c in range(B) for j in joints]
    J = len(trials)
    Qb = np.ascontiguousarray(np.stack([Q[c] for c, _ in trials]))
    Base = np.ascontiguousarray(np.stack([Vg[c] for c, _ in trials]))
    jidx = np.ascontiguousarray(np.array([j for _, j in trials], np.int32))
    newval = np.ascontiguousarray(np.array([Q[c][j] + 0.3 for c, j in trials], np.float32))
    Vincr = np.zeros((J, NP), np.uint8)
    lib.sidecar_incr_check(Qb.ctypes.data_as(FP), Base.ctypes.data_as(U8), jidx.ctypes.data_as(IP),
                           newval.ctypes.data_as(FP), Vincr.ctypes.data_as(U8), J, 0.0)
    # GPU full on q_new (affected-pair completeness gate; GPU-only, all trials)
    Qnew = Qb.copy()
    for t, (c, j) in enumerate(trials): Qnew[t, j] = newval[t]
    Qnew = np.ascontiguousarray(Qnew)
    Vfull_new = np.zeros((J, NP), np.uint8)
    lib.sidecar_full_check(Qnew.ctypes.data_as(FP), Vfull_new.ctypes.data_as(U8), J, 0.0)
    incr_vs_full = int((Vincr != Vfull_new).any(axis=1).sum())

    # CPU incremental cross-check on a stratified subsample (Python is ~40 ms/trial)
    sub = list(range(0, J, max(1, J // 1200)))
    incr_vs_cpu = 0
    t0 = time.time()
    for t in sub:
        c, j = trials[t]
        cpu_set = cpu.incremental_linkpair_verdict(Q[c], j, float(newval[t]), margin=0.0)
        gpu_set = set(np.nonzero(Vincr[t])[0].tolist())
        if cpu_set != gpu_set: incr_vs_cpu += 1

    print(f"FULL: {B} configs, {NP} pairs each -> set mismatches = {full_mismatch}" + (f"  first {first}" if first else ""))
    print(f"  neutral free = {neu_free}, crouch free = {cro_free}")
    print(f"  MuJoCo recall = {recall}/176 colliding caught, FN = {FN}")
    print(f"INCREMENTAL: {J} joint trials ({len(joints)} joints x {B} configs)")
    print(f"  GPU incr == GPU full(q_new): mismatches = {incr_vs_full}  (all {J} trials)")
    print(f"  GPU incr == CPU incr : mismatches = {incr_vs_cpu}  (subsample {len(sub)}, {time.time()-t0:.1f}s)")
    ok = (full_mismatch == 0 and neu_free and cro_free and FN == 0 and recall == 176
          and incr_vs_cpu == 0 and incr_vs_full == 0)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
