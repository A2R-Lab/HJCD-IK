"""Stage 2 parity: CUDA primitive narrow phases vs CPU oracle (Checkpoint 2).

(a) focused distance functions: pt_seg_dist, seg_seg_dist on random synthetic geometry.
(b) per-config min gap over every PRIMITIVE checked link-pair (full cross product, incl. 4-prim feet).
"""
from __future__ import annotations
import ctypes, json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HJCD = os.path.dirname(HERE)
GEN = os.path.join(HJCD, "generated")
SCR = os.environ.get("SIDECAR_SCRATCH", "/tmp/sidecar_build")
sys.path.insert(0, HERE)
from urdf_model import parse_urdf, HJCD_JOINT_ORDER  # noqa: E402
from collision_cpu import SidecarCPU, seg_seg_dist, pt_seg_dist  # noqa: E402
from parity_fk import build_lib  # noqa: E402

rng = np.random.default_rng(0)


def main():
    lib = ctypes.CDLL(build_lib())
    fp = ctypes.POINTER(ctypes.c_float)
    for nm in ("sidecar_seg_seg_probe", "sidecar_pt_seg_probe", "sidecar_prim_gaps"):
        getattr(lib, nm).argtypes = [fp, fp, ctypes.c_int]

    # (a) focused distance parity on random geometry (mix of near/overlapping/far)
    B = 5000
    ss = (rng.standard_normal((B, 12)) * 0.3).astype(np.float32)
    out = np.zeros(B, np.float32)
    lib.sidecar_seg_seg_probe(ss.ctypes.data_as(fp), out.ctypes.data_as(fp), B)
    ref = np.array([seg_seg_dist(ss[k, 0:3], ss[k, 3:6], ss[k, 6:9], ss[k, 9:12]) for k in range(B)])
    ss_err = np.abs(out - ref).max() * 1000.0

    ps = (rng.standard_normal((B, 9)) * 0.3).astype(np.float32)
    out2 = np.zeros(B, np.float32)
    lib.sidecar_pt_seg_probe(ps.ctypes.data_as(fp), out2.ctypes.data_as(fp), B)
    ref2 = np.array([pt_seg_dist(ps[k, 0:3], ps[k, 3:6], ps[k, 6:9]) for k in range(B)])
    ps_err = np.abs(out2 - ref2).max() * 1000.0

    # (b) primitive link-pair gaps over neutral + crouch + corpus
    cpu = SidecarCPU()
    NP = len(cpu.lp_class)  # == N_CHECKED_PAIRS
    ji = {n: i for i, n in enumerate(HJCD_JOINT_ORDER)}
    neutral = np.zeros(29, np.float32); crouch = np.zeros(29, np.float32)
    for s in ("left", "right"):
        crouch[ji[f"{s}_hip_pitch_joint"]] = -0.6; crouch[ji[f"{s}_knee_joint"]] = 1.2
        crouch[ji[f"{s}_ankle_pitch_joint"]] = -0.6
    corpus = json.load(open(os.path.join(GEN, "g1_collision_corpus.json")))
    Q = np.stack([neutral, crouch] + [np.asarray(s["q"], np.float32) for s in corpus["samples"]])
    Bn = Q.shape[0]
    gpu = np.zeros((Bn, NP), np.float32)
    lib.sidecar_prim_gaps(Q.ctypes.data_as(fp), gpu.ctypes.data_as(fp), Bn)

    prim_idx = [i for i, k in enumerate(cpu.lp_class) if k[0] == "prim"]
    worst = 0.0; worst_where = None
    for b in range(Bn):
        W, _ = cpu.world_primitives(Q[b])
        for idx in prim_idx:
            sub = cpu.lp_class[idx][1]
            ref_gap = min(cpu._pair_gap(W, i, j) for (i, j) in sub)
            e = abs(gpu[b, idx] - ref_gap) * 1000.0
            if e > worst:
                worst, worst_where = e, (b, idx)

    print(f"(a) seg_seg_dist max err = {ss_err:.6f} mm over {B} random pairs")
    print(f"(a) pt_seg_dist  max err = {ps_err:.6f} mm over {B} random pairs")
    print(f"(b) prim link-pair gap max err = {worst:.6f} mm over {Bn} configs x {len(prim_idx)} prim pairs")
    ok = ss_err < 1e-2 and ps_err < 1e-2 and worst < 1e-2
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
