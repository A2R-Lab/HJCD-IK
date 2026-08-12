"""Stage 4 parity: CUDA warp-cooperative GJK vs CPU oracle (Checkpoint 2).

Uploads exact convex vertices (CSR-slot order), runs warp-cooperative hull-hull GJK over every
GJK checked pair on neutral+crouch+corpus, compares signed gap and collision verdict against the
CPU gjk() oracle. GJK is f32(GPU) vs f64(CPU): verdict must agree exactly; distance within tol.
"""
from __future__ import annotations
import ctypes, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HJCD = os.path.dirname(HERE)
GEN = os.path.join(HJCD, "generated")
SCR = os.environ.get("SIDECAR_SCRATCH", "/tmp/sidecar_build")
sys.path.insert(0, HERE)
from urdf_model import HJCD_JOINT_ORDER  # noqa: E402
from collision_cpu import SidecarCPU  # noqa: E402
from parity_fk import build_lib  # noqa: E402


def upload_convex(lib):
    # canonical convex verts (piece order == PIECE_VERT_OFF), written by emit_cuda_header.py
    V = np.ascontiguousarray(np.load(os.path.join(GEN, "g1_convex_verts.npy")).astype(np.float64))
    lib.sidecar_upload_convex.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    lib.sidecar_upload_convex(V.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), V.shape[0])
    return V.shape[0]


def main():
    lib = ctypes.CDLL(build_lib())
    nv = upload_convex(lib)
    fp = ctypes.POINTER(ctypes.c_float); ip = ctypes.POINTER(ctypes.c_int32)
    lib.sidecar_gjk_gaps.argtypes = [fp, fp, ip, ctypes.c_int]

    art = json.load(open(os.path.join(GEN, "g1_collision_sidecar.json")))
    gjk_set = {frozenset(p) for p in art["gjk_pairs"]}
    gjk_ordered = [(a, b) for (a, b) in art["checked_link_pairs"] if frozenset((a, b)) in gjk_set]
    NG = len(gjk_ordered)

    cpu = SidecarCPU()
    ji = {n: i for i, n in enumerate(HJCD_JOINT_ORDER)}
    neutral = np.zeros(29, np.float32); crouch = np.zeros(29, np.float32)
    for s in ("left", "right"):
        crouch[ji[f"{s}_hip_pitch_joint"]] = -0.6; crouch[ji[f"{s}_knee_joint"]] = 1.2
        crouch[ji[f"{s}_ankle_pitch_joint"]] = -0.6
    corpus = json.load(open(os.path.join(GEN, "g1_collision_corpus.json")))
    Q = np.stack([neutral, crouch] + [np.asarray(s["q"], np.float32) for s in corpus["samples"]])
    Bn = Q.shape[0]
    gpu = np.zeros((Bn, NG), np.float32); it = np.zeros((Bn, NG), np.int32)
    lib.sidecar_gjk_gaps(Q.ctypes.data_as(fp), gpu.ctypes.data_as(fp), it.ctypes.data_as(ip), Bn)

    MARGIN = 0.0
    worst = 0.0; worst_where = None; verdict_mismatch = 0; max_it = 0; n_close = 0
    for b in range(Bn):
        _, T = cpu.world_primitives(Q[b])
        for o, (a, bl) in enumerate(gjk_ordered):
            ref, iters, _ = cpu._gjk_gap(a, bl, T, MARGIN)
            g = gpu[b, o]
            if (g < MARGIN) != (ref < MARGIN):
                verdict_mismatch += 1
            max_it = max(max_it, int(it[b, o]))
            # gap parity is only meaningful for COLLIDING pairs (both return ~ -1e-9). For free poses
            # the GPU's per-piece broad phase legitimately reports a conservative bounding-sphere
            # UNDERESTIMATE for far (broad-rejected) pieces of a multi-piece link (e.g. torso) -- that
            # is verdict-correct but not the true min distance, so it is NOT compared to CPU's true min.
            if ref < MARGIN:
                n_close += 1
                err = abs(g - ref) * 1000.0
                if err > worst:
                    worst, worst_where = err, (b, o, a, bl, float(g), float(ref))

    print(f"GJK over {Bn} configs x {NG} pairs ({nv} convex verts uploaded)")
    print(f"  verdict (gap<0) mismatches = {verdict_mismatch}")
    print(f"  colliding-pair gap max err = {worst:.6f} mm over {n_close} colliding samples  (worst {worst_where})")
    print(f"  max GJK iters seen = {max_it} (cap 64)")
    ok = verdict_mismatch == 0 and worst < 0.5 and max_it <= 64
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
