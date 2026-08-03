"""Stage 3 parity: CUDA cluster-SDF narrow phase vs CPU oracle (Checkpoint 2).

Uploads torso(0)+pelvis(1) int16 SDF grids, then compares per CLUSTER checked link-pair:
  * signed gap (trilinear sphere-SDF + adaptive capsule-SDF, incl. broad-phase reject)
  * adaptive-SDF evaluation count (exact branch-and-bound order + eval cap)
over neutral + crouch + all 289 corpus configs.
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

CID = {"TORSO": 0, "PELVIS": 1}


def upload_sdfs(lib):
    lib.sidecar_upload_sdf.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int16), ctypes.c_int]
    for name, cid in CID.items():
        fn = "g1_torso_sdf.npz" if cid == 0 else "g1_pelvis_sdf.npz"
        z = np.load(os.path.join(GEN, fn), allow_pickle=True)
        grid = np.ascontiguousarray(z["sdf_i16"].astype(np.int16).ravel(order="C"))
        lib.sidecar_upload_sdf(cid, grid.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)), grid.size)


def main():
    lib = ctypes.CDLL(build_lib())
    upload_sdfs(lib)
    fp = ctypes.POINTER(ctypes.c_float); ip = ctypes.POINTER(ctypes.c_int32)
    lib.sidecar_cluster_gaps.argtypes = [fp, fp, ip, ctypes.c_int]

    cpu = SidecarCPU()
    NP = len(cpu.lp_class)
    ji = {n: i for i, n in enumerate(HJCD_JOINT_ORDER)}
    neutral = np.zeros(29, np.float32); crouch = np.zeros(29, np.float32)
    for s in ("left", "right"):
        crouch[ji[f"{s}_hip_pitch_joint"]] = -0.6; crouch[ji[f"{s}_knee_joint"]] = 1.2
        crouch[ji[f"{s}_ankle_pitch_joint"]] = -0.6
    corpus = json.load(open(os.path.join(GEN, "g1_collision_corpus.json")))
    Q = np.stack([neutral, crouch] + [np.asarray(s["q"], np.float32) for s in corpus["samples"]])
    Bn = Q.shape[0]
    gpu = np.zeros((Bn, NP), np.float32); ev = np.zeros((Bn, NP), np.int32)
    lib.sidecar_cluster_gaps(Q.ctypes.data_as(fp), gpu.ctypes.data_as(fp), ev.ctypes.data_as(ip), Bn)

    cl_idx = [i for i, k in enumerate(cpu.lp_class) if k[0] == "cluster"]
    worst = 0.0; worst_where = None; ev_mismatch = 0; ev_max = 0
    for b in range(Bn):
        W, T = cpu.world_primitives(Q[b])
        for idx in cl_idx:
            _, cid, limbs = cpu.lp_class[idx]
            Tc = T[cpu.clusters[cid]["link"]]
            gaps, evs = [], 0
            for il in limbs:
                g, e, _ = cpu._cluster_gap(W[il], Tc, cid)
                gaps.append(g); evs += e
            ref = min(gaps)
            err = abs(gpu[b, idx] - ref) * 1000.0
            if err > worst:
                worst, worst_where = err, (b, idx, cid)
            if ev[b, idx] != evs:
                ev_mismatch += 1
            ev_max = max(ev_max, int(ev[b, idx]))

    print(f"cluster-SDF gap max err = {worst:.6f} mm over {Bn} configs x {len(cl_idx)} cluster pairs")
    print(f"  SDF eval-count mismatches = {ev_mismatch} (max evals/pair seen = {ev_max}, cap = 48)")
    print(f"  worst at {worst_where}")
    ok = worst < 1e-2 and ev_mismatch == 0 and ev_max <= 48
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
