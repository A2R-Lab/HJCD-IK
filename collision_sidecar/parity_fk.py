"""Stage 1 parity: CUDA sidecar FK vs the accepted CPU URDF oracle (Checkpoint 2).

Builds the sidecar shared lib, runs device FK on neutral + crouch + all 289 corpus configs,
compares every link's 4x4 world transform against urdf_model.URDFModel.fk. Reports max mm error.
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

NL = 40  # N_LINKS (asserted below)


def build_lib():
    os.makedirs(SCR, exist_ok=True)
    so = os.path.join(SCR, "libsidecar.so")
    src = os.path.join(HJCD, "src", "collision_sidecar.cu")
    cmd = ["/usr/local/cuda/bin/nvcc", "-std=c++17", "-arch=sm_89", "-O2",
           "--compiler-options", "-fPIC", "-shared",
           "-I", GEN, "-I", os.path.join(HJCD, "src"), src, "-o", so]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr); raise SystemExit("nvcc build failed")
    return so


def bfs_order(model):
    order, frontier = [], [model.root_link]
    while frontier:
        lk = frontier.pop(0); order.append(lk)
        for jn in model.children_joints.get(lk, ()):
            frontier.append(model.joint_by_name[jn].child)
    return order


def main():
    model = parse_urdf(os.path.join(HJCD, "csrc", "urdf", "g1_29dof_rev_1_0.urdf"))
    order = bfs_order(model)
    assert len(order) == NL, (len(order), NL)

    # config set: neutral, crouch, all corpus
    ji = {n: i for i, n in enumerate(HJCD_JOINT_ORDER)}
    neutral = np.zeros(29, np.float32)
    crouch = np.zeros(29, np.float32)
    for side in ("left", "right"):
        crouch[ji[f"{side}_hip_pitch_joint"]] = -0.6
        crouch[ji[f"{side}_knee_joint"]] = 1.2
        crouch[ji[f"{side}_ankle_pitch_joint"]] = -0.6
    corpus = json.load(open(os.path.join(GEN, "g1_collision_corpus.json")))
    assert corpus["hjcd_joint_order"] == list(HJCD_JOINT_ORDER)
    labels = ["neutral", "crouch"] + [f"corpus[{i}]:{s['category']}" for i, s in enumerate(corpus["samples"])]
    Q = np.stack([neutral, crouch] + [np.asarray(s["q"], np.float32) for s in corpus["samples"]])
    B = Q.shape[0]

    lib = ctypes.CDLL(build_lib())
    lib.sidecar_fk_batch.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    Tg = np.zeros((B, NL, 16), np.float32)
    lib.sidecar_fk_batch(Q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                         Tg.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), B)

    worst = 0.0; worst_where = None
    per_cfg_max = np.zeros(B)
    for b in range(B):
        Tcpu = model.fk(model.q_vector_to_names(Q[b]))
        for L, lk in enumerate(order):
            g = Tg[b, L].reshape(4, 4, order="F")  # column-major -> 4x4
            err_mm = np.abs(g[:3, 3] - Tcpu[lk][:3, 3]).max() * 1000.0
            rot_mm = np.abs(g[:3, :3] - Tcpu[lk][:3, :3]).max() * 1000.0  # unitless*1000 scale
            e = max(err_mm, rot_mm)
            per_cfg_max[b] = max(per_cfg_max[b], e)
            if e > worst:
                worst, worst_where = e, (labels[b], lk)

    print(f"FK parity over {B} configs x {NL} links = {B*NL} transforms")
    print(f"  neutral  max err = {per_cfg_max[0]:.6f} mm")
    print(f"  crouch   max err = {per_cfg_max[1]:.6f} mm")
    print(f"  corpus   max err = {per_cfg_max[2:].max():.6f} mm  (mean {per_cfg_max[2:].mean():.6f})")
    print(f"  WORST    = {worst:.6f} mm  at {worst_where}")
    ok = worst < 1e-2   # 0.01 mm tolerance (FP32 vs FP64 oracle)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
