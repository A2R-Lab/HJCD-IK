"""Checkpoint 3C.2 sec.4 -- NEW independent leg-focused held-out corpus (NOT used to tune geometry).

Distinct seeds/assignments from the 3C.1 held-out and the tuning set. Categories: crossed-leg,
folded-knee, left-over-right, right-over-left, near-boundary-free, ordinary. Records native + MuJoCo
verdicts, pairs, depth per candidate. Writes g1_heldout_leg_corpus.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HJCD = os.path.dirname(HERE)
GEN = os.path.join(HJCD, "generated")
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HJCD, "benchmark"))
import hjcdik  # noqa: E402
from corpus import MujocoOracle  # noqa: E402
from urdf_model import HJCD_JOINT_ORDER  # noqa: E402
from benchmark_collision_integration import mat2quat  # noqa: E402

JI = {n: i for i, n in enumerate(HJCD_JOINT_ORDER)}


def qref(kind, rng):
    q = np.clip(rng.normal(0, 0.3, 29), -1.0, 1.0)
    if kind == "crossed_leg":                       # both hip_yaw inward -> legs cross
        q[JI["left_hip_yaw_joint"]] = rng.uniform(0.4, 1.0)
        q[JI["right_hip_yaw_joint"]] = rng.uniform(-1.0, -0.4)
        q[JI["left_hip_roll_joint"]] = rng.uniform(-0.5, 0.0)
        q[JI["right_hip_roll_joint"]] = rng.uniform(0.0, 0.5)
    elif kind == "folded_knee":                     # deep knee flexion + hip pitch
        for s in ("left", "right"):
            q[JI[f"{s}_knee_joint"]] = rng.uniform(1.4, 2.2)
            q[JI[f"{s}_hip_pitch_joint"]] = rng.uniform(-1.2, -0.6)
    elif kind == "left_over_right":
        q[JI["left_hip_yaw_joint"]] = rng.uniform(0.6, 1.2)
        q[JI["left_hip_roll_joint"]] = rng.uniform(-0.7, -0.2)
        q[JI["left_knee_joint"]] = rng.uniform(0.6, 1.4)
    elif kind == "right_over_left":
        q[JI["right_hip_yaw_joint"]] = rng.uniform(-1.2, -0.6)
        q[JI["right_hip_roll_joint"]] = rng.uniform(0.2, 0.7)
        q[JI["right_knee_joint"]] = rng.uniform(0.6, 1.4)
    elif kind == "near_boundary":                   # legs close but mostly free
        q[JI["left_hip_yaw_joint"]] = rng.uniform(0.1, 0.4)
        q[JI["right_hip_yaw_joint"]] = rng.uniform(-0.4, -0.1)
    return np.clip(q, -1.4, 1.4)


def main():
    oracle = MujocoOracle()
    hjcdik._ensure_self_collision_sidecar()
    kw = dict(precision="float32", position_tol=0.02, orientation_tol=0.1)
    specs = [(4001, "crossed_leg", 500), (4002, "folded_knee", 500), (4003, "left_over_right", 500),
             (4004, "right_over_left", 500), (4005, "near_boundary", 500), (4006, "ordinary", 500)]
    rows = []
    for sd, kind, B in specs:
        rng = np.random.default_rng(sd)
        qr = qref(kind if kind != "ordinary" else "x", rng) if kind != "ordinary" else np.clip(rng.normal(0, 0.35, 29), -1.2, 1.2)
        T = np.asarray(hjcdik.target_transforms(qr[None]))[0]
        tpos = np.ascontiguousarray(np.broadcast_to(T[:, :3, 3], (B, 4, 3)))
        tquat = np.ascontiguousarray(np.broadcast_to(np.stack([mat2quat(T[k, :3, :3]) for k in range(4)]), (B, 4, 4)))
        seed_q = np.ascontiguousarray(qr[None] + np.random.default_rng(sd + 5).normal(0, 0.22, (B, 29)))
        out = hjcdik.solve(seed_q, tpos, tquat, seed=sd, self_collision_mode="off", **kw)
        q = np.ascontiguousarray(np.asarray(out["joint_config"], np.float32))
        succ = np.asarray(out["success"]).astype(bool)
        V = np.asarray(hjcdik._hjcdik.sidecar_full_check(q, 0.0)); nat = V.any(axis=1)
        for bi in range(B):
            lab = oracle.label(np.asarray(out["joint_config"][bi], float))
            rows.append(dict(spec=sd, kind=kind, candidate=int(bi), success=bool(succ[bi]),
                             q=[float(x) for x in out["joint_config"][bi]],
                             native_colliding=bool(nat[bi]), mujoco_colliding=bool(lab["colliding"]),
                             mujoco_min_depth_mm=float(min((d for _, d in lab["pairs"]), default=0.0)),
                             mujoco_pairs=[p[0] for p in lab["pairs"]]))
    json.dump(dict(n=len(rows), specs=[(s, k) for s, k, _ in specs],
                   sign_convention="native gap<0 = collision; MuJoCo depth NEGATIVE",
                   rows=rows), open(os.path.join(GEN, "g1_heldout_leg_corpus.json"), "w"))
    # summary
    import collections
    mj = np.array([r["mujoco_colliding"] for r in rows]); nat = np.array([r["native_colliding"] for r in rows])
    dep = np.array([r["mujoco_min_depth_mm"] for r in rows]); succ = np.array([r["success"] for r in rows])
    deep = mj & (dep < -5)
    deepFN = int((deep & ~nat).sum()); deepFN_succ = int((deep & ~nat & succ).sum())
    fnp = collections.Counter()
    for r in rows:
        if r["mujoco_colliding"] and not r["native_colliding"] and r["mujoco_min_depth_mm"] < -5:
            for p in r["mujoco_pairs"]: fnp[tuple(p)] += 1
    print(f"NEW leg held-out: {len(rows)} candidates, {int(mj.sum())} MuJoCo-colliding, {int(deep.sum())} deep")
    print(f"  global DEEP FN: {deepFN}  (among retained successes: {deepFN_succ})")
    print(f"  deep-FN pairs: {dict(fnp)}")
    print(f"  wrote generated/g1_heldout_leg_corpus.json")


if __name__ == "__main__":
    main()
