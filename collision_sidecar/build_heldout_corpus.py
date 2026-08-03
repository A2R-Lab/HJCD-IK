"""Checkpoint 3C.1 sec.4 -- INDEPENDENT held-out HJCD-output corpus (NOT used to tune geometry).

Different seeds + assignments from the tuning set, biased toward arm-across-chest / near the
shoulder_yaw<->torso boundary, plus ordinary configs. For every candidate records the native sidecar
verdict, MuJoCo verdict, colliding pairs, penetration depth, and per-pair narrow-phase info. Used
only to MEASURE generalization (was the failure localized or systematic). Writes g1_heldout_corpus.json.
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
ARM = [JI[f"{s}_{j}_joint"] for s in ("left", "right")
       for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")]


def arm_across_chest_qref(rng):
    """Reference config biased to bring the forearms across the torso (stresses shoulder_yaw<->torso)."""
    q = np.clip(rng.normal(0, 0.3, 29), -1.0, 1.0)
    q[JI["left_shoulder_roll_joint"]] = rng.uniform(-1.4, -0.6)     # arms inward
    q[JI["right_shoulder_roll_joint"]] = rng.uniform(0.6, 1.4)
    for s in ("left", "right"):
        q[JI[f"{s}_shoulder_yaw_joint"]] = rng.uniform(-1.2, 1.2)
        q[JI[f"{s}_elbow_joint"]] = rng.uniform(0.4, 1.6)
    return q


def main():
    oracle = MujocoOracle()
    hjcdik._ensure_self_collision_sidecar()
    kw = dict(precision="float32", position_tol=0.02, orientation_tol=0.1)
    rows = []
    # held-out problems: seeds/assignments DISTINCT from the tuning set (rng 3/101, seed 42)
    specs = [(1001, 700, "arm_across", True), (1002, 700, "arm_across", True),
             (2001, 500, "ordinary", False), (2002, 500, "ordinary", False),
             (3001, 600, "arm_across", True)]
    for sd, B, kind, arm in specs:
        rng = np.random.default_rng(sd)
        qref = arm_across_chest_qref(rng) if arm else np.clip(rng.normal(0, 0.35, 29), -1.2, 1.2)
        T = np.asarray(hjcdik.target_transforms(qref[None]))[0]
        tpos = np.ascontiguousarray(np.broadcast_to(T[:, :3, 3], (B, 4, 3)))
        tquat = np.ascontiguousarray(np.broadcast_to(np.stack([mat2quat(T[k, :3, :3]) for k in range(4)]), (B, 4, 4)))
        seed_q = np.ascontiguousarray(qref[None] + np.random.default_rng(sd + 7).normal(0, 0.25, (B, 29)))
        out = hjcdik.solve(seed_q, tpos, tquat, seed=sd, self_collision_mode="off", **kw)
        q = np.ascontiguousarray(np.asarray(out["joint_config"], np.float32))
        succ = np.asarray(out["success"]).astype(bool)
        V = np.asarray(hjcdik._hjcdik.sidecar_full_check(q, 0.0))
        native_coll = V.any(axis=1)
        for bi in range(B):
            lab = oracle.label(np.asarray(out["joint_config"][bi], float))
            depth = float(min((d for _, d in lab["pairs"]), default=0.0))
            rows.append(dict(spec=sd, kind=kind, candidate=int(bi), success=bool(succ[bi]),
                             q=[float(x) for x in out["joint_config"][bi]],
                             native_colliding=bool(native_coll[bi]),
                             mujoco_colliding=bool(lab["colliding"]),
                             mujoco_min_depth_mm=depth,
                             mujoco_pairs=[p[0] for p in lab["pairs"]]))
    out = dict(n=len(rows), n_specs=len(specs),
               sign_convention="penetration depth NEGATIVE (mm); MuJoCo colliding if any pair depth<0 past self_clearance",
               note="held-out: distinct seeds/assignments; NOT used for geometry tuning",
               rows=rows)
    json.dump(out, open(os.path.join(GEN, "g1_heldout_corpus.json"), "w"))
    # summary
    R = rows
    mj = np.array([r["mujoco_colliding"] for r in R])
    nat = np.array([r["native_colliding"] for r in R])
    depth = np.array([r["mujoco_min_depth_mm"] for r in R])
    deep = mj & (depth < -5.0)
    sh = np.array([any("shoulder_yaw" in p for p in r["mujoco_pairs"]) for r in R])
    FN = int((mj & ~nat).sum()); deepFN = int((deep & ~nat).sum())
    shFN = int((mj & sh & ~nat).sum())
    print(f"held-out corpus: {len(R)} candidates, {int(mj.sum())} MuJoCo-colliding "
          f"({int(deep.sum())} deep >5mm), {int(sh.sum())} shoulder_yaw-involved")
    print(f"  native FN (MuJoCo-colliding, sidecar-free): {FN}")
    print(f"  DEEP FN (>5mm): {deepFN}")
    print(f"  shoulder_yaw<->torso FN: {shFN}")
    print(f"  wrote generated/g1_heldout_corpus.json")


if __name__ == "__main__":
    main()
