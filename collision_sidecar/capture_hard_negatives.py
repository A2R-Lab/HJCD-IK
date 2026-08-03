"""Checkpoint 3C.1 sec.3 -- save the shoulder_yaw<->torso hard negatives as a deterministic
tuning set. These are HJCD-output configs the torso-SDF proxy missed but MuJoCo flags. Records q,
problem/seed id, MuJoCo geom/link pair + depth, and the PRE-FIX sidecar verdict. Writes
generated/g1_hard_negatives.json.  (Run BEFORE the geometry fix to capture the pre-fix verdict.)
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
from benchmark_collision_integration import problem  # noqa: E402


def main():
    oracle = MujocoOracle()
    kw = dict(seed=42, precision="float32", position_tol=0.02, orientation_tol=0.1)
    hard = []
    for B in (2000,):                                   # the batch the FN were discovered in
        sq, tp, tq = problem(B)
        fin = hjcdik.solve(sq, tp, tq, self_collision_mode="final", **kw)
        q = np.ascontiguousarray(np.asarray(fin["joint_config"], np.float32))
        V = np.asarray(hjcdik._hjcdik.sidecar_full_check(q, 0.0))
        natfree = np.nonzero(~V.any(axis=1))[0]
        for bi in natfree:
            lab = oracle.label(np.asarray(fin["joint_config"][bi], float))
            if lab["colliding"]:
                hard.append(dict(
                    problem=f"problem({B})", batch=B, candidate=int(bi),
                    q=[float(x) for x in fin["joint_config"][bi]],
                    mujoco_pairs=lab["pairs"],
                    mujoco_min_depth_mm=float(min(d for _, d in lab["pairs"])),
                    prev_sidecar_verdict="free",         # torso-SDF proxy missed it
                    prev_narrow_phase="cluster_sdf(TORSO)",
                    new_gjk_verdict=None))               # filled by CPU validation after the fix
    out = dict(n=len(hard), sign_convention="penetration depth NEGATIVE (mm); colliding if depth<0",
               source="HJCD final-mode outputs, problem(2000), seed=42, tol=(0.02,0.1)",
               hard_negatives=hard)
    json.dump(out, open(os.path.join(GEN, "g1_hard_negatives.json"), "w"), indent=2)
    print(f"captured {len(hard)} hard negatives -> generated/g1_hard_negatives.json")
    for h in hard:
        print(f"  cand {h['candidate']:4d}  depth {h['mujoco_min_depth_mm']:6.2f}mm  "
              f"pairs {[p[0] for p in h['mujoco_pairs']]}")


if __name__ == "__main__":
    main()
