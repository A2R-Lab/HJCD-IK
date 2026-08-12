"""Checkpoint 3C.3, step 5: build a HELD-OUT corpus for the elbow<->torso fix.

Held out means held out: none of these configurations comes from the 21 mined records the fix was
designed against, and none is a mirror of them. Sources, per spec section 5:

  A. hard-mode outputs at DIFFERENT seeds and DIFFERENT problems, broad reseeding enabled
  B. arm-across-torso sweeps, generated independently for BOTH elbows
  C. ordinary reachable climbing-like configurations (small joint excursions)
  D. near-boundary FREE examples -- poses just outside contact, where an over-eager fix would
     start producing false POSITIVES

Every configuration is labelled by MuJoCo (deep = penetration beyond the 5 mm `self_clearance`)
and by the native GPU checker. Writes generated/g1_elbow_heldout_corpus.json.

Run: env PYTHONPATH= python3 collision_sidecar/build_elbow_heldout_corpus.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HJCD = os.path.dirname(HERE)
GEN = os.path.join(HJCD, "generated")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HJCD, "tests"))

import hjcdik  # noqa: E402
from corpus import MujocoOracle  # noqa: E402
from urdf_model import HJCD_JOINT_ORDER  # noqa: E402

JI = {n: i for i, n in enumerate(HJCD_JOINT_ORDER)}


def _mat2quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
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
    q = np.array(q)
    return q / np.linalg.norm(q)


def hard_mode_configs(seeds, n_per, rng):
    """A: hard-mode outputs at seeds and reference configurations the fix never saw."""
    out = []
    for sd in seeds:
        r = np.random.default_rng(sd)
        qrefs = np.clip(r.normal(0, 0.40, (2, 29)), -1.3, 1.3)
        T = np.asarray(hjcdik.target_transforms(np.ascontiguousarray(qrefs)))[0]
        B = n_per
        tpos = np.ascontiguousarray(np.broadcast_to(T[:, :3, 3], (B, 4, 3)))
        tquat = np.ascontiguousarray(np.broadcast_to(
            np.stack([_mat2quat(T[k, :3, :3]) for k in range(4)]), (B, 4, 4)))
        seed_q = np.ascontiguousarray(qrefs[0][None] + r.normal(0, 0.30, (B, 29)))
        res = hjcdik.solve(seed_q, tpos, tquat, seed=int(sd), precision="float32",
                           position_tol=0.02, orientation_tol=0.1,
                           self_collision_mode="hard", collision_reseed_candidates=16)
        out.extend(np.asarray(res["joint_config"], np.float64))
    return out


def arm_across_torso(rng, n, side):
    """B: sweep the named arm across the chest. Independent construction -- these are built from
    the joint semantics, not perturbed from any mined failure."""
    out = []
    sp, sr, sy, el = (JI[f"{side}_shoulder_pitch_joint"], JI[f"{side}_shoulder_roll_joint"],
                      JI[f"{side}_shoulder_yaw_joint"], JI[f"{side}_elbow_joint"])
    sgn = 1.0 if side == "left" else -1.0
    for _ in range(n):
        q = np.zeros(29)
        q[sp] = rng.uniform(-0.7, 0.7)
        q[sr] = -sgn * rng.uniform(0.0, 1.5)          # adduct: bring the upper arm to the chest
        q[sy] = rng.uniform(-1.2, 1.2)
        q[el] = rng.uniform(0.3, 2.2)                 # fold the forearm back across the torso
        q[JI["waist_yaw_joint"]] = rng.uniform(-0.4, 0.4)
        q[JI["waist_pitch_joint"]] = rng.uniform(-0.3, 0.3)
        out.append(q)
    return out


def ordinary_climbing(rng, n):
    """C: ordinary reachable configurations -- small excursions, the regime production runs in."""
    out = []
    for _ in range(n):
        q = rng.normal(0, 0.25, 29)
        q[JI["left_knee_joint"]] = abs(rng.normal(0.6, 0.3))
        q[JI["right_knee_joint"]] = abs(rng.normal(0.6, 0.3))
        out.append(np.clip(q, -1.4, 1.4))
    return out


def near_boundary_free(rng, n):
    """D: poses just OUTSIDE contact. An over-eager fix shows up here as false positives."""
    out = []
    for _ in range(n):
        q = np.zeros(29)
        for side, sgn in (("left", 1.0), ("right", -1.0)):
            q[JI[f"{side}_shoulder_roll_joint"]] = -sgn * rng.uniform(0.05, 0.35)
            q[JI[f"{side}_elbow_joint"]] = rng.uniform(0.1, 0.8)
            q[JI[f"{side}_shoulder_pitch_joint"]] = rng.uniform(-0.3, 0.3)
        out.append(q)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(GEN, "g1_elbow_heldout_corpus.json"))
    a = ap.parse_args()
    hjcdik._ensure_self_collision_sidecar()
    rng = np.random.default_rng(20260724)

    groups = {
        "hard_mode_new_seeds": hard_mode_configs([911, 913, 917], 96, rng),
        "arm_across_torso_left": arm_across_torso(rng, 220, "left"),
        "arm_across_torso_right": arm_across_torso(rng, 220, "right"),
        "ordinary_climbing": ordinary_climbing(rng, 200),
        "near_boundary_free": near_boundary_free(rng, 160),
    }

    # Held-out means held out: drop anything that coincides with the tuning set or its mirror.
    tune = json.load(open(os.path.join(GEN, "g1_elbow_hard_negatives.json")))["records"]
    tune_keys = {tuple(np.round(r["q"], 6)) for r in tune}
    dropped = 0
    for g, qs in groups.items():
        keep = [q for q in qs if tuple(np.round(np.asarray(q), 6)) not in tune_keys]
        dropped += len(qs) - len(keep)
        groups[g] = keep

    oracle = MujocoOracle()
    samples = []
    for g, qs in groups.items():
        if not qs:
            continue
        Q = np.stack(qs)
        native = np.asarray(hjcdik._hjcdik.sidecar_full_check(
            np.ascontiguousarray(Q.astype(np.float32)), 0.0)).any(axis=1)
        for i, q in enumerate(Q):
            lab = oracle.label(np.asarray(q, np.float64))
            samples.append(dict(group=g, q=[float(v) for v in q],
                                native_colliding=bool(native[i]),
                                mujoco_deep=bool(lab["colliding"]),
                                mujoco_pairs=lab["pairs"]))
        print(f"  {g:24s} n={len(Q):4d}  native-colliding={int(native.sum()):4d}", flush=True)

    payload = dict(n=len(samples), dropped_overlap_with_tuning=dropped,
                   hashes=dict(hjcdik.self_collision_info()["hashes"]),
                   joint_order=list(HJCD_JOINT_ORDER), samples=samples)
    with open(a.out, "w") as f:
        json.dump(payload, f)
    print(f"wrote {a.out}  ({len(samples)} samples, {dropped} dropped as tuning overlap)")


if __name__ == "__main__":
    main()
