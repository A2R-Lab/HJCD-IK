"""Stage A step 1: enumerate EVERY native self-collision deep-false-negative pair class across the
production IK distribution AND a broad diversified corpus -- so the GJK routing covers the complete
set, not one seed's sample.

A deep false negative = native (margin 0, the conservative standalone regime) calls the pose
self-collision-free while MuJoCo records a self-collision past the 5 mm self_clearance. For each,
records q, the MuJoCo link pair, penetration, and the native narrow phase currently routing that
pair. Groups by exact unordered link pair. Verifies GPU==CPU on every hit (a native MODEL gap, not
a GPU bug).

Writes generated/g1_selfcoll_fn_corpus.json.
Run: env PYTHONPATH= python3 collision_sidecar/mine_all_selfcoll_fn.py [--seeds 128] [--problems 8]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HJCD = os.path.dirname(HERE)
GEN = os.path.join(HJCD, "generated")
ROOT = os.path.dirname(os.path.dirname(HJCD))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HJCD, "tests"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "production"))

import hjcdik  # noqa: E402
from corpus import MujocoOracle  # noqa: E402
from collision_cpu import SidecarCPU  # noqa: E402
from urdf_model import HJCD_JOINT_ORDER  # noqa: E402

JI = {n: i for i, n in enumerate(HJCD_JOINT_ORDER)}
N = 29

ART = json.load(open(os.path.join(GEN, "g1_collision_sidecar.json")))
CHECKED = {tuple(sorted(p)) for p in ART["checked_link_pairs"]}
GJK = {tuple(sorted(p)) for p in ART["gjk_pairs"]}
CLUSTER = {"torso_link", "base_link"}


def phase(a, b):
    key = tuple(sorted((a, b)))
    if key not in CHECKED:
        return "NOT_CHECKED"
    if key in GJK:
        return "convex_gjk(EXACT)"
    if a in CLUSTER or b in CLUSTER:
        return "cluster_sdf(APPROX)"
    return "primitive(APPROX)"


# ---- left/right mirror (same convention as the elbow held-out builder) -----------------------
LEG_NEG = (1, 2, 5)
ARM_NEG = (1, 2, 4, 6)


def mirror(q):
    m = np.array(q, float)
    m[0:6], m[6:12] = q[6:12].copy(), q[0:6].copy()
    m[15:22], m[22:29] = q[22:29].copy(), q[15:22].copy()
    for i in LEG_NEG:
        m[i] *= -1
        m[6 + i] *= -1
    for i in ARM_NEG:
        m[15 + i] *= -1
        m[22 + i] *= -1
    m[12] *= -1
    m[13] *= -1
    return m


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


def production_configs(n_problems, seeds):
    """The actual production IK distribution: HJCD outputs on the generated assignment set."""
    from bouldering.route_io import load_feature_route
    from bouldering.mujoco_utils import build_climbing_scene, wide_base_bounds
    from bouldering.ik import IKSolver, SUCCESS_TOL
    from bouldering.features import affordance_by_name
    from bouldering.contacts.assignments import AssignmentGenerationConfig
    from pipeline_ik import PROD_IK_BACKEND, PROD_MAX_SOLUTIONS
    from example_ik import ROUTE, WALL_ANGLE, build_generated_problems
    import pipeline_ik as PI
    feats = load_feature_route(ROUTE, wall_angle_deg=None)
    wa = float(WALL_ANGLE)
    model, _ = build_climbing_scene(floating_base=True, wall_angle=wa, features=feats)
    aff = affordance_by_name(feats)
    probs, _g, _ = build_generated_problems(aff, model, AssignmentGenerationConfig())
    ik = IKSolver(model, use_orientation=False, backend=PROD_IK_BACKEND,
                  base_pos_bounds=wide_base_bounds(aff.values()))
    sols, _t, _m = PI.solve_assignments_batched(
        ik, probs, seeds_per_assignment=seeds, max_solutions=PROD_MAX_SOLUTIONS,
        wall_angle=wa, hjcd_self_collision_mode="off")
    out = [np.asarray(p["qpos"], float)[7:7 + N]
           for s in sols if s.feasible for p in s.poses if p["max_err"] < SUCCESS_TOL]
    return out


def broad_configs(rng, n):
    """Diverse reachable + crossed-limb + near-boundary configs, both sides, to stress every pair."""
    out = []
    for _ in range(n):
        q = rng.normal(0, 0.55, N)
        out.append(np.clip(q, -1.5, 1.5))
    # crossed arms across the chest, both sides
    for side, sgn in (("left", 1.0), ("right", -1.0)):
        for _ in range(n // 2):
            q = rng.normal(0, 0.3, N)
            q[JI[f"{side}_shoulder_roll_joint"]] = -sgn * rng.uniform(0.3, 1.6)
            q[JI[f"{side}_elbow_joint"]] = rng.uniform(0.3, 2.2)
            q[JI[f"{side}_shoulder_yaw_joint"]] = rng.uniform(-1.4, 1.4)
            q[JI["waist_yaw_joint"]] = rng.uniform(-0.5, 0.5)
            out.append(np.clip(q, -1.7, 1.7))
    # crossed / tucked legs and hips toward torso
    for _ in range(n // 2):
        q = rng.normal(0, 0.3, N)
        for s in ("left", "right"):
            q[JI[f"{s}_hip_pitch_joint"]] = -rng.uniform(0.5, 1.8)
            q[JI[f"{s}_hip_yaw_joint"]] = rng.uniform(-1.0, 1.0)
            q[JI[f"{s}_hip_roll_joint"]] = rng.uniform(-0.6, 0.6)
            q[JI[f"{s}_knee_joint"]] = rng.uniform(0.4, 2.2)
        out.append(np.clip(q, -1.9, 1.9))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=128)
    ap.add_argument("--problems", type=int, default=8)
    ap.add_argument("--broad", type=int, default=1500)
    ap.add_argument("--out", default=os.path.join(GEN, "g1_selfcoll_fn_corpus.json"))
    a = ap.parse_args()

    hjcdik._ensure_self_collision_sidecar()
    rng = np.random.default_rng(11)

    print("gathering production IK configs...", flush=True)
    prod = production_configs(a.problems, a.seeds)
    broad = broad_configs(rng, a.broad)
    allq = [np.asarray(q, float) for q in prod + broad]
    allq += [mirror(q) for q in allq]                    # both sides
    Q = np.stack(allq)
    print(f"  production {len(prod)}, broad {len(broad)}, +mirror -> {len(Q)} configs", flush=True)

    native = np.asarray(hjcdik._hjcdik.sidecar_full_check(
        np.ascontiguousarray(Q.astype(np.float32)), 0.0)).any(axis=1)
    print(f"  native (margin 0) colliding: {int(native.sum())}/{len(Q)}", flush=True)

    oracle = MujocoOracle()
    cpu = SidecarCPU()
    fn_pairs = collections.Counter()
    fn_examples = []
    gpu_cpu_disagree = 0
    checked = 0
    for i in np.flatnonzero(~native):                    # native says FREE
        checked += 1
        lab = oracle.label(Q[i])
        if not lab["colliding"]:
            continue
        depth = min(d for _, d in lab["pairs"])
        cpu_free = cpu.collision_free(np.asarray(Q[i], float))
        if not cpu_free:
            gpu_cpu_disagree += 1
        for lp, d in lab["pairs"]:
            fn_pairs[tuple(sorted(lp))] += 1
        fn_examples.append(dict(q=[float(v) for v in Q[i]],
                                pairs=[[p[0], p[1], round(d * 1e3, 2)] for p, d in
                                       [(tuple(sorted(lp)), d) for lp, d in lab["pairs"]]],
                                penetration_mm=round(depth * 1e3, 2),
                                cpu_agrees_free=bool(cpu_free)))
        if checked % 2000 == 0:
            print(f"    ...scanned {checked} native-free, {len(fn_examples)} deep FN so far",
                  flush=True)

    print(f"\n=== deep self-collision false-negative pair classes (native margin 0) ===")
    for pk, cnt in sorted(fn_pairs.items(), key=lambda kv: -kv[1]):
        print(f"  {pk[0]:28s} <-> {pk[1]:24s} n={cnt:4d}  phase={phase(*pk)}")
    print(f"\n  distinct pair classes: {len(fn_pairs)}")
    print(f"  deep-FN poses: {len(fn_examples)} / {len(Q)}")
    print(f"  GPU/CPU disagreements (would indicate a GPU bug, not a model gap): {gpu_cpu_disagree}")

    payload = dict(
        n_configs=len(Q), n_production=len(prod), n_broad=len(broad),
        native_margin_m=0.0, deep_fn_poses=len(fn_examples),
        gpu_cpu_disagreements=gpu_cpu_disagree,
        pair_classes={f"{a2} <-> {b2}": dict(count=c, phase=phase(a2, b2))
                      for (a2, b2), c in fn_pairs.items()},
        hashes=dict(hjcdik.self_collision_info()["hashes"]),
        joint_order=list(HJCD_JOINT_ORDER),
        examples=fn_examples[:200])
    with open(a.out, "w") as f:
        json.dump(payload, f)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
