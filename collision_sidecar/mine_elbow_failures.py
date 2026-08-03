"""Checkpoint 3C.3, step 1: mine every deep MuJoCo false negative from hard-mode outputs.

A "deep false negative" is a configuration the native sidecar calls collision-free while MuJoCo
records a self-collision deeper than the repo-wide 5 mm `self_clearance` (collision.py records a
self_collision issue only when `dist < -self_clearance`, and negative == penetration).

For each failure this records everything the fix has to be designed against -- NOT just the link
pair: q, the problem it came from, the row, the MuJoCo geom AND link pair, the penetration, the
narrow phase the native checker currently routes that pair to, and the signed gap it reported.

Writes generated/g1_elbow_hard_negatives.json.
Run: env PYTHONPATH= python3 collision_sidecar/mine_elbow_failures.py [--problems 0,1,2,3] [--B 256]
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

ART = json.load(open(os.path.join(GEN, "g1_collision_sidecar.json")))
PAIRS = [tuple(p) for p in ART["checked_link_pairs"]]
GJK = {tuple(sorted(p)) for p in ART["gjk_pairs"]}
CLUSTER_LINKS = set(ART.get("cluster_links", {}).keys()) if isinstance(
    ART.get("cluster_links"), dict) else set()


def narrow_phase(pair):
    """Which narrow phase the CURRENT native model routes an unordered link pair to."""
    key = tuple(sorted(pair))
    if key in GJK:
        return "convex_gjk"
    for cl in ART.get("clusters", {}) if isinstance(ART.get("clusters"), dict) else {}:
        pass
    # cluster membership: torso/pelvis are the two rigid SDF clusters
    if any(l in ("torso_link", "pelvis") or "torso" in l or "pelvis" in l for l in pair):
        return "cluster_sdf"
    return "primitive"


def pair_gaps(Q):
    """Current native signed gap per checked pair, from the three narrow-phase probes."""
    Q = np.ascontiguousarray(np.asarray(Q, np.float32))
    prim = np.asarray(hjcdik._hjcdik.sidecar_prim_gaps(Q))
    clus, _ev = hjcdik._hjcdik.sidecar_cluster_gaps(Q)
    clus = np.asarray(clus)
    g = np.minimum(prim, clus)                    # +inf where the phase does not apply
    gjk, _it = hjcdik._hjcdik.sidecar_gjk_gaps(Q)
    return g, np.asarray(gjk)


def _problem(pi, B):
    from test_collision_integration import _problem as P
    return P(pi, B)


def _solve(sq, tp, tq, **kw):
    from test_collision_integration import _solve as S
    return S(sq, tp, tq, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="0,1,2,3")
    ap.add_argument("--B", type=int, default=256)
    ap.add_argument("--cap", type=int, default=256, help="MuJoCo labels per (problem, mode) cell")
    ap.add_argument("--out", default=os.path.join(GEN, "g1_elbow_hard_negatives.json"))
    a = ap.parse_args()

    hjcdik._ensure_self_collision_sidecar()
    oracle = MujocoOracle()
    recs = []

    for pi in [int(x) for x in a.problems.split(",")]:
        sq, tp, tq = _problem(pi, a.B)
        for mode, kw in (("hard", dict(collision_reseed_candidates=16)),
                         ("final", {})):
            out = _solve(sq, tp, tq, self_collision_mode=mode, **kw)
            q = np.asarray(out["joint_config"], np.float64)
            q32 = np.ascontiguousarray(q.astype(np.float32))
            verdict = np.asarray(hjcdik._hjcdik.sidecar_full_check(q32, 0.0))
            native_free = ~verdict.any(axis=1)
            succ = np.asarray(out["success"]).astype(bool)

            rows = np.flatnonzero(native_free)[: a.cap]
            if len(rows) == 0:
                continue
            gaps, gjk_gaps = pair_gaps(q32[rows])
            for k, i in enumerate(rows):
                lab = oracle.label(q[i])
                if not lab["colliding"]:
                    continue
                res = oracle.last_issues if hasattr(oracle, "last_issues") else None
                for (lp, depth) in lab["pairs"]:
                    lp = tuple(lp)
                    pi_idx = next((n for n, p in enumerate(PAIRS)
                                   if tuple(sorted(p)) == tuple(sorted(lp))), None)
                    recs.append(dict(
                        q=[float(v) for v in q[i]],
                        problem=int(pi), mode=mode, row=int(i),
                        native_success=bool(succ[i]),
                        mujoco_link_pair=sorted(lp),
                        mujoco_penetration_mm=float(depth),
                        checked_pair_index=pi_idx,
                        native_narrow_phase=narrow_phase(lp) if pi_idx is not None else "NOT_CHECKED",
                        native_signed_gap_m=(float(gaps[k, pi_idx])
                                             if pi_idx is not None and np.isfinite(gaps[k, pi_idx])
                                             else None),
                    ))
            print(f"  p{pi} {mode:5}: scanned {len(rows)} native-free, "
                  f"cumulative failures {len(recs)}", flush=True)

    # ---- group by exact unordered link pair -------------------------------------------------
    groups = {}
    for r in recs:
        key = " <-> ".join(r["mujoco_link_pair"])
        g = groups.setdefault(key, dict(count=0, max_depth_mm=0.0, phases=set(), problems=set()))
        g["count"] += 1
        g["max_depth_mm"] = min(g["max_depth_mm"], r["mujoco_penetration_mm"])
        g["phases"].add(r["native_narrow_phase"])
        g["problems"].add(r["problem"])
    for k in groups:
        groups[k]["phases"] = sorted(groups[k]["phases"])
        groups[k]["problems"] = sorted(groups[k]["problems"])

    print("\n=== deep false negatives grouped by unordered link pair ===")
    for k, g in sorted(groups.items(), key=lambda kv: -kv[1]["count"]):
        print(f"  {k:52s} n={g['count']:3d}  deepest={g['max_depth_mm']:7.2f} mm  "
              f"phase={g['phases']}  problems={g['problems']}")

    uniq_q = {tuple(np.round(r["q"], 9)) for r in recs}
    payload = dict(
        n_records=len(recs), n_unique_configs=len(uniq_q),
        grouped={k: v for k, v in groups.items()},
        self_clearance_mm=5.0,
        hashes=dict(hjcdik.self_collision_info()["hashes"]),
        joint_order=list(HJCD_JOINT_ORDER),
        records=recs,
    )
    with open(a.out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\nwrote {a.out}  ({len(recs)} records, {len(uniq_q)} unique configurations)")


if __name__ == "__main__":
    main()
