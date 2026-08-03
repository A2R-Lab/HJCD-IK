"""Checkpoint 3D/3E A/B/C benchmark: self_collision_mode off vs final vs hard.

Identical assignments, seeds and warm-up sequence across the three modes at every batch size, so
the only difference measured is the mode. Writes generated/benchmark_hard_mode.json.

Run: env PYTHONPATH= python3 benchmark/benchmark_hard_mode.py [--sizes 1,10,100,500,1000,2000]
     [--mujoco]     also label every returned candidate with MuJoCo deep self-collision

MEASUREMENT DISCIPLINE. Device stage times (coarse / LM) come from the CUDA events of the SAME
invocation whose wall time is reported -- never from independently-timed medians, which is what
once produced "coarse + LM > end-to-end" rows. Wall time is a median over repeats after a warm-up
call that is discarded (it pays workspace growth and, in hard mode, the model bind).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HJCD = os.path.dirname(HERE)
GEN = os.path.join(HJCD, "generated")
sys.path.insert(0, os.path.join(HJCD, "collision_sidecar"))
sys.path.insert(0, os.path.join(HJCD, "tests"))

import hjcdik  # noqa: E402

POS_TOL, ORI_TOL = 0.02, 0.1
MODES = ("off", "final", "hard")


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


def problem(B, pi=0):
    """The SAME problem family the Checkpoint 3 test suites use, so numbers are comparable."""
    rng = np.random.default_rng(3)
    qrefs = np.clip(rng.normal(0, 0.35, (4, 29)), -1.2, 1.2)
    T = np.asarray(hjcdik.target_transforms(np.ascontiguousarray(qrefs)))[pi]
    tpos = np.ascontiguousarray(np.broadcast_to(T[:, :3, 3], (B, 4, 3)))
    tquat = np.ascontiguousarray(np.broadcast_to(
        np.stack([_mat2quat(T[k, :3, :3]) for k in range(4)]), (B, 4, 4)))
    seed_q = np.ascontiguousarray(
        qrefs[pi][None] + np.random.default_rng(100 + pi).normal(0, 0.2, (B, 29)))
    return seed_q, tpos, tquat


def native_free(q):
    q32 = np.ascontiguousarray(np.asarray(q, np.float32))
    return ~np.asarray(hjcdik._hjcdik.sidecar_full_check(q32, 0.0)).any(axis=1)


class MujocoOracle:
    """Deep self-collision per the repo-wide stance-graph rule: contacts with penetration depth
    beyond `min_depth_mm`, so limbs merely grazing are not counted as collisions."""

    def __init__(self, min_depth_mm=5.0):
        from corpus import MujocoOracle as _O  # noqa
        self._o = _O()
        self.min_depth_mm = min_depth_mm

    def colliding(self, q):
        out = np.zeros(len(q), bool)
        for i, qi in enumerate(q):
            lab = self._o.label(np.asarray(qi, np.float64))
            out[i] = bool(lab["colliding"])
        return out


def solve(sq, tp, tq, mode, top_k=3, diagnostics=False):
    return hjcdik.solve(sq, tp, tq, seed=42, precision="float32",
                        position_tol=POS_TOL, orientation_tol=ORI_TOL,
                        self_collision_mode=mode, collision_top_k=top_k,
                        diagnostics=diagnostics)


def bench_one(B, repeats, mj=None):
    sq, tp, tq = problem(B)
    row = {"batch": B, "modes": {}}
    for mode in MODES:
        solve(sq, tp, tq, mode)                       # warm-up, discarded
        walls = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            out = solve(sq, tp, tq, mode)
            walls.append((time.perf_counter() - t0) * 1e3)
        diag = solve(sq, tp, tq, mode, diagnostics=True)   # counters, not timed

        q = np.asarray(out["joint_config"])
        succ = np.asarray(out["success"]).astype(bool)
        free = native_free(q)
        m = {
            "wall_ms_median": statistics.median(walls),
            "wall_ms_min": min(walls),
            "coarse_kernel_ms": float(out.get("coarse_kernel_ms", 0.0)),
            "lm_kernel_ms": float(out.get("lm_kernel_ms", 0.0)),
            "returned_success": int(succ.sum()),
            "native_collision_free": int(free.sum()),
            "success_and_free": int((succ & free).sum()),
            "success_but_colliding": int((succ & ~free).sum()),
        }
        sc = out.get("self_collision")
        if sc:
            m["self_collision_check_ms"] = float(sc.get("kernel_ms", sc.get("lm_check_ms", 0.0)))
        if mode == "hard":
            d = diag["self_collision"]
            m.update({
                "init_ms": d["init_ms"],
                "initially_free": d["initially_free"],
                "initially_colliding": d["initially_colliding"],
                "reseed_attempts": d["reseed_attempts"],
                "recovered": d["recovered"],
                "seed_failures": d["seed_failures"],
                "lm_colliding": d["lm_colliding"],
                "used_collision_fallback": d["used_collision_fallback"],
                "fallback_success": d["fallback_success"],
                "unrecoverable": d["unrecoverable"],
                "proposals_checked": d["proposals_checked"],
                "proposals_rejected": d["proposals_rejected"],
                "all_k_colliding": d["all_k_colliding"],
                "accept_by_rank": d["accept_by_rank"],
                "gjk_pairs": d["gjk_pairs"],
                "gjk_iters": d["gjk_iters"],
                "nongjk_pairs": d["nongjk_pairs"],
                "trials_without_gjk": d["trials_without_gjk"],
                "perturbations_skipped": d["perturbations_skipped"],
                "reject_by_joint": d["reject_by_joint"],
            })
            p = max(1, d["proposals_checked"])
            m["gjk_pairs_per_trial"] = d["gjk_pairs"] / p
            m["nongjk_pairs_per_trial"] = d["nongjk_pairs"] / p
            m["gjk_iters_per_gjk_pair"] = d["gjk_iters"] / max(1, d["gjk_pairs"])
            m["pct_trials_without_gjk"] = 100.0 * d["trials_without_gjk"] / p
            m["collision_reject_rate"] = 100.0 * d["proposals_rejected"] / p
        if mj is not None:
            sel = np.flatnonzero(succ)[:200]          # bounded: MuJoCo labelling is host-side
            if len(sel):
                mjc = mj.colliding(np.asarray(q, np.float64)[sel])
                m["mujoco_checked"] = int(len(sel))
                m["mujoco_colliding_among_successes"] = int(mjc.sum())
            else:
                m["mujoco_checked"] = 0
                m["mujoco_colliding_among_successes"] = 0
        row["modes"][mode] = m
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1,10,100,500,1000,2000")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--mujoco", action="store_true")
    a = ap.parse_args()

    hjcdik._ensure_self_collision_sidecar()
    mj = MujocoOracle() if a.mujoco else None

    info = hjcdik.self_collision_info()
    res = {"info": {k: v for k, v in info.items()},
           "pos_tol": POS_TOL, "ori_tol": ORI_TOL, "repeats": a.repeats, "rows": []}
    for B in [int(x) for x in a.sizes.split(",")]:
        print(f"--- B = {B} ---", flush=True)
        row = bench_one(B, a.repeats, mj)
        res["rows"].append(row)
        for mode in MODES:
            m = row["modes"][mode]
            extra = ""
            if mode == "hard":
                extra = (f" init={m['init_ms']:.1f}ms seedfail={m['seed_failures']}"
                         f" rank={m['accept_by_rank'][:3]} fb={m['used_collision_fallback']}")
            mjs = ""
            if "mujoco_colliding_among_successes" in m:
                mjs = f" mj_coll={m['mujoco_colliding_among_successes']}/{m['mujoco_checked']}"
            print(f"  {mode:5} wall={m['wall_ms_median']:8.2f}ms coarse={m['coarse_kernel_ms']:7.2f}"
                  f" lm={m['lm_kernel_ms']:7.2f} succ={m['returned_success']:5d}"
                  f" free={m['native_collision_free']:5d}"
                  f" succ&coll={m['success_but_colliding']:4d}{mjs}{extra}", flush=True)

    out = os.path.join(GEN, "benchmark_hard_mode.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("wrote", out)


if __name__ == "__main__":
    main()
