"""Checkpoint 3D.1 benchmark: the dedicated collision-free seed generator vs the legacy stall kick.

Two sweeps, both on the exact seed pools that previously produced 0/132 recovery at B=2000 and
147-253 failures out of 256 on problems 1-3:

  A. candidate-pool sweep   R in {4, 8, 16, 32, 64}, per problem: recovery, timing breakdown,
                            which distribution component was selected, and the resulting IK yield
  B. mode sweep             off / final / hard(legacy reseed) / hard(new reseed) at B in
                            {100, 500, 1000, 2000}

Writes generated/benchmark_reseed.json.
Run: env PYTHONPATH= python3 benchmark/benchmark_reseed.py [--mujoco]
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
sys.path.insert(0, HERE)

import hjcdik  # noqa: E402
from benchmark_hard_mode import MujocoOracle, native_free, problem  # noqa: E402

POS_TOL, ORI_TOL = 0.02, 0.1
R_SWEEP = (4, 8, 16, 32, 64)


def solve(sq, tp, tq, mode, **kw):
    return hjcdik.solve(sq, tp, tq, seed=42, precision="float32",
                        position_tol=POS_TOL, orientation_tol=ORI_TOL,
                        self_collision_mode=mode, **kw)


def sweep_R(problems, B, mj=None):
    """Recovery and yield as a function of candidate-pool size, per problem."""
    rows = []
    for pi in problems:
        sq, tp, tq = problem(B, pi)
        base_coll = int((~native_free(sq)).sum())
        for R in R_SWEEP:
            for mode_id, label in ((0, "legacy"), (1, "generator")):
                kw = dict(diagnostics=True, collision_reseed_mode=mode_id,
                          collision_reseed_candidates=R, collision_reseed_rounds=2)
                solve(sq, tp, tq, "hard", **kw)                    # warm-up, discarded
                t0 = time.perf_counter()
                out = solve(sq, tp, tq, "hard", **kw)
                wall = (time.perf_counter() - t0) * 1e3
                h = out["self_collision"]
                q = np.asarray(out["joint_config"])
                succ = np.asarray(out["success"]).astype(bool)
                free = native_free(q)
                rec = h["recovered"]
                r = {
                    "problem": pi, "R": R, "reseed": label,
                    "seed_pool_colliding": base_coll,
                    "initially_colliding": h["initially_colliding"],
                    "recovered": rec,
                    "recovery_pct": 100.0 * rec / max(1, h["initially_colliding"]),
                    "unrecovered": h["seed_failures"],
                    "rounds_run": h["reseed_rounds_run"],
                    "candidates_checked": h["reseed_candidates_checked"],
                    "candidates_per_recovered": (h["reseed_candidates_checked"] / rec) if rec else None,
                    "gen_ms": h["reseed_gen_ms"], "check_ms": h["reseed_check_ms"],
                    "select_ms": h["reseed_select_ms"], "verify_ms": h["reseed_verify_ms"],
                    "total_reseed_ms": h["init_ms"],
                    "wall_ms": wall,
                    "selected_perturb": h["selected_perturb"],
                    "selected_nominal": h["selected_nominal"],
                    "selected_broad": h["selected_broad"],
                    "ik_success": int(succ.sum()),
                    "native_free": int(free.sum()),
                    "success_but_colliding": int((succ & ~free).sum()),
                }
                if mj is not None and label == "generator" and R == 16:
                    sel = np.flatnonzero(succ)[:150]
                    if len(sel):
                        c = mj.colliding(np.asarray(q, np.float64)[sel])
                        r["mujoco_checked"] = int(len(sel))
                        r["mujoco_colliding"] = int(c.sum())
                    else:
                        r["mujoco_checked"] = 0
                        r["mujoco_colliding"] = 0
                rows.append(r)
                print(f"  p{pi} R={R:3d} {label:10} coll={r['initially_colliding']:4d} "
                      f"rec={rec:4d} ({r['recovery_pct']:5.1f}%) unrec={r['unrecovered']:4d} "
                      f"reseed={r['total_reseed_ms']:7.1f}ms wall={wall:8.1f}ms "
                      f"succ={r['ik_success']:4d} free={r['native_free']:4d} "
                      f"sel(P/N/B)={h['selected_perturb']}/{h['selected_nominal']}/"
                      f"{h['selected_broad']}", flush=True)
    return rows


def sweep_modes(sizes, repeats, mj=None):
    rows = []
    for B in sizes:
        sq, tp, tq = problem(B)
        for label, mode, kw in (("off", "off", {}),
                                ("final", "final", {}),
                                ("hard_legacy", "hard", dict(collision_reseed_mode=0)),
                                ("hard_generator", "hard", dict(collision_reseed_mode=1))):
            solve(sq, tp, tq, mode, **kw)                          # warm-up
            walls = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                out = solve(sq, tp, tq, mode, **kw)
                walls.append((time.perf_counter() - t0) * 1e3)
            q = np.asarray(out["joint_config"])
            succ = np.asarray(out["success"]).astype(bool)
            free = native_free(q)
            r = {"batch": B, "mode": label,
                 "wall_ms_median": statistics.median(walls),
                 "coarse_kernel_ms": float(out.get("coarse_kernel_ms", 0.0)),
                 "lm_kernel_ms": float(out.get("lm_kernel_ms", 0.0)),
                 "ik_success": int(succ.sum()),
                 "native_free": int(free.sum()),
                 "success_and_free": int((succ & free).sum()),
                 "success_but_colliding": int((succ & ~free).sum())}
            sc = out.get("self_collision")
            if sc and sc.get("mode") == "hard":
                r.update({"initially_colliding": sc["initially_colliding"],
                          "recovered": sc["recovered"],
                          "seed_failures": sc["seed_failures"],
                          "reseed_ms": sc["init_ms"]})
            if mj is not None:
                sel = np.flatnonzero(succ)[:200]
                if len(sel):
                    c = mj.colliding(np.asarray(q, np.float64)[sel])
                    r["mujoco_checked"] = int(len(sel))
                    r["mujoco_colliding"] = int(c.sum())
                else:
                    r["mujoco_checked"] = 0
                    r["mujoco_colliding"] = 0
            rows.append(r)
            mjs = (f" mj={r['mujoco_colliding']}/{r['mujoco_checked']}"
                   if "mujoco_checked" in r else "")
            ex = (f" coll={r.get('initially_colliding','-')} rec={r.get('recovered','-')}"
                  f" fail={r.get('seed_failures','-')}" if "recovered" in r else "")
            print(f"  B={B:5d} {label:15} wall={r['wall_ms_median']:9.2f}ms "
                  f"succ={r['ik_success']:5d} free={r['native_free']:5d} "
                  f"succ&coll={r['success_but_colliding']:4d}{mjs}{ex}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="0,1,2,3")
    ap.add_argument("--B", type=int, default=256)
    ap.add_argument("--sizes", default="100,500,1000,2000")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--mujoco", action="store_true")
    a = ap.parse_args()

    hjcdik._ensure_self_collision_sidecar()
    mj = MujocoOracle() if a.mujoco else None

    print("=== A. candidate-pool sweep ===", flush=True)
    rows_R = sweep_R([int(x) for x in a.problems.split(",")], a.B, mj)
    print("=== B. mode sweep ===", flush=True)
    rows_M = sweep_modes([int(x) for x in a.sizes.split(",")], a.repeats, mj)

    out = os.path.join(GEN, "benchmark_reseed.json")
    with open(out, "w") as f:
        json.dump({"info": hjcdik.self_collision_info(), "B_for_R_sweep": a.B,
                   "R_sweep": rows_R, "mode_sweep": rows_M}, f, indent=1, default=float)
    print("wrote", out)


if __name__ == "__main__":
    main()
