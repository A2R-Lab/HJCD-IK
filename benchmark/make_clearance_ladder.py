#!/usr/bin/env python3
"""Derive tighter-clearance variants of MotionBenchMaker problem sets (the "clearance ladder").

Every cuboid obstacle grows by `delta` on each face (dims += 2*delta), so the free space around the dataset's
goal shrinks by exactly delta everywhere; goals and cylinders are untouched (the r = 1 cm cylinders are the
grasp objects and the r = 5 cm ones are table legs, growing either would change the task, not the clearance).
A problem survives a level only if at least one of the dataset's own `goal_ik` solutions is still collision
free under the FCL hull-mesh oracle (benchmark/collision_oracles.py, MotionBenchMaker's own collision geometry), so every kept problem is known-feasible and
the per-solver success rate measures the solver, not the level. Problem indices are NOT preserved: survivors
are renumbered 0..n-1 and keep their source index in `source_idx`.

  python benchmark/make_clearance_ladder.py --out benchmark/problems/mb_tight.json \\
      --sets cage_panda table_pick_panda table_under_pick_panda --deltas-cm 1 2 3

Output: one problems JSON in the mb_problems.json schema with sets named `<set>_tight<D>cm`, plus a
`ladder` block recording per level how many problems survived and how many goal_ik each kept.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collision_oracles import MeshOracle  # noqa: E402
from panda_collision import mb_instance_to_world_dict  # noqa: E402

DEFAULT_SETS = ("cage_panda", "table_pick_panda", "table_under_pick_panda")


def grow_cuboids(inst: dict, delta_m: float) -> dict:
    out = copy.deepcopy(inst)
    for o in out["obstacles"].get("cuboid", {}).values():
        o["dims"] = [float(d) + 2.0 * delta_m for d in o["dims"]]
    return out


def feasible_goal_iks(inst: dict, mesh: MeshOracle, tol_m: float) -> list[list[float]]:
    world = mb_instance_to_world_dict(inst)
    return [q for q in inst["goal_ik"] if mesh.collision_free(np.asarray(q, float)[:7], world, tol_m=tol_m)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", default=str(Path(__file__).resolve().parents[1] / "tests/mb_problems.json"))
    ap.add_argument("--sets", nargs="+", default=list(DEFAULT_SETS))
    ap.add_argument("--deltas-cm", nargs="+", type=float, default=[1.0, 2.0, 3.0])
    ap.add_argument("--mesh-tol-mm", type=float, default=1.0, help="touch tolerance of the feasibility check")
    ap.add_argument("--include-level0", action="store_true",
                    help="also emit `<set>_tight0cm` = the source set filtered by the same feasibility check")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    problems = json.load(open(args.problems))["problems"]
    mesh = MeshOracle()
    tol = args.mesh_tol_mm * 1e-3
    out_sets, ladder = {}, {}
    levels = ([0.0] if args.include_level0 else []) + list(args.deltas_cm)
    for pset in args.sets:
        if pset not in problems:
            sys.exit(f"unknown set {pset!r}; have {sorted(problems)}")
        for d_cm in levels:
            name = f"{pset}_tight{d_cm:g}cm"
            kept, n_goal_ik = [], []
            for idx, inst in enumerate(problems[pset]):
                grown = grow_cuboids(inst, d_cm * 1e-2)
                ok = feasible_goal_iks(grown, mesh, tol)
                if not ok:
                    continue
                grown["goal_ik"] = ok
                grown["source_idx"] = idx
                kept.append(grown)
                n_goal_ik.append(len(ok))
            out_sets[name] = kept
            ladder[name] = {"source_set": pset, "delta_cm": d_cm, "kept": len(kept), "of": len(problems[pset]),
                            "goal_ik_per_problem_median": float(np.median(n_goal_ik)) if n_goal_ik else 0.0}
            print(f"{name}: {len(kept)}/{len(problems[pset])} problems keep a mesh-feasible goal_ik "
                  f"(median {ladder[name]['goal_ik_per_problem_median']:g} of them)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"problems": out_sets, "ladder": {"source": str(args.problems), "mesh_tol_mm": args.mesh_tol_mm,
                                                    "rule": "cuboid dims += 2*delta; cylinders and goals unchanged",
                                                    "levels": ladder}}, f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
