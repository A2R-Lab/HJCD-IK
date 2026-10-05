#!/usr/bin/env python3
"""Re-score returned configurations under several collision oracles.

Input: JSON-lines dumps written by `hjcd_ik_bench.py --configs-out` / `baseline_bench.py --configs_out`
(one record per problem: solver, problem_set, problem_idx, batch, q, pos_err_mm, ori_err_rad). Output: per
(problem_set, solver, batch) the pose-success rate and the "pose AND collision-free" rate under each oracle
in benchmark/collision_oracles.py, plus the rate under which every oracle agrees. This separates "the solver
reached the pose" from "whose collision model says it is clear", which a single oracle cannot.

  python benchmark/score_collision_oracles.py dumps/*.jsonl --problems tests/mb_problems.json --out table.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collision_oracles import MeshOracle, make_oracles  # noqa: E402
from panda_collision import mb_instance_to_world_dict  # noqa: E402
from query_results import validate_groups, pose_errors

POS_OK_MM, ORI_OK_RAD = 5.0, 0.05     # the baseline harness's pose-success thresholds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="+", help="JSON-lines configuration dumps")
    ap.add_argument("--problems", required=True, help="mb_problems.json")
    ap.add_argument("--problem-sets", nargs="+", help="Explicit subset; otherwise every set in the problem manifest is required")
    ap.add_argument("--mesh-tols-mm", default="1,5", help="mesh judges' touch tolerances to report (mm; obstacles shrunk)")
    ap.add_argument("--mesh-geometries", default="hull,visual",
                    help="which Panda meshes judge: hull = franka collision hulls (MoveIt's), visual = visual-link mesh approximation")
    ap.add_argument("--out", default="", help="write the markdown table here as well as stdout")
    ap.add_argument("--json-out", default="", help="also write the rows as JSON (for plot_clearance_ladder.py etc.)")
    ap.add_argument("--allow-legacy-reports", action="store_true",
                    help="Historical dumps only: trust reported errors when target/frame metadata is absent")
    ap.add_argument("--allow-missing-oracles", action="store_true",
                    help="Exploratory scoring only: permit explicitly reported missing optional judges")
    args = ap.parse_args()

    problems = json.load(open(args.problems))["problems"]
    if args.problem_sets:
        problems = {name: problems[name] for name in args.problem_sets}
    worlds = {}
    oracles = make_oracles(("hjcd", "paper", "curobo"))
    if len(oracles) != 3 and not args.allow_missing_oracles:
        raise RuntimeError("required sphere oracle unavailable; install baselines or explicitly allow missing oracles")
    meshes = {}
    for g in [g for g in args.mesh_geometries.split(",") if g]:
        try:
            meshes[g] = MeshOracle(geometry=g)
        except ImportError as e:
            if not args.allow_missing_oracles:
                raise
            print(f"[score] {g} mesh oracle unavailable ({e})")
    mesh_tols = [float(t) for t in args.mesh_tols_mm.split(",")] if meshes else []
    columns = list(oracles) + [f"{g}<={t:g}mm" for g in meshes for t in mesh_tols]

    records = [json.loads(line) for f in args.dumps for line in open(f) if line.strip()]
    groups = validate_groups(records, problems)

    rows = []
    for (pset, solver, batch), recs in sorted(groups.items()):
        n = len(recs)
        pose_ok = 0
        free = {c: 0 for c in columns}
        unanimous = 0
        for r in recs:
            key = (pset, int(r["problem_idx"]))
            if key not in worlds:
                worlds[key] = mb_instance_to_world_dict(problems[pset][key[1]])
            w = worlds[key]
            if r.get("q") is None:
                continue  # explicit empty output remains in the denominator
            q = np.asarray(r["q"], float)
            if "target" in r and "ee_target" in r:
                pe, oe = pose_errors(q, r["target"], r["ee_target"])
            elif args.allow_legacy_reports:
                pe, oe = r["pos_err_mm"], r["ori_err_rad"]
            else:
                raise ValueError("dump lacks target/frame metadata; recollect or explicitly allow legacy reports")
            ok = pe < POS_OK_MM and oe < ORI_OK_RAD
            pose_ok += ok
            if not ok:
                continue
            verdicts = {c: bool(f(q, w)) for c, f in oracles.items()}
            for g, mo in meshes.items():
                for t in mesh_tols:
                    verdicts[f"{g}<={t:g}mm"] = mo.collision_free(q, w, tol_m=t * 1e-3)
            for c, v in verdicts.items():
                free[c] += v
            unanimous += all(verdicts.values())
        rows.append((pset, solver, batch, n, pose_ok, free, unanimous))

    head = f"| set | solver | B | n | pose ok | " + " | ".join(f"pose & {c}" for c in columns) + " | all oracles |"
    sep = "| --- | --- | --- | --- | --- | " + " | ".join("---" for _ in columns) + " | --- |"
    lines = ["## Pose-success AND collision-free (%), per oracle", "", head, sep]
    for pset, solver, batch, n, pose_ok, free, unanimous in rows:
        pct = lambda k: f"{100.0 * k / n:.0f}" if n else "-"
        lines.append(f"| {pset} | {solver} | {batch} | {n} | {pct(pose_ok)} | "
                     + " | ".join(pct(free[c]) for c in columns) + f" | {pct(unanimous)} |")
    lines.append("")
    lines.append(f"_pose ok = pos < {POS_OK_MM:g} mm and ori < {ORI_OK_RAD:g} rad (independent URDF FK for new dumps; "
                 "legacy solver reports only when explicitly permitted); oracles ignore the "
                 "base link and check environment obstacles only; sphere columns permit touching, mesh columns shrink every "
                 "obstacle by the stated tolerance (hull = franka collision hulls, MoveIt's geometry; visual = visual-link mesh approximation); "
                 "`all oracles` = every column agrees free._")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps([
            {"problem_set": pset, "solver": solver, "batch": batch, "n": n, "pose_ok": pose_ok,
             "pose_and_free": free, "all_oracles": unanimous}
            for pset, solver, batch, n, pose_ok, free, unanimous in rows], indent=1) + "\n")


if __name__ == "__main__":
    main()
