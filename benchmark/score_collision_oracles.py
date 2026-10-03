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

POS_OK_MM, ORI_OK_RAD = 5.0, 0.05     # the baseline harness's pose-success thresholds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="+", help="JSON-lines configuration dumps")
    ap.add_argument("--problems", required=True, help="mb_problems.json")
    ap.add_argument("--mesh-tols-mm", default="1,5", help="mesh oracle touch tolerances to report (mm)")
    ap.add_argument("--out", default="", help="write the markdown table here as well as stdout")
    args = ap.parse_args()

    problems = json.load(open(args.problems))["problems"]
    worlds = {}
    oracles = make_oracles(("hjcd", "paper"))
    mesh = None
    try:
        mesh = MeshOracle()
    except ImportError as e:
        print(f"[score] mesh oracle unavailable ({e})")
    mesh_tols = [float(t) for t in args.mesh_tols_mm.split(",")] if mesh else []
    columns = list(oracles) + [f"mesh<={t:g}mm" for t in mesh_tols]

    records = [json.loads(line) for f in args.dumps for line in open(f) if line.strip()]
    groups = defaultdict(list)
    for r in records:
        groups[(r["problem_set"], r["solver"], int(r["batch"]))].append(r)

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
            q = np.asarray(r["q"], float)
            ok = r["pos_err_mm"] < POS_OK_MM and r["ori_err_rad"] < ORI_OK_RAD
            pose_ok += ok
            if not ok:
                continue
            verdicts = {c: bool(f(q, w)) for c, f in oracles.items()}
            if mesh:
                depth = mesh.max_penetration(q, w)
                for t in mesh_tols:
                    verdicts[f"mesh<={t:g}mm"] = depth <= t * 1e-3
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
    lines.append(f"_pose ok = pos < {POS_OK_MM:g} mm and ori < {ORI_OK_RAD:g} rad on the solver's own report; oracles ignore the "
                 "base link, check environment obstacles only, touching permitted; `all oracles` = every column agrees free._")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
