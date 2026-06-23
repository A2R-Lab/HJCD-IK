#!/usr/bin/env python3
"""Compute paper Table IV (MMD / MMD²) from per-solver joint-config dumps vs a ground-truth dump.

Uses the CANONICAL estimator in `benchmark/compute_mmd.py` (the co-author's IMQ / multi-bandwidth /
UNBIASED MMD², joint-space, per-pose then averaged — "IKFlow-style"). Each solver (and the ground truth)
emits a *config dump* JSON over the SAME target set (schema in benchmark/mmd.py):
    {"solver": "<name>", "num_dof": 7, "configs": [ [ [q1..qd], ...up to K ], ...one list per target ]}
HJCD-IK    : python benchmark/hjcd_ik_bench.py ... --mmd-dump dumps/hjcdik.json
baselines  : python benchmark/baseline_bench.py --mode <m> ... --mmd_dump dumps/<m>.json
IKFlow     : python benchmark/baseline_ikflow.py ... --mmd-dump dumps/ikflow.json
groundtruth: python benchmark/gen_groundtruth_tracik.py ... --out dumps/groundtruth.json   (needs TRAC-IK)

  python benchmark/run_mmd.py --groundtruth dumps/groundtruth.json \
      --solver-dump dumps/hjcdik.json dumps/pyroki.json dumps/ikflow.json --out results/table4_mmd.md

For each solver this also writes a flat `<dump>.csv` (`solver,pose_id,q1..qN`) next to the dump, so you can
reproduce a single column directly with the co-author's tool:
  python benchmark/compute_mmd.py --ref dumps/groundtruth.csv --cmp dumps/hjcdik.csv --group_col pose_id

Lower MMD = closer to the ground-truth solution manifold (paper: HJCD-IK lowest). numpy only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mmd import config_dump_to_csv, load_config_dump  # noqa: E402
from compute_mmd import compute_mmd_ikflow  # noqa: E402


def _mmd_vs_gt(solver_configs, gt_configs, beta, scales):
    """Per-pose canonical MMD (IMQ, multi-bandwidth, unbiased) then average over poses.

    Mirrors compute_mmd.run's group path: mean of MMD² over groups, MMD = sqrt(mean MMD²).
    """
    mmd2s = []
    for s, g in zip(solver_configs, gt_configs):
        X = np.asarray(g, float)   # reference (ground truth)
        Y = np.asarray(s, float)   # comparison (solver)
        if X.ndim != 2 or Y.ndim != 2 or len(X) < 2 or len(Y) < 2:
            continue
        d = min(X.shape[1], Y.shape[1])     # align dof (same robot, defensive)
        m2, _ = compute_mmd_ikflow(X[:, :d], Y[:, :d], beta=beta, scales=scales)
        if np.isfinite(m2):
            mmd2s.append(m2)
    if not mmd2s:
        return dict(mmd2=float("nan"), mmd=float("nan"), n_targets=0)
    mean2 = float(np.mean(mmd2s))
    return dict(mmd2=mean2, mmd=float(np.sqrt(max(mean2, 0.0))), n_targets=len(mmd2s))


def render(results, note):
    out = ["## Table IV — MMD / MMD² (lower = better)\n",
           "| Metric | " + " | ".join(r["solver"] for r in results) + " |",
           "| --- | " + " | ".join("---" for _ in results) + " |",
           "| MMD ↓ | " + " | ".join(f"{r['mmd']:.5f}" for r in results) + " |",
           "| MMD² ↓ | " + " | ".join(f"{r['mmd2']:.5f}" for r in results) + " |",
           f"\n_{note}; n_targets per solver: " +
           ", ".join(f"{r['solver']}={r['n_targets']}" for r in results) + "_"]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groundtruth", required=True, help="ground-truth config dump (e.g. TRAC-IK)")
    ap.add_argument("--solver-dump", nargs="+", required=True, help="per-solver config dumps")
    ap.add_argument("--beta", type=float, default=0.5, help="IMQ exponent (compute_mmd default)")
    ap.add_argument("--scales", default="0.2,0.5,1,2,5",
                    help="comma-separated c = median_dist*scale (compute_mmd default)")
    ap.add_argument("--no-csv", action="store_true", help="skip writing the per-dump <name>.csv files")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    scales = [float(s) for s in args.scales.split(",") if s.strip()]

    gt_dump = load_config_dump(args.groundtruth)
    gt = gt_dump["configs"]
    if not args.no_csv:
        config_dump_to_csv(gt_dump, Path(args.groundtruth).with_suffix(".csv"))

    results = []
    for dpath in args.solver_dump:
        d = load_config_dump(dpath)
        if not args.no_csv:
            config_dump_to_csv(d, Path(dpath).with_suffix(".csv"))
        r = _mmd_vs_gt(d["configs"], gt, args.beta, scales)
        r["solver"] = d.get("solver", Path(dpath).stem)
        results.append(r)

    md = render(results, f"canonical IMQ MMD (beta={args.beta}, scales={args.scales}), per-pose then averaged")
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md + "\n", encoding="utf-8")
        print(f"\n[OK] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
