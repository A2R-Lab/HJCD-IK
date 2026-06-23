#!/usr/bin/env python3
"""Compute paper Table IV (MMD / MMD²) from per-solver joint-config dumps vs a ground-truth dump.

Each solver (and the ground truth) emits a *config dump* JSON over the SAME target set:
    {"solver": "<name>", "num_dof": 7, "configs": [ [ [q1..qd], ...up to K ], ...one list per target ]}
HJCD-IK   : python benchmark/hjcd_ik_bench.py ... --mmd-dump dumps/hjcdik.json
baselines : python benchmark/baseline_bench.py --mode <m> ... --mmd_dump dumps/<m>.json
groundtruth: python benchmark/gen_groundtruth_tracik.py ... --out dumps/groundtruth.json   (needs TRAC-IK)

  python benchmark/run_mmd.py --groundtruth dumps/groundtruth.json \
      --solver-dump dumps/hjcdik.json dumps/pyroki.json dumps/curobo.json --out results/table4_mmd.md

Lower MMD = closer to the ground-truth solution manifold (paper: HJCD-IK lowest). numpy only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mmd import load_config_dump, mmd_over_targets  # noqa: E402


def render(results, sigma_note):
    out = ["## Table IV — MMD / MMD² (lower = better)\n",
           "| Metric | " + " | ".join(r["solver"] for r in results) + " |",
           "| --- | " + " | ".join("---" for _ in results) + " |",
           "| MMD ↓ | "  + " | ".join(f"{r['mmd']:.5f}" for r in results) + " |",
           "| MMD² ↓ | " + " | ".join(f"{r['mmd2']:.5f}" for r in results) + " |",
           f"\n_{sigma_note}; n_targets per solver: " +
           ", ".join(f"{r['solver']}={r['n_targets']}" for r in results) + "_"]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groundtruth", required=True, help="ground-truth config dump (e.g. TRAC-IK)")
    ap.add_argument("--solver-dump", nargs="+", required=True, help="per-solver config dumps")
    ap.add_argument("--sigma", type=float, default=None,
                    help="Gaussian bandwidth; default = shared median heuristic across all solvers/targets")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gt = load_config_dump(args.groundtruth)["configs"]

    # Shared bandwidth across every solver+target so MMDs are comparable (median over the pooled cloud).
    sigma = args.sigma
    if sigma is None:
        import numpy as np
        from mmd import median_sigma
        pool = []
        for dpath in args.solver_dump:
            for tgt in load_config_dump(dpath)["configs"]:
                pool.extend(tgt)
        for tgt in gt:
            pool.extend(tgt)
        sigma = median_sigma(np.asarray(pool, float)) if pool else 1.0

    results = []
    for dpath in args.solver_dump:
        d = load_config_dump(dpath)
        r = mmd_over_targets(d["configs"], gt, sigma=sigma)
        r["solver"] = d.get("solver", Path(dpath).stem)
        results.append(r)

    md = render(results, f"Gaussian kernel, shared σ={sigma:.4g}")
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md + "\n", encoding="utf-8")
        print(f"\n[OK] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
