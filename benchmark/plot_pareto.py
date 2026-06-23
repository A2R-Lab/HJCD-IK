#!/usr/bin/env python3
"""Accuracy-latency Pareto plots from per-solver benchmark CSVs (paper Figs 4 / 5).

Two panels, matching the paper:
  (left)  Position error (mm) vs Orientation error (rad)         — log-log
  (right) Combined error (mm + rad) vs Solve time (ms)           — log-log
One marker per (solver, batch); markers along a solver are joined in batch order so the accuracy-latency
trajectory is visible. HJCD-IK should sit lower-left (more accurate) and/or further left (faster).

  python benchmark/plot_pareto.py benchmark/results/open_*.csv --out benchmark/results/pareto_open.png \
      --title "Panda open-world"

Needs matplotlib (`pip install -e ".[plots]"`). Reads both CSV schemas via benchmark/_results_io.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _results_io import load_runs, solvers, batches  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default="pareto.png")
    ap.add_argument("--title", default="")
    ap.add_argument("--annotate-batch", action="store_true", help="label each point with its batch size")
    args = ap.parse_args()

    rows = load_runs(args.csv)
    if not rows:
        sys.exit("no usable rows found in the given CSVs")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by = {(r["solver"], r["batch"]): r for r in rows}
    svs = solvers(rows)
    cmap = plt.get_cmap("tab10")
    colors = {s: cmap(i % 10) for i, s in enumerate(svs)}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))

    for s in svs:
        bts = [b for b in batches(rows) if (s, b) in by]
        pos = [by[(s, b)]["pos_mm"] for b in bts]
        ori = [max(by[(s, b)]["ori_rad"], 1e-12) for b in bts]
        comb = [by[(s, b)]["pos_mm"] + by[(s, b)]["ori_rad"] for b in bts]
        tms = [by[(s, b)]["time_ms"] for b in bts]
        c = colors[s]
        axL.plot(ori, pos, "-o", color=c, label=s, markersize=6)
        axR.plot(tms, comb, "-o", color=c, label=s, markersize=6)
        if args.annotate_batch:
            for b, x, y in zip(bts, ori, pos):
                axL.annotate(str(b), (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
            for b, x, y in zip(bts, tms, comb):
                axR.annotate(str(b), (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")

    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xlabel("Orientation Error (rad)"); axL.set_ylabel("Position Error (mm)")
    axL.set_title("Accuracy"); axL.grid(True, which="both", alpha=0.3); axL.legend()

    axR.set_xscale("log"); axR.set_yscale("log")
    axR.set_xlabel("Solve Time (ms)"); axR.set_ylabel("Combined Error (mm + rad)")
    axR.set_title("Accuracy vs Latency"); axR.grid(True, which="both", alpha=0.3); axR.legend()

    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
