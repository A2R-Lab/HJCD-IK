#!/usr/bin/env python3
"""Success-vs-clearance curves from the clearance ladder (make_clearance_ladder.py + score_collision_oracles.py).

Reads the scorer's `--json-out` rows for sets named `<set>_tight<D>cm`, plots per source set the
pose-AND-collision-free rate (under the chosen oracle column) against D for every solver, one line style
per batch size, and prints the same numbers as a markdown table.

  python benchmark/plot_clearance_ladder.py scores.json --ladder mb_tight.json --out ladder.png --table ladder.md
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

NAME_RE = re.compile(r"^(?P<set>.+)_tight(?P<delta>[0-9.]+)cm$")
SOLVER_ORDER = ["hjcdik", "hjcdik_cusph", "curobo", "pyroki"]
SOLVER_LABEL = {"hjcdik": "HJCD-IK", "hjcdik_cusph": "HJCD-IK/cuRobo-spheres", "curobo": "cuRobo", "pyroki": "PyRoki"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scores", help="score_collision_oracles.py --json-out file")
    ap.add_argument("--ladder", default="", help="mb_tight.json (for the kept-problem counts in the table)")
    ap.add_argument("--oracle", default="visual<=1mm", help="column of pose_and_free to plot")
    ap.add_argument("--out", default="", help="PNG path (needs matplotlib)")
    ap.add_argument("--table", default="", help="markdown table path")
    args = ap.parse_args()

    rows = json.load(open(args.scores))
    kept = {}
    if args.ladder:
        kept = {k: v["kept"] for k, v in json.load(open(args.ladder)).get("ladder", {}).get("levels", {}).items()}
    # curves[set][solver][batch] = [(delta, rate%)]
    curves = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        m = NAME_RE.match(r["problem_set"])
        if not m or r["n"] == 0:
            continue
        rate = 100.0 * r["pose_and_free"][args.oracle] / r["n"]
        curves[m["set"]][r["solver"]][r["batch"]].append((float(m["delta"]), rate, r["n"]))
    for s in curves.values():
        for b in s.values():
            for lst in b.values():
                lst.sort()

    lines = [f"## Clearance ladder — pose reached AND collision-free under `{args.oracle}` (%)", ""]
    for pset in sorted(curves):
        batches = sorted({b for sv in curves[pset].values() for b in sv})
        deltas = sorted({d for sv in curves[pset].values() for b in sv.values() for d, _, _ in b})
        lines.append(f"### {pset}")
        lines.append("| clearance −D | problems | " + " | ".join(f"{SOLVER_LABEL.get(s, s)} B={b}" for s in SOLVER_ORDER
                                                             if s in curves[pset] for b in batches) + " |")
        lines.append("| --- | --- | " + " | ".join("---" for s in SOLVER_ORDER if s in curves[pset] for _ in batches) + " |")
        for d in deltas:
            n = kept.get(f"{pset}_tight{d:g}cm", next((n for sv in curves[pset].values() for b in sv.values()
                                                       for dd, _, n in b if dd == d), "?"))
            cells = []
            for s in SOLVER_ORDER:
                if s not in curves[pset]:
                    continue
                for b in batches:
                    v = [r for dd, r, _ in curves[pset][s].get(b, []) if dd == d]
                    cells.append(f"{v[0]:.0f}" if v else "-")
            lines.append(f"| {d:g} cm | {n} | " + " | ".join(cells) + " |")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if args.table:
        Path(args.table).write_text(text + "\n")

    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sets = sorted(curves)
        fig, axes = plt.subplots(1, len(sets), figsize=(4.2 * len(sets), 3.6), sharey=True, squeeze=False)
        styles = {}
        for ax, pset in zip(axes[0], sets):
            for s in SOLVER_ORDER:
                if s not in curves[pset]:
                    continue
                for b, pts in sorted(curves[pset][s].items()):
                    ls = styles.setdefault(b, ["-", "--", ":", "-."][len(styles) % 4])
                    ax.plot([d for d, _, _ in pts], [r for _, r, _ in pts], ls, marker="o", ms=3,
                            color={"hjcdik": "C0", "hjcdik_cusph": "C3", "curobo": "C1", "pyroki": "C2"}.get(s, None),
                            label=f"{SOLVER_LABEL.get(s, s)} B={b}")
            ax.set_title(pset)
            ax.set_xlabel("clearance reduction D (cm)")
            ax.grid(alpha=0.3)
            ax.set_ylim(0, 102)
        axes[0][0].set_ylabel(f"pose & free ({args.oracle}) [%]")
        axes[0][-1].legend(fontsize=7, loc="lower left")
        fig.tight_layout()
        fig.savefig(args.out, dpi=150)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
