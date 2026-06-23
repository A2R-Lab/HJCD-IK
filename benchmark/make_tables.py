#!/usr/bin/env python3
"""Merge per-solver benchmark CSVs into a paper-style markdown table (Tables I–III).

Reads any mix of HJCD-IK and baseline CSVs (see benchmark/_results_io.py), groups by batch size, and
prints one block per batch with Time(ms) / Pos(mm) / Ori(rad) per solver — the layout of paper Tables
I/II/III. Stdlib only.

  python benchmark/make_tables.py benchmark/results/*.csv --title "Panda open-world (Table I)"
  python benchmark/make_tables.py --out results_table.md benchmark/results/open_*.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _results_io import load_runs, solvers, batches  # noqa: E402


def _fmt(x, sci_below=1e-3):
    if x == 0:
        return "0"
    return f"{x:.4g}" if abs(x) >= sci_below else f"{x:.3e}"


def render(rows, title=None):
    svs = solvers(rows)
    bts = batches(rows)
    by = {(r["solver"], r["batch"]): r for r in rows}

    out = []
    if title:
        out.append(f"## {title}\n")
    # header: Batch | <solver> Time/Pos/Ori | ...
    head = ["Batch"]
    sub = ["---"]
    for s in svs:
        head += [f"{s} Time(ms)", f"{s} Pos(mm)", f"{s} Ori(rad)"]
        sub += ["---", "---", "---"]
    out.append("| " + " | ".join(head) + " |")
    out.append("| " + " | ".join(sub) + " |")
    for b in bts:
        cells = [str(b)]
        for s in svs:
            r = by.get((s, b))
            if r is None:
                cells += ["—", "—", "—"]
            else:
                cells += [_fmt(r["time_ms"]), _fmt(r["pos_mm"]), _fmt(r["ori_rad"])]
        out.append("| " + " | ".join(cells) + " |")
    n_note = ", ".join(f"{s}: n={max((by[(s,b)]['n'] for b in bts if (s,b) in by), default=0)}/batch"
                       for s in svs)
    out.append(f"\n_solvers: {n_note}_")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="per-solver CSV files (HJCD and/or baseline schema)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default=None, help="write markdown here (also printed)")
    args = ap.parse_args()

    rows = load_runs(args.csv)
    if not rows:
        sys.exit("no usable rows found in the given CSVs")
    md = render(rows, args.title)
    print(md)
    if args.out:
        Path(args.out).write_text(md + "\n", encoding="utf-8")
        print(f"\n[OK] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
