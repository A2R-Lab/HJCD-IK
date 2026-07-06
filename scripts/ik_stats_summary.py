#!/usr/bin/env python3
import csv
import sys
from collections import defaultdict
from pathlib import Path

CSV_PATH = Path(__file__).parent.parent / "ik_stats.csv"

if len(sys.argv) > 1:
    CSV_PATH = Path(sys.argv[1])

if not CSV_PATH.exists():
    print(f"No file found at {CSV_PATH}")
    sys.exit(1)

rows = []
with open(CSV_PATH, newline="") as f:
    rows = list(csv.DictReader(f))

if not rows:
    print("CSV is empty.")
    sys.exit(0)

expected_cols = set(rows[0].keys())
dropped = [r for r in rows if set(r.keys()) != expected_cols or None in r]
rows = [r for r in rows if set(r.keys()) == expected_cols and None not in r]
if dropped:
    print(f"[warn] skipped {len(dropped)} row(s) with mismatched columns (stale CSV entries)")
if not rows:
    print("No valid rows after filtering.")
    sys.exit(0)

# Group by b_size
groups = defaultdict(list)
for row in rows:
    groups[int(row["b_size"])].append(row)


print(f"{'b_size':>8}  {'runs':>5}  {'pct_cf':>8}  {'ik_lost_rate':>13}")
print("-" * 42)

for b_size in sorted(groups):
    g = groups[b_size]
    n = len(g)

    total_returned = sum(int(r["n_returned"])           for r in g)
    total_cf       = sum(int(r["n_returned_coll_free"]) for r in g)
    pct_cf = 100.0 * total_cf / total_returned if total_returned > 0 else 0.0

    total_ik_good = sum(int(r["n_ik_accurate"]) for r in g)
    total_ik_lost = sum(int(r["n_ik_lost"])     for r in g)
    ik_lost_rate  = 100.0 * total_ik_lost / total_ik_good if total_ik_good > 0 else 0.0

    print(f"{b_size:>8}  {n:>5}  {pct_cf:>7.1f}%  {ik_lost_rate:>12.1f}%")
