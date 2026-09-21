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
dropped = [r for r in rows if set(r.keys()) != expected_cols or None in r or None in r.values()]
rows = [r for r in rows if set(r.keys()) == expected_cols and None not in r and None not in r.values()]
if dropped:
    print(f"[warn] skipped {len(dropped)} row(s) with mismatched columns (stale CSV entries)")
if not rows:
    print("No valid rows after filtering.")
    sys.exit(0)

# Group by b_size
groups = defaultdict(list)
for row in rows:
    groups[int(row["b_size"])].append(row)


def percent(numerator, denominator):
    return f"{100.0 * numerator / denominator:.1f}%" if denominator > 0 else "n/a"


print("Collision metrics follow the producer's mode; soft-only is environment-only.")
print("Do not mix hard/both and soft-only runs in one input file.")
print(f"{'b_size':>8}  {'runs':>5}  {'cc_runs':>7}  {'pct_cf':>8}  {'ik_lost_rate':>13}")
print("-" * 51)

for b_size in sorted(groups):
    g = groups[b_size]
    n = len(g)

    # -1 means the check was not performed. Exclude its denominator too: open-world
    # returns must not dilute percentages from the collision-checked subset.
    checked = [r for r in g if int(r["n_returned_coll_free"]) >= 0]
    total_returned = sum(int(r["n_returned"]) for r in checked)
    total_cf = sum(int(r["n_returned_coll_free"]) for r in checked)
    pct_cf = percent(total_cf, total_returned)

    measured_loss = [r for r in g if int(r["n_ik_lost"]) >= 0]
    total_ik_good = sum(int(r["n_ik_accurate"]) for r in measured_loss)
    total_ik_lost = sum(int(r["n_ik_lost"]) for r in measured_loss)
    ik_lost_rate = percent(total_ik_lost, total_ik_good)

    print(f"{b_size:>8}  {n:>5}  {len(checked):>7}  {pct_cf:>8}  {ik_lost_rate:>13}")
