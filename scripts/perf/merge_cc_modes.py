#!/usr/bin/env python3
"""Merge the per-collision-mode collision-free CSVs (collfree_<mode>.csv, one per HJCD_CC_MODE) that
run_all_timing_sweeps.sh's collfree leg emits into a single side-by-side comparison — so the Table II
collision-free table gains a column per mode (soft = penetration cost, hard = grid_collision::config_free
filter, both = soft+hard).

Each input CSV is hjcd_ik_bench --csv-out format:
  solver,Batch-Size,time_ms,pos_err_mm,ori_err_rad,collision_free(%)

Writes <out_dir>/collfree_compare.csv and prints a markdown table. Pure stdlib.

  python scripts/perf/merge_cc_modes.py <out_dir> soft,hard
"""
import csv
import os
import sys


def main():
    if len(sys.argv) < 3:
        print("usage: merge_cc_modes.py <out_dir> <mode1,mode2,...>", file=sys.stderr)
        sys.exit(2)
    out_dir, modes = sys.argv[1], [m for m in sys.argv[2].split(",") if m]

    # rows[B][mode] = {time_ms, pos_err_mm, ori_err_rad, collision_free(%)}
    rows, present = {}, []
    for mode in modes:
        path = os.path.join(out_dir, f"collfree_{mode}.csv")
        if not os.path.exists(path):
            print(f"[merge] skip {mode}: {path} missing", file=sys.stderr)
            continue
        present.append(mode)
        for r in csv.DictReader(open(path)):
            rows.setdefault(int(r["Batch-Size"]), {})[mode] = r
    if not present:
        print("[merge] no mode CSVs found", file=sys.stderr)
        sys.exit(1)

    batches = sorted(rows)
    # --- comparison CSV ---
    fields = ["Batch-Size"]
    for m in present:
        fields += [f"{m}_time_ms", f"{m}_cf%"]
    fields += ["pos_err_mm", "ori_err_rad"]  # accuracy is mode-agnostic; take the first present mode
    comp_path = os.path.join(out_dir, "collfree_compare.csv")
    with open(comp_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for B in batches:
            row = [B]
            for m in present:
                r = rows[B].get(m, {})
                row += [r.get("time_ms", ""), r.get("collision_free(%)", "")]
            base = next((rows[B][m] for m in present if m in rows[B]), {})
            row += [base.get("pos_err_mm", ""), base.get("ori_err_rad", "")]
            w.writerow(row)
    print(f"[merge] wrote {comp_path}")

    # --- markdown table ---
    def cell(v):
        try:
            return f"{float(v):.4g}"
        except (TypeError, ValueError):
            return str(v)

    hdr = "| B | " + " | ".join(f"{m} time(ms) | {m} CF% " for m in present) + "| pos(mm) | ori(rad) |"
    sep = "|" + "---|" * (1 + 2 * len(present) + 2)
    print("\n" + hdr)
    print(sep)
    for B in batches:
        cells = [str(B)]
        for m in present:
            r = rows[B].get(m, {})
            cells += [cell(r.get("time_ms")), cell(r.get("collision_free(%)"))]
        base = next((rows[B][m] for m in present if m in rows[B]), {})
        cells += [cell(base.get("pos_err_mm")), cell(base.get("ori_err_rad"))]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
