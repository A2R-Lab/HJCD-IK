"""Shared loader that normalizes the two benchmark CSV schemas into one long format.

Both harnesses emit per-(solver, batch) accuracy/latency, but with different column names:
  * benchmark/hjcd_ik_bench.py  --csv-out : solver, Batch-Size, time_ms,      pos_err_mm,     ori_err_rad
  * benchmark/baseline_bench.py  .csv      : solver, Batch-Size, IK-time(ms),  Pos-Error(mm),  Ori-Error  (+ per-problem rows)

`load_runs()` reads any mix of these, aggregates by (solver, batch) MEAN, and returns rows
``{solver, batch, time_ms, pos_mm, ori_rad, n}`` — consumed by make_tables.py and plot_pareto.py.
Stdlib only (csv) so the analysis tools stay dependency-light.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

_TIME = ("time_ms", "IK-time(ms)")
_POS = ("pos_err_mm", "Pos-Error(mm)")
_ORI = ("ori_err_rad", "Ori-Error")
_BATCH = ("Batch-Size", "batch")
_SOLVER = ("solver",)


def _pick(row, names):
    for n in names:
        v = row.get(n)
        if v not in (None, ""):
            return v
    return None


def load_runs(paths, *, default_solver=None):
    """Read CSVs → list of normalized rows aggregated by (solver, batch), sorted by (solver, batch)."""
    acc = defaultdict(lambda: {"t": [], "p": [], "o": []})
    for path in paths:
        path = Path(path)
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                solver = _pick(row, _SOLVER) or default_solver or path.stem
                b, t, p, o = (_pick(row, _BATCH), _pick(row, _TIME),
                              _pick(row, _POS), _pick(row, _ORI))
                if None in (b, t, p, o):
                    continue
                key = (str(solver), int(float(b)))
                acc[key]["t"].append(float(t))
                acc[key]["p"].append(float(p))
                acc[key]["o"].append(float(o))
    rows = []
    for (solver, batch), v in acc.items():
        n = len(v["t"])
        rows.append(dict(solver=solver, batch=batch, n=n,
                         time_ms=sum(v["t"]) / n, pos_mm=sum(v["p"]) / n, ori_rad=sum(v["o"]) / n))
    rows.sort(key=lambda r: (r["solver"], r["batch"]))
    return rows


def solvers(rows):
    return sorted({r["solver"] for r in rows})


def batches(rows):
    return sorted({r["batch"] for r in rows})
