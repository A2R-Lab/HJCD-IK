"""Benchmark-side serialisation of solver diagnostics (the solver itself never opens files)."""
from __future__ import annotations

import csv
from pathlib import Path

# Legacy ik_stats.csv header, kept byte-identical so scripts/ik_stats_summary.py and stored
# campaign files stay readable. Column -> hjcdik.Stats field.
STATS_COLUMNS = [
    ("b_size", "batch_size"), ("krep", "n_refined"),
    ("n_ik_accurate", "n_ik_accurate"), ("n_coll_free_refined", "n_coll_free_refined"),
    ("n_feasible", "n_feasible"), ("n_ik_lost", "n_ik_lost"),
    ("n_coll_in_refined", "n_coll_in_refined"), ("n_coll_in_coarse", "n_coll_in_coarse"),
    ("env_cost_min_mm", "env_cost_min_mm"), ("env_cost_max_mm", "env_cost_max_mm"),
    ("env_cost_mean_mm", "env_cost_mean_mm"),
    ("n_returned", "n_returned"), ("n_returned_ik_accurate", "n_returned_ik_accurate"),
    ("n_returned_coll_free", "n_returned_coll_free"), ("pct_returned_coll_free", "pct_returned_coll_free"),
    ("n_returned_feasible", "n_returned_feasible"),
]


def append_stats_csv(path, result) -> int:
    """Append one row per target of ``result.stats`` to ``path``; write the header on a new file."""
    if result.stats is None:
        raise ValueError("solve(..., stats=True) is required to write statistics")
    path = Path(path)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as stream:
        writer = csv.writer(stream)
        if new:
            writer.writerow([column for column, _ in STATS_COLUMNS])
        for t in range(len(result.stats)):
            row = result.stats[t]
            writer.writerow([row[field] for _, field in STATS_COLUMNS])
    return len(result.stats)
