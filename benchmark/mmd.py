"""Joint-config *dump* I/O for the MMD / Table IV pipeline.

Each solver (and the TRAC-IK ground truth) emits a per-target *config dump* over the SAME target set;
`benchmark/run_mmd.py` then scores them with the CANONICAL estimator in `benchmark/compute_mmd.py`
(the co-author's IMQ / multi-bandwidth / unbiased MMD²). This module is just the shared dump schema +
a CSV exporter into the layout `compute_mmd.py` expects (`solver,pose_id,q1..qN`).

NOTE: the old Gaussian/biased-V-statistic MMD that used to live here was NON-canonical and has been
removed — `compute_mmd.py` is the single source of truth for the metric. Stdlib + numpy only.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


# ---- config-dump I/O (shared by the harnesses + run_mmd.py) ----
def save_config_dump(path, solver, configs_per_target, num_dof=None):
    """Write a config dump: per-target lists of K joint vectors.

    Schema: {"solver": <name>, "num_dof": <int|null>,
             "configs": [ [ [q1..qd], ...up to K ], ...one list per target ]}.
    """
    data = {
        "solver": str(solver),
        "num_dof": int(num_dof) if num_dof is not None else None,
        "configs": [[[float(v) for v in q] for q in tgt] for tgt in configs_per_target],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def load_config_dump(path):
    return json.loads(Path(path).read_text())


def config_dump_to_csv(dump, csv_path):
    """Flatten a config dump into the CSV `compute_mmd.py` consumes: `solver,pose_id,q1..qN`.

    `dump` is a path or an already-loaded dict. One row per (target, joint vector); `pose_id` is the
    target index (the group column for `compute_mmd.py --group_col pose_id`). Returns the written path.
    """
    if not isinstance(dump, dict):
        dump = load_config_dump(dump)
    solver = dump.get("solver", "solver")
    configs = dump.get("configs", [])
    ndof = dump.get("num_dof") or max((len(q) for tgt in configs for q in tgt), default=0)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["solver", "pose_id"] + [f"q{j + 1}" for j in range(ndof)])
        for pose_id, tgt in enumerate(configs):
            for q in tgt:
                w.writerow([solver, pose_id] + [f"{float(v):.10g}" for v in q])
    return csv_path
