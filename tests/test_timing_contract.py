"""CPU contracts for the timing gate; no performance measurements."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/perf" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_driver_accepts_neutral_generator_target_schema(tmp_path):
    import json
    m = module("timing_driver")
    target = [0, 0, 0, 1, 0, 0, 0]
    path = tmp_path / "targets.json"
    for record in (target, {"problem_idx": 0, "target": target}):
        path.write_text(json.dumps({"targets": [record]}))
        assert m.load_shared_targets(path) == [target]


def test_driver_names_precision_and_never_times_correctness(monkeypatch):
    m = module("timing_driver")
    calls = []
    def solve(target, **kwargs):
        calls.append(kwargs)
        return dict(count=1, joint_config=np.zeros((1, 7)), pos_errors=[0], ori_errors=[0])
    monkeypatch.setattr(m.hjcdik, "generate_solutions", solve)
    monkeypatch.setattr(m, "pose_errors", lambda *args: (0, 0))
    monkeypatch.setattr(m.time, "perf_counter", lambda: pytest.fail("correctness read a clock"))
    assert m.time_leg([[0]*7], 8, 1, False, "", "", 1, correctness_only=True) == ([], 1, 0, 1)
    assert all(c["refine_fp64"] == -1 and c["write_stats"] is False for c in calls)


def test_gate_rejects_missing_duplicate_and_mixed_protocol_cells():
    m = module("run_audit_gate")
    config = dict(rounds=1, solutions=[1], batches={"table1": [8]}, target_counts={"table1": 1},
                  endpoints={ep: {"header_sha": "h"} for ep in ("main", "branch")})
    rows = [dict(label=f"{ep}-S1", leg="table1", batch="8", round="1", n_targets="1",
                 header_sha="h", correctness_only="True", refine_fp64="-1", solved=str(n), empty="0")
            for ep, n in (("main", 1), ("branch", 0))]
    assert m.analyze(config, rows, True)[0]["quality_review_required"]
    for bad in (rows[:1], rows + rows[:1], [*rows[:1], {**rows[1], "refine_fp64": "0"}]):
        with pytest.raises(ValueError):
            m.analyze(config, bad, True)
