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


@pytest.mark.parametrize("changed", [None, "commit", "header", "binary", "input"])
def test_gate_preflight_checks_provenance_without_gpu_work(tmp_path, monkeypatch, changed):
    import json
    m = module("run_audit_gate")
    binary = tmp_path / "solver.so"
    binary.write_bytes(b"precompiled")
    workload = tmp_path / "input.json"
    workload.write_text("{}")
    config = {key: str(workload) for key in ("targets_json", "problems_json")}
    config.update({key + "_sha": m.digest(workload) for key in ("targets_json", "problems_json")})
    config["endpoints"] = {"main": dict(repo="repo", python="python", commit="commit",
                                      header_sha="header", binary_sha=m.digest(binary))}
    info = dict(build=dict(grid_header_sha256="header", ee_target="panda_hand_joint"),
                binary=str(binary), python="3.12", numpy="2.5")
    if changed == "binary":
        binary.write_bytes(b"changed")
    if changed == "input":
        workload.write_text("changed")
    if changed == "header":
        info["build"]["grid_header_sha256"] = "changed"
    def check_output(cmd, **kwargs):
        if cmd[0] == "git":
            return "changed" if changed == "commit" else "commit"
        assert cmd[:2] == ["python", "-c"]
        return json.dumps(info)
    monkeypatch.setattr(m.subprocess, "check_output", check_output)
    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **kw: pytest.fail("preflight launched a worker"))
    monkeypatch.setattr(m, "foreign_gpu_pids", lambda *a: pytest.fail("preflight queried the GPU"))
    path, out = tmp_path / "gate.json", tmp_path / "no-output"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(m.sys, "argv", ["gate", "--config", str(path), "--out", str(out), "--check"])
    if changed:
        with pytest.raises(ValueError, match="changed"):
            m.main()
    else:
        m.main()
    assert not out.exists()
