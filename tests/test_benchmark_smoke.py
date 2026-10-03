"""End-to-end smoke test of the benchmark harness (`benchmark/hjcd_ik_bench.py`).

`test_benchmark_setup.py` checks the harness's codegen plumbing without running a solver; this file runs
the harness itself, minimally, in both modes it offers. It exists because the open-world path was broken on
`main` for a month (it passed `problem_idx=-1`, which the API rejects) and nothing noticed: the harness is
how the paper tables are produced, so it is tested like any other entry point. Tiny sizes, no timing claims.

Requires a CUDA GPU + a built, collision-enabled `hjcdik` extension; skips cleanly otherwise.
"""
import csv
import subprocess
import sys
from pathlib import Path

import pytest

hjcdik = pytest.importorskip("hjcdik")

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark" / "hjcd_ik_bench.py"
COLUMNS = ["solver", "Batch-Size", "time_ms", "pos_err_mm", "ori_err_rad", "collision_free(%)"]


def _run(tmp_path, *extra):
    csv_out = tmp_path / "summary.csv"
    cmd = [sys.executable, str(BENCH), "--skip-grid-codegen", "--batches", "1,64",
           "--yaml-out", str(tmp_path / "results.yml"), "--csv-out", str(csv_out), *extra]
    # Plain `python benchmark/hjcd_ik_bench.py`, as the paper script runs it: the harness imports its
    # sibling modules (panda_collision, panda_model, mmd) through the script directory on sys.path.
    run = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, f"benchmark exited {run.returncode}\n--- stdout\n{run.stdout}\n--- stderr\n{run.stderr}"
    rows = list(csv.DictReader(csv_out.open()))
    assert [c for c in rows[0]] == COLUMNS
    assert sorted(int(r["Batch-Size"]) for r in rows) == [1, 64]
    for r in rows:
        assert float(r["time_ms"]) > 0
        assert float(r["pos_err_mm"]) >= 0 and float(r["ori_err_rad"]) >= 0
    # The harness writes its outputs where it is told, plus a provenance sidecar beside each one;
    # nothing may land outside the working directory.
    assert (tmp_path / "results.yml").exists()
    return rows, run


def test_open_world_harness_runs(tmp_path):
    rows, run = _run(tmp_path, "--num-targets", "2")
    assert all(r["collision_free(%)"] in ("", "-1", "-1.0") for r in rows), rows
    assert "[build info]" in run.stdout


@pytest.mark.skipif(not hjcdik.collision_enabled(), reason="needs a collision-enabled grid.cuh build")
def test_collision_free_harness_runs(tmp_path):
    rows, _ = _run(tmp_path, "--collision-free", "--problem-set", "box_panda", "--problem-idx", "0",
                   "--collision-mode", "hard", "--collision-validation-model", "hjcd")
    for r in rows:
        assert 0.0 <= float(r["collision_free(%)"]) <= 100.0
    assert (tmp_path / "summary.csv.metadata.json").exists()
