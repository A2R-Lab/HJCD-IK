"""Host-only regressions for the diagnostic CSV summary consumer."""
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/ik_stats_summary.py"
HEADER = "b_size,n_returned,n_returned_coll_free,n_ik_accurate,n_ik_lost\n"
MEASURED = "32,2,1,4,1\n"
UNMEASURED = "32,100,-1,100,-1\n"


@pytest.mark.parametrize("rows,expected", [
    (UNMEASURED, ["32", "1", "0", "n/a", "n/a"]),
    (MEASURED, ["32", "1", "1", "50.0%", "25.0%"]),
    (MEASURED + UNMEASURED, ["32", "2", "1", "50.0%", "25.0%"]),
    ("32,0,0,0,0\n", ["32", "1", "1", "n/a", "n/a"]),
])
def test_summary_uses_only_measured_denominators(tmp_path, rows, expected):
    path = tmp_path / "stats.csv"
    path.write_text(HEADER + rows)
    run = subprocess.run([sys.executable, str(SCRIPT), str(path)],
                         capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    assert run.stdout.splitlines()[-1].split() == expected
    assert "soft-only is environment-only" in run.stdout


def test_summary_skips_truncated_rows(tmp_path):
    path = tmp_path / "stats.csv"
    path.write_text(HEADER + MEASURED + "32,2\n")
    run = subprocess.run([sys.executable, str(SCRIPT), str(path)],
                         capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    assert "skipped 1 row(s)" in run.stdout
    assert run.stdout.splitlines()[-1].split() == ["32", "1", "1", "50.0%", "25.0%"]
