"""Diagnostics must distinguish measurements from uncomputed fields, in the API and in the CSV."""
import csv
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
hjcdik = pytest.importorskip("hjcdik")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
from _results_io import STATS_COLUMNS, append_stats_csv  # noqa: E402

UNMEASURED = (
    "n_coll_free_refined", "n_feasible", "n_ik_lost", "n_coll_in_refined",
    "n_coll_in_coarse", "env_cost_min_mm", "env_cost_max_mm", "env_cost_mean_mm",
    "n_returned_coll_free", "pct_returned_coll_free", "n_returned_feasible",
)


@pytest.fixture
def solver():
    return hjcdik.Solver()


def test_open_world_stats_report_accuracy_without_claiming_collision_checks(solver):
    target = solver.sample_targets(1, seed=31)[0]
    result = solver.solve(target, batch_size=32, collision=None, stats=True)
    row = result.stats[0]
    accurate = int(result.ok(5, 1e-3).sum())
    assert accurate > 0
    assert row["n_returned"] == int(result.count[0])
    assert row["n_returned_ik_accurate"] == accurate
    for key in UNMEASURED:
        assert row[key] == -1, key


@pytest.mark.parametrize("mode", ["hard", "soft", "both"])
def test_collision_stats_use_the_selected_checks(solver, mode):
    target = solver.sample_targets(1, seed=31)[0]
    solver.set_world(spheres=[[0, 0, 0, 100]])
    result = solver.solve(target, batch_size=32, num_solutions=4, collision=mode, stats=True)
    row = result.stats[0]
    assert row["n_coll_in_refined"] == row["n_refined"]
    assert row["n_coll_in_coarse"] == 32
    assert row["n_coll_free_refined"] == row["n_feasible"] == 0
    assert row["n_ik_lost"] == row["n_ik_accurate"]
    assert row["n_returned"] == int(result.count[0])
    assert row["n_returned_ik_accurate"] == int(result.ok(5, 1e-3).sum())
    assert row["n_returned_coll_free"] == row["n_returned_feasible"] == 0
    assert row["pct_returned_coll_free"] == 0
    for key in ("env_cost_min_mm", "env_cost_max_mm", "env_cost_mean_mm"):
        if mode == "hard":
            assert row[key] == -1
        else:
            assert row[key] > 0
    assert int(result.count[0]) > 0 if mode == "soft" else int(result.count[0]) == 0


def test_both_mode_counts_include_self_collisions_with_empty_environment(solver):
    target = solver.sample_targets(1, seed=31)[0]
    solver.clear_world()
    result = solver.solve(target, batch_size=32, num_solutions=4, collision="both", stats=True)
    row = result.stats[0]
    assert row["n_coll_in_refined"] + row["n_coll_free_refined"] == row["n_refined"]
    assert row["n_ik_lost"] + row["n_feasible"] == row["n_ik_accurate"]
    assert row["n_returned_coll_free"] == int(result.count[0])
    assert row["env_cost_min_mm"] == row["env_cost_max_mm"] == row["env_cost_mean_mm"] == 0


def test_stats_csv_keeps_the_legacy_columns_and_appends(solver, tmp_path):
    targets = solver.sample_targets(2, seed=37)
    result = solver.solve(targets, batch_size=32, collision=None, stats=True)
    path = tmp_path / "ik_stats.csv"
    assert append_stats_csv(path, result) == 2
    assert append_stats_csv(path, result) == 2
    lines = path.read_text().splitlines()
    assert lines[0] == ",".join(column for column, _ in STATS_COLUMNS)
    assert lines[0].startswith("b_size,krep,")
    assert len(lines) == 5 and all(line.startswith("32,") for line in lines[1:])
    rows = list(csv.DictReader(path.open()))
    assert [int(r["n_returned"]) for r in rows] == result.count.tolist() * 2
    with pytest.raises(ValueError, match="stats=True"):
        append_stats_csv(path, solver.solve(targets[0], batch_size=32, collision=None))


def test_stats_do_not_change_solver_outputs(solver):
    target = solver.sample_targets(1, seed=59)[0]
    # Coarse search always stops on the first successful block, even when multiple
    # outputs disable LM early-stop. Use one coarse block so scheduling cannot
    # change its candidate pool between calls; keep the strict output comparison.
    kwargs = dict(batch_size=1, num_solutions=4, precision="fp32", collision=None)
    normal = solver.solve(target, **kwargs)
    logged = solver.solve(target, stats=True, **kwargs)
    assert normal.count.tolist() == logged.count.tolist()
    for key in ("q", "pose", "pos_err_mm", "ori_err_rad"):
        np.testing.assert_allclose(getattr(normal, key), getattr(logged, key), rtol=1e-6, atol=1e-8,
                                   equal_nan=True)
