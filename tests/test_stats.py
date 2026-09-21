"""Diagnostic CSV values must distinguish measurements from uncomputed fields."""
import csv
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
hjcdik = pytest.importorskip("hjcdik")


def read_row(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    return {key: float(value) for key, value in rows[0].items()}


def test_open_world_stats_report_accuracy_without_claiming_collision_checks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = hjcdik.sample_targets(1, seed=31)[0]
    result = hjcdik.generate_solutions(target, batch_size=32, write_stats=True)
    row = read_row(tmp_path / "ik_stats.csv")
    accurate = int(((result["pos_errors"] < 5) & (result["ori_errors"] < 1e-3)).sum())
    assert accurate > 0
    assert row["n_returned"] == result["count"]
    assert row["n_returned_ik_accurate"] == accurate
    for key in (
        "n_coll_free_refined", "n_feasible", "n_ik_lost", "n_coll_in_refined",
        "n_coll_in_coarse", "env_cost_min_mm", "env_cost_max_mm", "env_cost_mean_mm",
        "n_returned_coll_free", "pct_returned_coll_free", "n_returned_feasible",
    ):
        assert row[key] == -1, key


@pytest.mark.parametrize("mode", ["hard", "soft", "both"])
def test_collision_stats_use_the_selected_checks(tmp_path, monkeypatch, mode):
    monkeypatch.chdir(tmp_path)
    target = hjcdik.sample_targets(1, seed=31)[0]
    scene = json.dumps({"problems": {"blocked": [{"obstacles": {
        "sphere": [{"position": [0, 0, 0], "radius": 100}],
    }}]}})
    result = hjcdik.generate_solutions(
        target, batch_size=32, num_solutions=4, collision_free=True,
        problems_json_text=scene, problem_set_name="blocked",
        collision_mode=mode, write_stats=True,
    )
    row = read_row(tmp_path / "ik_stats.csv")
    assert row["n_coll_in_refined"] == row["krep"]
    assert row["n_coll_in_coarse"] == 32
    assert row["n_coll_free_refined"] == row["n_feasible"] == 0
    assert row["n_ik_lost"] == row["n_ik_accurate"]
    assert row["n_returned"] == result["count"]
    assert row["n_returned_ik_accurate"] == int(
        ((result["pos_errors"] < 5) & (result["ori_errors"] < 1e-3)).sum()
    )
    assert row["n_returned_coll_free"] == row["n_returned_feasible"] == 0
    assert row["pct_returned_coll_free"] == 0
    for key in ("env_cost_min_mm", "env_cost_max_mm", "env_cost_mean_mm"):
        if mode == "hard":
            assert row[key] == -1
        else:
            assert row[key] > 0
    assert result["count"] > 0 if mode == "soft" else result["count"] == 0


def test_both_mode_counts_include_self_collisions_with_empty_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = hjcdik.sample_targets(1, seed=31)[0]
    result = hjcdik.generate_solutions(
        target, batch_size=32, num_solutions=4, collision_free=True,
        problems_json_text='{"problems":{"empty":[{"obstacles":{}}]}}',
        problem_set_name="empty", collision_mode="both", write_stats=True,
    )
    row = read_row(tmp_path / "ik_stats.csv")
    assert row["n_coll_in_refined"] + row["n_coll_free_refined"] == row["krep"]
    assert row["n_ik_lost"] + row["n_feasible"] == row["n_ik_accurate"]
    assert row["n_returned_coll_free"] == result["count"]
    assert row["env_cost_min_mm"] == row["env_cost_max_mm"] == row["env_cost_mean_mm"] == 0


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="requires /dev/full")
def test_buffered_stats_write_failure_raises_and_next_solve_recovers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ik_stats.csv").symlink_to("/dev/full")
    target = hjcdik.sample_targets(1, seed=37)[0]
    with pytest.raises(RuntimeError, match="cannot write ik_stats.csv"):
        hjcdik.generate_solutions(target, batch_size=32, write_stats=True)
    assert hjcdik.generate_solutions(target, batch_size=32)["count"] == 1


def test_stats_do_not_change_solver_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = hjcdik.sample_targets(1, seed=59)[0]
    kwargs = dict(batch_size=32, num_solutions=4, refine_fp64=0)
    normal = hjcdik.generate_solutions(target, **kwargs)
    logged = hjcdik.generate_solutions(target, write_stats=True, **kwargs)
    assert normal["count"] == logged["count"]
    for key in ("joint_config", "pose", "pos_errors", "ori_errors"):
        np.testing.assert_allclose(normal[key], logged[key], rtol=1e-6, atol=1e-8)
