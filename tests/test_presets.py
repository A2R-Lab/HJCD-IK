"""Shipped robot presets: a URDF, an end-effector frame and a validated sphere model per name.

`Solver("<preset>")` builds the preset through GRiD's artifact store on first use (about twenty
seconds per robot on an RTX 5090) and reuses it afterwards.
"""
import json
import subprocess
import sys

import numpy as np
import pytest

hjcdik = pytest.importorskip("hjcdik")
pytest.importorskip("grid_rbd")


def test_presets_are_complete_and_their_files_ship():
    presets = hjcdik.presets()
    assert set(presets) >= {"panda", "iiwa14"}
    for name, preset in presets.items():
        assert preset["urdf"].is_file(), name
        assert preset["spheres"] in hjcdik.sphere_presets(), name
        assert hjcdik.sphere_presets()[preset["spheres"]].is_file(), name
        assert preset["ee_link"] and preset["source"]
    out = subprocess.run([sys.executable, "-m", "hjcdik", "presets"], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and {json.loads(line)["name"] for line in out.stdout.splitlines()} >= {"panda", "iiwa14"}


def test_panda_preset_is_the_packaged_robot():
    solver = hjcdik.Solver("panda")
    assert solver.info.artifact is None and solver.info.robot == "panda"


def test_iiwa14_preset_builds_and_solves_with_collision():
    solver = hjcdik.Solver("iiwa14")
    assert solver.info.robot == "iiwa14" and solver.info.dof == 7 and solver.info.ee_frame == "iiwa_joint_ee"
    assert solver.info.collision_enabled and solver.info.artifact is not None
    targets = solver.sample_targets(6, seed=3)
    open_world = solver.solve(targets, batch_size=512, num_solutions=2, collision=None)
    assert (open_world.count >= 1).all() and (open_world.pos_err_mm[:, 0] < 0.01).all()
    hard = solver.solve(targets, batch_size=512, num_solutions=2, stats=True)      # default hard, self only
    assert (hard.count >= 1).sum() >= 4
    solver.set_world(spheres=[[0, 0, 0, 100]])
    assert solver.solve(targets[0], batch_size=128).count.tolist() == [0]
    solver.clear_world()
    assert hjcdik.Solver("iiwa14", allow_build=False).info.robot == "iiwa14"     # prepared now, loads without a build
