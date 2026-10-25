"""Offline UR5e preset checks; no changing upstream descriptions are downloaded by tests."""
import json
import runpy
from pathlib import Path
import numpy as np
import pytest
import hjcdik
from test_general_joints import build_solver, diagnostic, targets_for, assert_poses, independent_quality

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ur5e():
    preset=hjcdik.presets()["ur5e"]
    solver,module=build_solver(preset["urdf"],"test-ur5e-diagnostic",preset["spheres"],preset["ee_link"])
    return solver,module,preset


def test_ur5e_provenance_and_geometry():
    source=json.loads((ROOT/"csrc/collision/ur5e_provenance.json").read_text())
    assert source["commit"] == "22f055da2fa7e2158254426107d1f257fd56aebb"
    assert source["license"] == "BSD-3-Clause"
    assert len(source["meshes"]) == 14
    for rows in source["fidelity"].values():
        assert len(rows)==7 and all("unresolved" not in row and row["spheres"]>0 for row in rows)
    assert (ROOT/"csrc/licenses/ur5e.txt").is_file()


@pytest.mark.parametrize("precision",["fp32","fp64"])
def test_ur5e_independent_fk_and_collision_ik(ur5e,precision):
    solver,module,preset=ur5e
    limits=np.array(solver.info.joint_limits)
    q=np.random.default_rng(43).uniform(limits[:,0],limits[:,1],(200,6))
    poses,_,free=diagnostic(solver,module,q,precision=="fp64")
    targets=targets_for(preset["urdf"],q,preset["ee_link"])
    assert_poses(poses,targets,3e-6 if precision=="fp32" else 1e-9)
    assert free.sum()>=100  # not an always-colliding default model
    selected=targets[free][:32]
    out=solver.solve(selected,batch_size=2000,num_solutions=4,precision=precision)
    assert independent_quality(solver,preset["urdf"],selected,out,preset["ee_link"])[0].sum()>=31
    for result in out:
        if result.count:
            assert diagnostic(solver,module,result.q)[2].all()
    solver.set_world(spheres=[[0,0,0,100]])
    assert solver.solve(selected[0],batch_size=37,precision=precision).count[0]==0
    solver.clear_world()


def test_shipped_ur5e_preset_loads():
    solver=hjcdik.Solver("ur5e")
    assert solver.info.dof==6 and solver.info.ee_frame=="flange-tool0"
    assert solver.info.joint_units==("rad",)*6
    assert hjcdik.Solver("ur5e",allow_build=False).info.artifact==solver.info.artifact


def test_mixed_joint_example_runs_from_other_directory(tmp_path,monkeypatch):
    example=runpy.run_path(str(ROOT/"examples/mixed_joint_warm_start.py"))
    monkeypatch.chdir(tmp_path)
    solver,out,warm,blocked=example["run"]()
    assert solver.info.joint_units==("rad","m")
    assert out.ok(5,.001).any(axis=1).all()
    assert warm is not None and warm.ok(5,.001).any(axis=1).all()
    assert blocked.count.tolist()==[0]
