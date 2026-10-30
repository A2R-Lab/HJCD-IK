"""General-joint correctness, independent derivatives and the predeclared scale gate.

No wall-clock performance assertions. Runtime artifacts are cached between runs.
The tiny mixed chain isolates scale/axis conventions; Panda variants cover redundancy.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import hjcdik
from hjcdik._build import import_artifact
from test_joint_fixtures import ROOT, configs, fk, gt

CACHE = os.environ.get("HJCDIK_TEST_CACHE") or None


def build_solver(path, name, spheres=None, target="panda_grasptarget_hand", diagnostics=True):
    artifact = hjcdik.build(path, ee_link=target, spheres=spheres, name=name, cache_dir=CACHE,
                            _diagnostics=diagnostics)
    return hjcdik.Solver(name, cache_dir=CACHE), import_artifact(artifact)


@pytest.fixture(scope="module")
def panda_models():
    result = {}
    for name in ("panda", "panda_axes", "panda_negz", "panda_slider"):
        path = ROOT / ("csrc/urdf/panda.urdf" if name == "panda" else f"tests/robots/{name}.urdf")
        spheres = "panda_foam" if name == "panda" else (
            ROOT / "tests/robots/panda_axes_spheres.urdf" if name == "panda_axes" else None)
        result[name] = (*build_solver(path, "test-general-"+name, spheres), path)
    return result


def diagnostic(solver, module, q, fp64=True, margin=0, check_self=True):
    q = np.atleast_2d(q)
    raw = module._diagnostic_model(q.ravel().tolist(), fp64, solver._world, solver.info.device, margin, check_self)
    rows = np.array(raw).reshape(len(q), 8+6*solver.info.dof)
    return rows[:, :7], rows[:, 7:-1].reshape(len(q), 6, solver.info.dof), rows[:, -1].astype(bool)


def targets_for(path, qs, target="panda_grasptarget_hand"):
    return np.array([np.r_[t[:3, 3], gt._quat_from_R(t[:3, :3])] for t in (fk(path,q,target) for q in qs)])


def assert_poses(actual, expected, tolerance):
    np.testing.assert_allclose(actual[:, :3], expected[:, :3], atol=tolerance, rtol=0)
    dots = np.sum(actual[:, 3:]*expected[:, 3:], axis=1)
    np.testing.assert_allclose(actual[:, 3:]*np.where(dots<0, -1, 1)[:,None], expected[:, 3:],
                               atol=tolerance, rtol=0)


@pytest.mark.parametrize("name", ["panda", "panda_axes", "panda_negz", "panda_slider"])
@pytest.mark.parametrize("fp64", [False, True])
def test_fk_and_actual_jacobian(panda_models, name, fp64):
    solver, module, path = panda_models[name]
    q = configs()
    if name == "panda_negz": q = -q
    if name == "panda_slider": q = np.column_stack([np.linspace(0, .5, len(q)), q])
    poses, jacobian, _ = diagnostic(solver,module,q,fp64)
    # GRiD folds near-zero fixed-frame entries; the unchanged Panda differs from
    # direct URDF trig by ~6e-12 m. Use the planned 1e-9 double FK tolerance.
    assert_poses(poses, targets_for(path,q), 3e-6 if not fp64 else 1e-9)
    for sample in range(12):
        center = fk(path,q[sample])
        for h in (1e-5, 3e-6):
            expected = np.empty((6,solver.info.dof))
            for j in range(solver.info.dof):
                delta = np.zeros(solver.info.dof); delta[j]=h
                plus, minus = fk(path,q[sample]+delta), fk(path,q[sample]-delta)
                expected[:3,j]=(plus[:3,3]-minus[:3,3])/(2*h)
                omega=((plus[:3,:3]-minus[:3,:3])/(2*h)) @ center[:3,:3].T
                expected[3:,j]=[omega[2,1],omega[0,2],omega[1,0]]
            np.testing.assert_allclose(jacobian[sample],expected,atol=3e-6 if not fp64 else 2e-9,rtol=0)


def test_collision_verdicts_match_reexpressed_robot(panda_models):
    q=configs()
    a,am,_=panda_models["panda"]; b,bm,_=panda_models["panda_axes"]
    problem=hjcdik.worlds.select_problem(json.loads((ROOT/"tests/mb_problems.json").read_text()),"box_panda",0)
    arrays=hjcdik.worlds.from_mbm_problem(problem)
    a.set_world(**arrays); b.set_world(**arrays)
    for margin in (0,.02):
        for check_self in (False,True):
            np.testing.assert_array_equal(diagnostic(a,am,q,margin=margin,check_self=check_self)[2],
                                          diagnostic(b,bm,q,margin=margin,check_self=check_self)[2])
    a.clear_world(); b.clear_world()


def independent_quality(solver, path, targets, outputs, target="panda_grasptarget_hand"):
    good=[]; positions=[]; angles=[]
    limits=np.array(solver.info.joint_limits)
    for pose,result in zip(targets,outputs):
        success=False
        for q in result.q:
            assert np.isfinite(q).all()
            assert ((q>=limits[:,0]-2e-6)&(q<=limits[:,1]+2e-6)).all()
            t=fk(path,q,target)
            p=np.linalg.norm(t[:3,3]-pose[:3])
            dot=min(1.,abs(np.dot(gt._quat_from_R(t[:3,:3]),pose[3:])))
            angle=2*np.arccos(dot)
            success |= p<5e-3 and angle<1e-3
            positions.append(p); angles.append(angle)
        good.append(success)
    return np.array(good), np.array(positions), np.array(angles)


@pytest.mark.parametrize("solutions", [1,4])
@pytest.mark.parametrize("collision", [None,"hard"])
def test_ik_representation_parity(panda_models, solutions, collision):
    a,_,path=panda_models["panda"]
    b,_,variant=panda_models["panda_axes"]
    if collision:
        problem=hjcdik.worlds.select_problem(json.loads((ROOT/"tests/mb_problems.json").read_text()),"box_panda",0)
        arrays=hjcdik.worlds.from_mbm_problem(problem); a.set_world(**arrays); b.set_world(**arrays)
    targets=targets_for(path,configs(100))
    results=[]
    for solver,robot in ((a,path),(b,variant)):
        out=solver.solve(targets,batch_size=2000,num_solutions=solutions,collision=collision)
        results.append(independent_quality(solver,robot,targets,out))
    assert abs(results[0][0].mean()-results[1][0].mean())<=.02+1e-12
    # Compare medians over all returned candidates, including fallback candidates.
    for k,floor in ((1,1e-5),(2,1e-4)):
        x,y=results[0][k],results[1][k]
        assert len(x) and len(y)
        assert np.quantile(y,.5)<=max(np.quantile(x,.5)*1.1,floor)
    a.clear_world(); b.clear_world()


def tiny_robot(path, travel, slider_first):
    root=ET.Element("robot",name="mixed_chain")
    for i in range(4):
        link=ET.SubElement(root,"link",name=f"link{i}")
        if i:
            inertia=ET.SubElement(link,"inertial")
            ET.SubElement(inertia,"mass",value="1")
            ET.SubElement(inertia,"inertia",ixx="1",iyy="1",izz="1",ixy="0",ixz="0",iyz="0")
            collision=ET.SubElement(link,"collision")
            ET.SubElement(collision,"origin",xyz="0 0 0")
            ET.SubElement(ET.SubElement(collision,"geometry"),"sphere",radius="0.02")
    kinds=["prismatic","revolute"] if slider_first else ["revolute","prismatic"]
    for i,kind in enumerate(kinds+["fixed"]):
        j=ET.SubElement(root,"joint",name="tool" if kind=="fixed" else f"j{i}",type=kind)
        ET.SubElement(j,"parent",link=f"link{i}"); ET.SubElement(j,"child",link=f"link{i+1}")
        ET.SubElement(j,"origin",xyz="0.2 0.1 0.15",rpy="0.1 0.2 -0.1")
        if kind!="fixed":
            ET.SubElement(j,"axis",xyz="0.6 0 0.8" if kind=="prismatic" else "0 0.6 0.8")
            ET.SubElement(j,"limit",lower="0" if kind=="prismatic" else "-2",
                          upper=str(travel) if kind=="prismatic" else "2",effort="10",velocity="1")
    ET.ElementTree(root).write(path)


@pytest.fixture(scope="module",params=[(travel,first) for travel in (.01,.5,5.) for first in (False,True)])
def mixed_model(request,tmp_path_factory):
    travel,first=request.param
    path=tmp_path_factory.mktemp("mixed-model")/"robot.urdf"
    tiny_robot(path,travel,first)
    # A content-based side-header location ensures the temporary URDF directory is not a cache input.
    solver,module=build_solver(path,f"test-mixed-{travel}-{first}",target="tool")
    return solver,module,path,travel,first


@pytest.mark.parametrize("precision", ["fp32","fp64"])
@pytest.mark.parametrize("solutions", [1,4])
def test_mixed_joint_scale_gate(mixed_model,precision,solutions,record_property):
    solver,module,path,travel,first=mixed_model
    limits=np.array(solver.info.joint_limits); slider=0 if first else 1
    cells=[]
    for seed in (13,29):
        qs=np.random.default_rng(seed).uniform(limits[:,0],limits[:,1],(32,2))
        qs[:2]=limits.T  # include exact bounds
        # Half the targets stay in the common 1 cm workspace across travel ranges.
        qs[16:,slider]*=.01/travel
        targets=targets_for(path,qs,"tool")
        poses,_,_=diagnostic(solver,module,qs,precision=="fp64")
        assert_poses(poses,targets,4e-6 if precision=="fp32" else 1e-9)
        for warm in (False,True):
            init=np.clip(qs+np.array([.003,.003]),limits[:,0],limits[:,1])[:,None,:] if warm else None
            out=solver.solve(targets,batch_size=512,num_solutions=solutions,precision=precision,
                             collision=None,q_init=init,seed=seed)
            good,positions,angles=independent_quality(solver,path,targets,out,"tool")
            cells.append(dict(travel_m=travel,slider_first=first,precision=precision,solutions=solutions,
                              seed=seed,warm=warm,attempted=len(targets),accurate=int(good.sum()),
                              empty=int((out.count==0).sum()),max_position_m=float(positions.max(initial=0)),
                              max_orientation_rad=float(angles.max(initial=0))))
            # Known-feasible, low-dimensional analytic fixtures: >=31/32 per cell, not pooled.
            assert good.sum()>=31,(travel,first,precision,solutions,seed,warm,good.tolist())
    record_property("scale_cells",json.dumps(cells,sort_keys=True))


def test_runtime_header_identity_across_processes(tmp_path):
    code="""import hjcdik,json
p=hjcdik.build('tests/robots/panda_axes.urdf',ee_link='panda_grasptarget_hand',name='test-general-stable',plan_only=True)
print('KEY='+p['input_key'])
"""
    keys=[]
    for _ in range(2):
        run=subprocess.run([sys.executable,"-c",code],cwd=ROOT,capture_output=True,text=True,check=True)
        keys.append(next(line for line in run.stdout.splitlines() if line.startswith("KEY=")))
    assert keys[0]==keys[1]


def test_joint_metadata_and_negative_axis_ik(panda_models):
    solver, _, path = panda_models["panda_negz"]
    assert solver.info.joint_names == tuple(f"panda_joint{i}" for i in range(1,8))
    assert solver.info.joint_units == ("rad",)*7
    assert solver.info.joint_types == ("revolute",)*7
    assert isinstance(solver.info.joint_limits, tuple)
    assert all(isinstance(bounds, tuple) for bounds in solver.info.joint_limits)
    targets = targets_for(path, -configs(32))
    out = solver.solve(targets, batch_size=2000, num_solutions=4, collision=None)
    assert independent_quality(solver,path,targets,out)[0].sum() >= 31


def test_slider_reaches_beyond_fixed_panda_bound(panda_models):
    solver, _, path = panda_models["panda_slider"]
    assert solver.info.joint_units == ("m",)+("rad",)*7
    fixed = ROOT/"csrc/urdf/panda.urdf"
    chain = gt._chain_to_target(gt._parse_joints(fixed), "panda_grasptarget_hand")
    # Triangle inequality: each rotated origin contributes at most its norm.
    bound = sum(np.linalg.norm(j["xyz"]) for j in chain)
    qs = np.column_stack([np.full(800,.5), configs(800)])
    targets = targets_for(path, qs)
    selected = np.flatnonzero(np.linalg.norm(targets[:,:3],axis=1)>bound+.01)[:8]
    assert len(selected) >= 4, "fixture must exceed a geometric bound, not just defeat fixed-arm IK"
    targets, qs = targets[selected], qs[selected]
    for warm in (False,True):
        out = solver.solve(targets,batch_size=2000,num_solutions=4,collision=None,
                           q_init=qs[:,None,:] if warm else None)
        assert independent_quality(solver,path,targets,out)[0].all()


@pytest.fixture(scope="module")
def mixed_collision(tmp_path_factory):
    path = tmp_path_factory.mktemp("mixed-collision")/"robot.urdf"
    tiny_robot(path,.5,False)
    solver,module = build_solver(path,"test-mixed-collision",spheres=path,target="tool")
    return solver,module,path


@pytest.mark.parametrize("precision", ["fp32","fp64"])
@pytest.mark.parametrize("mode", ["hard","both"])
def test_mixed_collision_partial_warps(mixed_collision,precision,mode,monkeypatch):
    solver,module,path = mixed_collision
    monkeypatch.setenv("HJCD_LM_WARPS","3")  # B*16 is not divisible by three.
    qs = np.array([[.2,0],[.2,.5]])
    targets = targets_for(path,qs,"tool")
    solver.set_world(spheres=[[*targets[1,:3],.08]])
    for margin in (0,.02):
        for check_self in (False,True):
            free = diagnostic(solver,module,qs,precision=="fp64",margin,check_self)[2]
            assert free.tolist() == [True,False]
            out = solver.solve(targets,batch_size=37,num_solutions=4,precision=precision,
                               collision=mode,collision_margin_m=margin,self_collision=check_self,
                               q_init=qs[:,None,:],seed=17)
            assert independent_quality(solver,path,targets[:1],[out[0]],"tool")[0].all()
            for result in out:
                if result.count:
                    assert diagnostic(solver,module,result.q,precision=="fp64",margin,check_self)[2].all()
            # The tool sphere's entire 5 mm pose band lies inside the obstacle.
            assert out.count[1] == 0
    solver.clear_world()


@pytest.mark.parametrize("reverse", [False,True])
def test_same_dof_modules_isolated_without_toolchain(panda_models,reverse,tmp_path):
    order = ["panda_axes","panda_negz"]
    if reverse: order.reverse()
    code = '''import os,shutil,sys,numpy as np,hjcdik
assert shutil.which("nvcc") is None
import grid_rbd
def no_build(*a,**k): raise AssertionError("load must not compile")
grid_rbd.build_artifact=no_build
sys.path.insert(0,"tests")
from test_general_joints import diagnostic,targets_for,configs,assert_poses,ROOT
from hjcdik._build import import_artifact
for name in sys.argv[1:]:
    artifact=hjcdik.load("test-general-"+name,cache_dir=os.environ.get("HJCDIK_TEST_CACHE") or None)
    solver=hjcdik.Solver("test-general-"+name,allow_build=False,cache_dir=os.environ.get("HJCDIK_TEST_CACHE") or None)
    q=configs(12)*(-1 if name=="panda_negz" else 1)
    actual=diagnostic(solver,import_artifact(artifact),q)[0]
    assert_poses(actual,targets_for(ROOT/("tests/robots/"+name+".urdf"),q),1e-9)
'''
    env = dict(os.environ,PATH=str(tmp_path))
    subprocess.run([sys.executable,"-c",code,*order],cwd=ROOT,env=env,check=True,
                   capture_output=True,text=True)
