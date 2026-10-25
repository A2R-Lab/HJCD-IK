"""Compile and solve the advertised one- and 32-coordinate boundaries, without timing."""
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from test_general_joints import build_solver,targets_for,independent_quality,diagnostic,assert_poses


@pytest.fixture(scope="module",params=[1,32])
def boundary_model(request,tmp_path_factory):
    n=request.param
    root=ET.Element("robot",name=f"boundary_{n}")
    for i in range(n+2):
        link=ET.SubElement(root,"link",name=f"link{i}")
        if i:
            inertia=ET.SubElement(link,"inertial")
            ET.SubElement(inertia,"mass",value="1")
            ET.SubElement(inertia,"inertia",ixx="1",iyy="1",izz="1",ixy="0",ixz="0",iyz="0")
    for i in range(n+1):
        kind="fixed" if i==n else "revolute"
        joint=ET.SubElement(root,"joint",name="tool" if i==n else f"joint{i}",type=kind)
        ET.SubElement(joint,"parent",link=f"link{i}"); ET.SubElement(joint,"child",link=f"link{i+1}")
        ET.SubElement(joint,"origin",xyz="0.04 0.01 0.02",rpy="0.3 0.2 0.1")
        if kind!="fixed":
            ET.SubElement(joint,"axis",xyz="0 0 -1" if n==1 else "0 0 1")
            ET.SubElement(joint,"limit",lower="-1",upper="1",
                          effort="1",velocity="1")
    path=tmp_path_factory.mktemp("boundary")/"robot.urdf"
    ET.ElementTree(root).write(path)
    solver,module=build_solver(path,f"test-boundary-{n}",target="tool")
    return solver,module,path


@pytest.mark.parametrize("precision",["fp32","fp64"])
def test_joint_count_boundary(boundary_model,precision):
    solver,module,path=boundary_model
    limits=np.array(solver.info.joint_limits)
    qs=np.random.default_rng(97).uniform(limits[:,0],limits[:,1],(8,solver.info.dof))
    targets=targets_for(path,qs,"tool")
    assert_poses(diagnostic(solver,module,qs,precision=="fp64")[0],targets,5e-6 if precision=="fp32" else 1e-9)
    out=solver.solve(targets,batch_size=2000,num_solutions=4,precision=precision,collision=None)
    assert independent_quality(solver,path,targets,out,"tool")[0].all()
    if solver.info.dof==1:
        unreachable=targets[0].copy(); unreachable[0]+=10
        failed=solver.solve(unreachable,batch_size=37,precision=precision,collision=None)
        assert not failed.ok(5,.001).any()  # does not infer infeasibility from a failed solver
