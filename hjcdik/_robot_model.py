"""Shared, CUDA-free validation of the serial models HJCD can solve.

The source generator loads this module by path so generating a first header does not
depend on an installed CUDA extension. Keep imports independent of the hjcdik package.
"""
import math


def validate_solver_robot(robot, ee_link=None):
    """Reject models outside the solver's current kinematic and sampling contract."""
    if robot.floating_base or not robot.is_serial_chain():
        raise ValueError("HJCD-IK requires a fixed-base serial chain")
    if not 1 <= robot.get_num_joints() <= 32:
        raise ValueError("HJCD-IK requires 1 to 32 actuated joints (one joint per warp lane)")
    for joint in robot.get_joints_ordered_by_id():
        if joint.is_mimic or joint.jtype not in ("revolute", "continuous"):
            raise ValueError("HJCD-IK requires independent revolute or continuous joints")
        if list(robot.get_S_by_id(joint.jid).reshape(-1)) != [0, 0, 1, 0, 0, 0]:
            raise ValueError("HJCD-IK requires each actuated joint to rotate around local +Z")
        if joint.jtype != "continuous":
            limits = joint.get_joint_limits()
            if (len(limits) != 2 or not all(math.isfinite(float(x)) for x in limits)
                    or not float(limits[0]) < float(limits[1])):
                raise ValueError(f"joint {joint.name!r} requires finite, strictly ordered limits")
    if ee_link is not None:
        target = robot.get_fixed_joint_by_name(ee_link)
        if target is None:
            raise ValueError(f"no fixed joint named {ee_link!r} (the end-effector target frame)")
        last = robot.get_joint_by_id(robot.get_num_joints() - 1)
        if target.parent_name != last.name:
            raise ValueError("HJCD-IK requires a fixed end-effector target attached to the final actuated joint")
