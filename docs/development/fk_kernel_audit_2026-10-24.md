# Audit: HJCD-IK's `forward_kinematics_kernel` against GRiD's end-effector pose (2026-10-24)

Written for: HJCD-IK maintainers. Closes the "FK-kernel duplicate" item of the P1 plan.

## What the kernel is

`forward_kinematics_kernel` (`csrc/kernel/hjcd_kernel.cu`) maps `B` joint configurations to end-effector
poses `[x, y, z, qw, qx, qy, qz]`, one block per configuration. It is used only by
`sample_random_target_poses`, that is by `Solver.sample_targets`. Its body is three calls:
GRiD's `load_update_XmatsHom_helpers` (joint transforms), HJCD's `ee_fk_thread` (GRiD's
`ee_pose_inner_thread` plus the fixed tool-frame product at `GRASP_FIXED_IDX`, the same path the
solver uses for every candidate) and `store_ee_pose7` (rotation matrix to a scalar-first quaternion).
Thirty lines, no numerics of its own.

## What GRiD offers

The generated header carries `end_effector_pose_device_<target>` and the kernels around it. They
compose the same named fixed frame, but their output is GRiD's six-vector `[xyz; rpy]`: roll, pitch
and yaw from `atan2`/`asin` of the rotation matrix. HJCD needs a quaternion. Converting on the host
would add a conversion step with its own branch at the pitch singularity and a second source of
rounding, and the solver's own FK path (`ee_fk_thread`) would no longer be the one that produced the
sampled targets, which `tests/test_fk_equivalence.py` relies on when it checks the returned pose
against an independent URDF FK.

## Verdict

Keep the kernel. It is not a duplicate of GRiD's kinematics; it is a thirty-line driver over GRiD's
kinematics with the output representation the solver needs, and it keeps one FK path between
sampling and solving. The only thing GRiD could add is a quaternion output option on
`end_effector_pose` (a small ask if it ever matters for another consumer); nothing in HJCD waits on it.
