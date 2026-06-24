"""01 — Open-world IK: batch-solve a single 6-DOF target.

Generate many candidate solutions in parallel for one end-effector pose, then inspect the best ones.
Run: ``python examples/01_open_world_solve.py``
"""
from hjcdik import generate_solutions, sample_targets, num_joints

print(f"robot DOF: {num_joints()}")

# A reachable target pose: [x, y, z, qw, qx, qy, qz] (position + unit quaternion).
target = sample_targets(num_targets=1, seed=0)[0]
print("target:", target)

# Explore 2000 candidates in parallel; return the 4 best distinct solutions.
out = generate_solutions(target, batch_size=2000, num_solutions=4)

print(f"returned {out['count']} solutions")
print("joint configs shape:", out["joint_config"].shape)   # (num_solutions, DOF)
print("best position error (m):   ", float(out["pos_errors"].min()))
print("best orientation error (rad):", float(out["ori_errors"].min()))
