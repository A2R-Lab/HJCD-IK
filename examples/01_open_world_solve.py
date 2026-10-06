"""01 — Open-world IK: batch-solve a single 6-DOF target.

Generate many candidate solutions in parallel for one end-effector pose, then inspect the best ones.
Run: ``python examples/01_open_world_solve.py``
"""
from hjcdik import generate_solutions, sample_targets, num_joints

print(f"robot DOF: {num_joints()}")

# A reachable target pose: [x, y, z, qw, qx, qy, qz] (position + unit quaternion).
target = sample_targets(num_targets=1, seed=0)[0]
print("target:", target)

# Explore 2000 candidates in parallel; request up to 4 distinct solutions.
out = generate_solutions(target, batch_size=2000, num_solutions=4)

print(f"returned {out['count']} solutions")
print("joint configs shape:", out["joint_config"].shape)   # (count, DOF)
if out["count"]:
    # Keep the two errors paired: their separate minima can belong to different candidates.
    for i, (pos, ori) in enumerate(zip(out["pos_errors"], out["ori_errors"])):
        print(f"candidate {i}: position (mm)={pos:.3e}, orientation (rad)={ori:.3e}")
    acceptable = (out["pos_errors"] < 1.0) & (out["ori_errors"] < 0.001)
    print("candidates within 1 mm and 0.001 rad:", out["joint_config"][acceptable])
else:
    print("No candidate was returned.")
# A returned candidate is approximate: check BOTH errors against your tolerances.
