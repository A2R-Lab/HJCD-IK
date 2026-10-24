"""01 — Open-world IK: batch-solve one 6-DoF target.

Explore many candidate configurations in parallel for one end-effector pose, then inspect the
best ones. Run: ``python examples/01_open_world_solve.py``
"""
import hjcdik

solver = hjcdik.Solver()
print(f"robot: {solver.info.robot}, DOF: {solver.info.dof}")

# A reachable target pose: [x, y, z, qw, qx, qy, qz] (position + unit quaternion).
target = solver.sample_targets(1, seed=0)[0]
print("target:", target)

# Explore 2000 candidates in parallel; request up to 4 distinct solutions.
# collision=None is an open-world solve: no self- or environment-collision check.
out = solver.solve(target, batch_size=2000, num_solutions=4, collision=None)
r = out[0]   # the un-padded view of the single target

print(f"returned {r.count} solutions")
print("joint configs shape:", r.q.shape)   # (count, DOF)
if r.count:
    # Keep the two errors paired: their separate minima can belong to different candidates.
    for i, (pos, ori) in enumerate(zip(r.pos_err_mm, r.ori_err_rad)):
        print(f"candidate {i}: position (mm)={pos:.3e}, orientation (rad)={ori:.3e}")
    print("candidates within 1 mm and 0.001 rad:", r.q[r.ok(pos_mm=1.0, ori_rad=0.001)])
else:
    print("No candidate was returned.")
# A returned candidate is approximate: check BOTH errors against your tolerances.
