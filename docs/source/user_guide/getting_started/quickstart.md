# Quickstart

```python
from hjcdik import generate_solutions, sample_targets, num_joints

print("DOF:", num_joints())

# Sample a reachable target: [x, y, z, qw, qx, qy, qz]
target = sample_targets(num_targets=1, seed=0)[0]

# Generate a batch of candidate IK solutions
out = generate_solutions(
    target,
    batch_size=2000,     # candidates explored in parallel
    num_solutions=4,     # distinct solutions to return
)
print("returned:", out["count"])
print("best position error:", out["pos_errors"].min())
print("joint configs shape:", out["joint_config"].shape)
```

## Collision-free solving
Pass `collision_free=True` along with a problem set (see the
[collision-free benchmark tutorial](../tutorials/collision_free_benchmark.md)).
