"""03 — Batch-size sweep: accuracy vs. batch size.

Larger batches explore more candidates for ONE target. Coarse search uses one block per
candidate; refinement uses one warp per candidate. This reports accuracy, not performance:
accuracy need not improve monotonically, and runtime depends on hardware and settings.

Run: ``python examples/03_batch_sweep.py``
"""
from hjcdik import generate_solutions, sample_targets

target = sample_targets(num_targets=1, seed=0)[0]

print(f"{'batch':>8}  {'best pos err (mm)':>18}  {'best ori err (rad)':>20}")
for batch in (1, 10, 100, 1000, 2000):
    out = generate_solutions(target, batch_size=batch, num_solutions=1)
    if out["count"]:
        print(f"{batch:>8}  {float(out['pos_errors'].min()):>18.3e}  {float(out['ori_errors'].min()):>20.3e}")
    else:
        print(f"{batch:>8}  no candidate returned")
