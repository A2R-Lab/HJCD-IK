"""03 — Batch-size sweep: accuracy vs. batch size.

Larger batches explore more candidates in parallel, so the best returned solution is usually more
accurate — at a modest extra cost, since it is all one GPU block per problem. This sweeps the batch size
over a fixed target and reports the best position error at each.

Run: ``python examples/03_batch_sweep.py``
"""
from hjcdik import generate_solutions, sample_targets

target = sample_targets(num_targets=1, seed=0)[0]

print(f"{'batch':>8}  {'best pos err (m)':>18}  {'best ori err (rad)':>20}")
for batch in (1, 10, 100, 1000, 2000):
    out = generate_solutions(target, batch_size=batch, num_solutions=1)
    print(f"{batch:>8}  {float(out['pos_errors'].min()):>18.3e}  {float(out['ori_errors'].min()):>20.3e}")
