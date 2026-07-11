#!/usr/bin/env python3
"""Minimal driver for profiling `generate_solutions` under nsys/ncu.

Runs a fixed-seed target through the solver: a few warmup calls (to amortize
context/JIT/codegen init) then N timed iterations at a chosen batch size. Kept
deliberately tiny so an nsys timeline / ncu report is dominated by the kernel
pipeline, not Python.

Usage:
    python scripts/perf/drive_generate.py --batch 2000 --warmup 3 --iters 10
    nsys profile -o report python scripts/perf/drive_generate.py --batch 2000 --iters 5

Isolate timing: confirm the GPU is idle (`nvidia-smi`, util ~0) before trusting numbers.
"""
import argparse
import statistics
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--num-solutions", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import hjcdik

    target = hjcdik.sample_targets(num_targets=1, seed=args.seed)[0]

    for _ in range(args.warmup):
        hjcdik.generate_solutions(target, batch_size=args.batch,
                                  num_solutions=args.num_solutions)

    wall = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        hjcdik.generate_solutions(target, batch_size=args.batch,
                                  num_solutions=args.num_solutions)
        wall.append((time.perf_counter() - t0) * 1e3)

    def stats(xs):
        return f"median={statistics.median(xs):.3f} ms  min={min(xs):.3f}  max={max(xs):.3f}"

    med = statistics.median(wall)
    print(f"batch={args.batch} num_solutions={args.num_solutions} "
          f"iters={args.iters} (warmup={args.warmup})")
    # Python wall-clock; generate_ik_solutions ends with cudaDeviceSynchronize so this
    # captures the full device pipeline (plus a small fixed Python/pybind overhead).
    print(f"  wall: {stats(wall)}  ({med*1e3/args.batch:.3f} us/sample)")


if __name__ == "__main__":
    main()
