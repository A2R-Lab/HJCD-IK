#!/usr/bin/env python3
"""DoF-24 regression: fp32-vs-fp64 LM-solve A/B for the CURRENTLY-BUILT robot.

Tests the prime hypothesis from docs/open-tasks/dof_scaling_regression_2026-06-18.md:
the high-DoF time blow-up (24-DoF ~17 ms vs paper ~4.6 ms) is the fp64 O(DoF^3)
warp-Cholesky in the LM solve. num_solutions=1 auto-selects fp64; forcing fp32
(refine_fp64=0) should collapse the time toward flat if fp64 is the cause.

This script times ONE build (whatever grid.cuh is currently compiled in). The
shell orchestrator `dof_scaling_ab.sh` regenerates+rebuilds each DoF and calls this.

Two modes:
  (default A/B)      time fp64 and fp32 + report accuracy + the fp64/fp32 ratio
  --profile-prec P   run only precision P (warmup + iters), for wrapping under nsys

SAFETY: refuses to run unless the GPU is quiet (shared with other agents). Timing
under contention is meaningless — it aborts and you rerun when free.

Usage:
    .venv/bin/python scripts/perf/dof_scaling_ab.py --batch 1000
    nsys profile -o /tmp/dof_fp64 --force-overwrite true \
        .venv/bin/python scripts/perf/dof_scaling_ab.py --profile-prec fp64 --batch 1000
"""
import argparse, os, subprocess, statistics, sys, time


def gpu_util():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            text=True).strip().splitlines()[0]
        util, mem = [int(x) for x in out.split(",")]
        return util, mem
    except Exception as e:
        print("WARN: nvidia-smi failed:", e); return 0, 0


def foreign_apps():
    """Compute apps NOT belonging to this process (our own CUDA context is not contention)."""
    me = os.getpid()
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            text=True).strip()
        rows = [l for l in out.splitlines() if l.strip()]
        return [l for l in rows if l.split(",")[0].strip().isdigit() and int(l.split(",")[0]) != me]
    except Exception:
        return []


def quiet_or_die(max_util, where, check_util=True):
    apps = foreign_apps()
    util, mem = gpu_util()
    if (check_util and util > max_util) or apps:
        print(f"\n!! GPU NOT QUIET at {where}: util={util}% mem={mem}MiB foreign_apps={len(apps)}")
        print("   Timing under contention is meaningless. Aborting — rerun when free.")
        for a in apps: print("   foreign app:", a)
        sys.exit(2)


def _median(xs):
    try:
        import numpy as np
        return float(np.median(np.asarray(xs, dtype=float)))
    except Exception:
        flat = list(xs) if hasattr(xs, "__iter__") else [xs]
        return statistics.median(flat) if flat else float("nan")


def accuracy(out):
    """median position error (mm) and orientation error (rad) from a solve dict."""
    pos = out.get("pos_errors", [])
    ori = out.get("ori_errors", [])
    return _median(pos) * 1e3, _median(ori)


def set_tol(tol_str):
    """Apply a 'pos:ori' tolerance (metres:radians) via the LM early-stop env knobs.
    The kernel reads HJCD_LM_EPS_POS/ORI at generate_solutions() call time (no recompile)."""
    pos, ori = tol_str.split(":")
    os.environ["HJCD_LM_EPS_POS"] = pos
    os.environ["HJCD_LM_EPS_ORI"] = ori
    return float(pos), float(ori)


def time_prec(hjcdik, target, B, ns, fp64, warmup, iters, max_util):
    def solve():
        return hjcdik.generate_solutions(target, batch_size=B, num_solutions=ns,
                                         refine_fp64=(1 if fp64 else 0))
    last = None
    for _ in range(warmup):
        last = solve()
    quiet_or_die(max_util, f"{'fp64' if fp64 else 'fp32'} (pre)", check_util=False)
    wall = []
    for _ in range(iters):
        t0 = time.perf_counter(); last = solve(); wall.append((time.perf_counter() - t0) * 1e3)
    quiet_or_die(max_util, f"{'fp64' if fp64 else 'fp32'} (post)", check_util=False)
    return wall, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--num-solutions", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-util", type=int, default=3)
    ap.add_argument("--profile-prec", choices=["fp32", "fp64"], default=None,
                    help="run ONLY this precision (for nsys/ncu wrapping); no A/B")
    ap.add_argument("--tols", default="1e-8:1e-8",
                    help="comma-separated 'pos:ori' tolerances (m:rad) to sweep; "
                         "default tight 1e-8:1e-8. e.g. 1e-8:1e-8,1e-4:1e-3,1e-3:1e-2")
    ap.add_argument("--tol", default=None, help="single 'pos:ori' tol for --profile-prec mode")
    ap.add_argument("--precisions", default="fp64,fp32", help="comma list: fp64,fp32")
    args = ap.parse_args()

    quiet_or_die(args.max_util, "startup")
    import hjcdik
    N = hjcdik.num_joints()
    target = hjcdik.sample_targets(num_targets=1, seed=args.seed)[0]

    if args.profile_prec:
        if args.tol:
            set_tol(args.tol)
        fp64 = (args.profile_prec == "fp64")
        time_prec(hjcdik, target, args.batch, args.num_solutions, fp64, args.warmup, args.iters, args.max_util)
        print(f"# profiled DoF={N} {args.profile_prec} tol={args.tol or 'default'} B={args.batch} "
              f"(warmup={args.warmup} iters={args.iters}) — see nsys report")
        return

    tols = [t for t in args.tols.split(",") if t]
    precs = [p for p in args.precisions.split(",") if p]
    print(f"\n## DoF={N}  B={args.batch}  num_solutions={args.num_solutions}  "
          f"(warmup={args.warmup} iters={args.iters})")
    print("| DoF | precision | tol(pos:ori) | median(ms) | min | max | us/sample | pos_err(mm) | ori_err(rad) |")
    print("|---|---|---|---|---|---|---|---|---|")
    res = {}  # (prec, tol) -> median ms
    for prec in precs:
        fp64 = (prec == "fp64")
        for tol in tols:
            set_tol(tol)
            wall, out = time_prec(hjcdik, target, args.batch, args.num_solutions, fp64,
                                  args.warmup, args.iters, args.max_util)
            med, lo, hi = statistics.median(wall), min(wall), max(wall)
            pmm, orad = accuracy(out)
            res[(prec, tol)] = med
            print(f"| {N} | {prec} | {tol} | {med:.3f} | {lo:.3f} | {hi:.3f} | "
                  f"{med*1e3/args.batch:.3f} | {pmm:.4f} | {orad:.2e} |")

    # fp32-vs-fp64 ratio at the tightest tol (hypothesis #1: fp64 solve dominates?)
    tight = tols[0]
    if ("fp64", tight) in res and ("fp32", tight) in res and res[("fp32", tight)]:
        r = res[("fp64", tight)] / res[("fp32", tight)]
        print(f"\n**DoF={N} @ tol={tight}: fp64/fp32 = {r:.2f}x** "
              f"({'fp32 faster -> fp64 solve dominates' if r > 1.5 else 'fp32 NOT the dominant cost'})")
    # tolerance effect (hypothesis #3: does looser tol early-stop shorten the loop?)
    if len(tols) > 1:
        for prec in precs:
            base = res.get((prec, tight))
            if not base:
                continue
            deltas = []
            for tol in tols[1:]:
                m = res.get((prec, tol))
                if m:
                    deltas.append(f"{tol}={m:.3f}ms ({base/m:.2f}x)")
            print(f"**DoF={N} {prec} tol effect** (vs tight {base:.3f}ms): " + ", ".join(deltas)
                  + ("  <- looser tol shortens loop" if any(base/res[(prec,t)] > 1.15
                     for t in tols[1:] if res.get((prec,t))) else "  <- tol has ~no effect (early-stop not firing)"))


if __name__ == "__main__":
    main()
