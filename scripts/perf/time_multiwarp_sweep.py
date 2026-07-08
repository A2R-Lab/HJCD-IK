#!/usr/bin/env python3
"""Turnkey multi-warp LM timing sweep — RUN ONLY ON A QUIET GPU.

Sweeps HJCD_LM_WARPS (W) x refine precision x batch B and reports the median wall
time of generate_solutions (which ends with cudaDeviceSynchronize, so wall captures
the full device pipeline). The lm_tuner floor is ~60% of wall (prior nsys: lm_tuner
~0.9ms of ~1.5ms at B=256), so a W win shows here; for the lm_tuner-only number, use
--nsys (extracts the per-kernel time via `nsys stats`).

SAFETY: refuses to run unless the GPU is quiet (util <= --max-util, no foreign compute
apps), re-checks after every config, and flags high-variance configs as likely
contended. Other agents share this GPU — if it aborts, just rerun when free.

Usage:
    .venv/bin/python scripts/perf/time_multiwarp_sweep.py                 # wall-clock sweep
    .venv/bin/python scripts/perf/time_multiwarp_sweep.py --nsys          # + lm_tuner kernel time
    .venv/bin/python scripts/perf/time_multiwarp_sweep.py --warps 1,2,4,8 --batches 256,2000,16384
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
    """Compute apps NOT belonging to this process (so the sweep's own CUDA context,
    which holds ~0.5GB, is never mistaken for contention)."""
    me = os.getpid()
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            text=True).strip()
        rows = [l for l in out.splitlines() if l.strip()]
        return [l for l in rows if l.split(",")[0].strip().isdigit() and int(l.split(",")[0]) != me]
    except Exception:
        return []

def quiet_or_die(max_util, where, check_util=True, settle_retries=6, settle_delay=2.0):
    """Abort on contention. util (time-averaged) is only meaningful BEFORE the sweep does
    its own GPU work — once we're solving, util reflects OUR kernels, so mid-sweep we only
    gate on FOREIGN compute apps (a different pid holding a context).

    A FOREIGN compute app → abort immediately. A high `util` reading with NO foreign app is
    treated as transient: `utilization.gpu` is time-averaged, so firing right after a prior
    sweep's teardown (or a rebuild) can read hot for a second or two. We poll a few times and
    only abort if it stays busy. Mid-sweep callers pass check_util=False and never sleep."""
    util = mem = 0
    for attempt in range(max(1, settle_retries)):
        apps = foreign_apps()
        if apps:
            util, mem = gpu_util()
            print(f"\n!! GPU NOT QUIET at {where}: util={util}% mem={mem}MiB foreign_apps={len(apps)}")
            print("   Timing under contention is meaningless. Aborting — rerun when free.")
            for a in apps: print("   foreign app:", a)
            sys.exit(2)
        util, mem = gpu_util()
        if not check_util or util <= max_util:
            return
        if attempt < settle_retries - 1:
            print(f"   [settle] {where}: util={util}% (no foreign apps) — likely teardown; "
                  f"waiting {settle_delay:.0f}s ({attempt + 1}/{settle_retries})")
            time.sleep(settle_delay)
    print(f"\n!! GPU still busy at {where}: util={util}% after {settle_retries} settles, no foreign apps.")
    print("   Aborting — rerun when the box is idle.")
    sys.exit(2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warps", default="1,2,4,8")
    ap.add_argument("--batches", default="256,2000,16384")
    ap.add_argument("--precisions", default="fp64,fp32")
    ap.add_argument("--num-solutions", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-util", type=int, default=3)
    ap.add_argument("--nsys", action="store_true", help="also extract lm_tuner kernel time via nsys")
    args = ap.parse_args()

    Ws = [int(x) for x in args.warps.split(",")]
    Bs = [int(x) for x in args.batches.split(",")]
    precs = args.precisions.split(",")

    quiet_or_die(args.max_util, "startup")
    import hjcdik
    target = hjcdik.sample_targets(num_targets=1, seed=args.seed)[0]

    print(f"# multi-warp timing sweep  num_solutions={args.num_solutions} "
          f"warmup={args.warmup} iters={args.iters}\n")
    print(f"| B | precision | W | median(ms) | min | max | spread% | us/sample |")
    print(f"|---|---|---|---|---|---|---|---|")

    rows = []
    for B in Bs:
        for prec in precs:
            fp64 = (prec == "fp64")
            for W in Ws:
                os.environ["HJCD_LM_WARPS"] = str(W)
                def solve():
                    return hjcdik.generate_solutions(target, batch_size=B,
                        num_solutions=args.num_solutions, refine_fp64=fp64)
                for _ in range(args.warmup): solve()
                quiet_or_die(args.max_util, f"B={B} {prec} W={W} (pre)", check_util=False)
                wall = []
                for _ in range(args.iters):
                    t0 = time.perf_counter(); solve(); wall.append((time.perf_counter()-t0)*1e3)
                quiet_or_die(args.max_util, f"B={B} {prec} W={W} (post)", check_util=False)
                med, lo, hi = statistics.median(wall), min(wall), max(wall)
                spread = 100.0*(hi-lo)/med if med else 0
                flag = " ⚠contended?" if spread > 25 else ""
                print(f"| {B} | {prec} | {W} | {med:.3f} | {lo:.3f} | {hi:.3f} | {spread:.0f}% | {med*1e3/B:.3f} |{flag}")
                rows.append((B, prec, W, med))

    # per-(B,precision) best-W summary
    print("\n## best W per (B, precision)")
    seen = {}
    for B, prec, W, med in rows:
        k = (B, prec)
        if k not in seen or med < seen[k][1]: seen[k] = (W, med)
    base = {(B,prec): med for (B,prec,W,med) in rows if W == Ws[0]}
    for (B, prec), (W, med) in sorted(seen.items()):
        b = base.get((B,prec))
        sp = f" ({b/med:.2f}x vs W={Ws[0]})" if b else ""
        print(f"- B={B} {prec}: best W={W} @ {med:.3f}ms{sp}")

    if args.nsys:
        print("\n## lm_tuner kernel time via nsys (run separately per config to keep it clean):")
        print("   for W in {0}; do HJCD_LM_WARPS=$W nsys profile -o /tmp/mw_$W --force-overwrite true \\".format(" ".join(map(str,Ws))))
        print("       .venv/bin/python scripts/perf/drive_generate.py --batch 2000 --iters 5; \\")
        print("     nsys stats --report cuda_gpu_kern_sum /tmp/mw_$W.nsys-rep 2>/dev/null | grep lm_tuner; done")

if __name__ == "__main__":
    main()
