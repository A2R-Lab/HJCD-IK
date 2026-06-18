#!/usr/bin/env python3
"""fp32 vs fp64 LM-refine across convergence tolerance, num_solutions=1 (early-stop / latency-bound).

At the default 1e-8 m tol, fp32 can't reach the threshold (below its representable floor at ~0.5 m
coords) so it grinds all 40 iters and ends ~1.2x slower than fp64. This sweeps a precision-appropriate
looser tol (eps_pos[m]=eps_ori[rad]) to find where fp32 early-stops sooner and beats fp64 while staying
accurate enough. RUN ON A QUIET GPU (guard excludes our own pid; util-gated at startup).

Both eps knobs are read per-call (env), so this sweeps in one process. Reports median wall + achieved
accuracy (mean/max best pos err, mm) + solved-rate.
"""
import os, subprocess, statistics, sys, time
import numpy as np

B = 2000
TOLS = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]   # eps_pos (m) == eps_ori (rad)
N_ACC_TARGETS = 16
WARMUP, ITERS = 5, 30
SOLVE_POS_MM, SOLVE_ORI = 1.0, 1e-2     # solved if best pos<1mm and ori<0.01rad

def foreign_apps():
    me = os.getpid()
    try:
        out = subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,used_memory","--format=csv,noheader"],text=True).strip()
        return [l for l in out.splitlines() if l.strip() and l.split(",")[0].strip().isdigit() and int(l.split(",")[0])!=me]
    except Exception:
        return []

def util():
    try:
        return int(subprocess.check_output(["nvidia-smi","--query-gpu=utilization.gpu","--format=csv,noheader,nounits"],text=True).split("\n")[0])
    except Exception:
        return 0

def quiet_or_die(where, check_util=False):
    apps = foreign_apps()
    if apps or (check_util and util() > 3):
        print(f"\n!! GPU NOT QUIET at {where}: util={util()}% foreign_apps={len(apps)} -> abort")
        for a in apps: print("   foreign:", a)
        sys.exit(2)

def main():
    quiet_or_die("startup", check_util=True)
    import hjcdik
    os.environ["HJCD_LM_WARPS"] = "1"   # W=1 is fastest (W>1 hurts on big GPUs)
    targets = hjcdik.sample_targets(num_targets=N_ACC_TARGETS, seed=0)
    t_time = targets[0]

    print(f"# fp32 tolerance sweep  B={B} num_solutions=1 W=1  warmup={WARMUP} iters={ITERS}\n")
    print("| eps (m=rad) | prec | median(ms) | mean best_pos(mm) | max best_pos(mm) | solved/N |")
    print("|---|---|---|---|---|---|")

    rows = []
    for tol in TOLS:
        os.environ["HJCD_LM_EPS_POS"] = repr(tol)
        os.environ["HJCD_LM_EPS_ORI"] = repr(tol)
        for prec, rf in [("fp64", 1), ("fp32", 0)]:
            def solve(t):
                return hjcdik.generate_solutions(t, batch_size=B, num_solutions=1, refine_fp64=rf)
            for _ in range(WARMUP): solve(t_time)
            quiet_or_die(f"tol={tol:.0e} {prec} (pre)")
            wall = []
            for _ in range(ITERS):
                t0 = time.perf_counter(); solve(t_time); wall.append((time.perf_counter()-t0)*1e3)
            # accuracy over the target set
            errs, solved = [], 0
            for t in targets:
                o = solve(t)
                pe = float(np.min(np.array(o["pos_errors"]))); oe = float(np.min(np.array(o["ori_errors"])))
                errs.append(pe)
                if pe < SOLVE_POS_MM and oe < SOLVE_ORI: solved += 1
            quiet_or_die(f"tol={tol:.0e} {prec} (post)")
            med = statistics.median(wall)
            print(f"| {tol:.0e} | {prec} | {med:.3f} | {np.mean(errs):.3e} | {np.max(errs):.3e} | {solved}/{len(targets)} |")
            rows.append((tol, prec, med, float(np.mean(errs)), solved))

    # summary: fp32-vs-fp64 wall ratio per tol + best fp32 that stays accurate
    print("\n## fp32/fp64 wall ratio per tol (and fp32 mean accuracy)")
    by = {}
    for tol, prec, med, acc, solved in rows: by[(tol,prec)] = (med, acc, solved)
    for tol in TOLS:
        m64,_,_ = by[(tol,"fp64")]; m32,a32,s32 = by[(tol,"fp32")]
        tag = "fp32 FASTER" if m32 < m64 else "fp32 slower"
        print(f"- tol={tol:.0e}: fp32 {m32:.3f}ms vs fp64 {m64:.3f}ms -> {m64/m32:.2f}x ({tag}); "
              f"fp32 acc {a32:.2e}mm solved {s32}/{N_ACC_TARGETS}")

if __name__ == "__main__":
    main()
