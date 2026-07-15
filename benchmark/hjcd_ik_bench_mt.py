#!/usr/bin/env python3
"""Multi-target counterpart of benchmark/hjcd_ik_bench.py (G1, or any K>=2 build).

hjcd_ik_bench.py drives generate_solutions(), which does not exist on a multi-target build: a
7-vector target is ambiguous when the robot has K target frames. This harness drives hjcdik.solve()
instead, and otherwise mirrors hjcd_ik_bench exactly -- same flag names, same CSV columns, same flat
YAML, same "Batch-Size" semantics (candidate restarts per target), same target sampling.

TARGET SAMPLING matches hjcd_ik_bench: a Cranley-Patterson-scrambled Halton sequence over the FULL
joint limits, then FK to read off where the K frames land. gen_targets._halton(cranley-patterson) is
a validated transcription of the CUDA sampler (tests/test_fk_equivalence.py checks it bitwise), so
both harnesses draw target configurations from the same distribution.

Build for G1 first (see scripts/dev/g1_check.sh for the exact codegen line), then:

    python benchmark/hjcd_ik_bench_mt.py --num-targets 100 --batches 1,10,100,1000,2000
    python benchmark/hjcd_ik_bench_mt.py --masks both_hands,all_four --csv-out g1.csv
    python benchmark/hjcd_ik_bench_mt.py --seed-mode nearby        # warm-start regime

Multi-target-only flags (no hjcd_ik_bench equivalent): --masks, --coarse-mode, --coarse-iters,
--lm-iters, --seed-mode, --sampler, --margin, --position-tol, --orientation-tol.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hjcdik                                   # noqa: E402
from gen_targets import _halton                 # noqa: E402
from hjcd_ik_bench import _parse_batches        # noqa: E402  (same batch-list parsing)

NAMES = ["left_hand", "right_hand", "left_foot", "right_foot"]
MASKS = {
    "left_hand": 0b0001, "right_hand": 0b0010, "both_hands": 0b0011,
    "left_foot": 0b0100, "right_foot": 0b1000, "both_feet": 0b1100,
    "all_four": 0b1111, "hands_and_feet": 0b1111,
}


def quat_from_R(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25*s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s]
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2
            q = [(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s]
        elif i == 1:
            s = np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2
            q = [(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s]
        else:
            s = np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2
            q = [(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s]
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def sample_target_configs(n, lo, hi, sampler, margin, seed):
    """Joint configs drawn the way hjcdik.sample_targets draws them: scrambled Halton, full range."""
    span = hi - lo
    a, b = lo + margin * span, hi - margin * span
    if sampler == "halton":
        u = _halton(n, len(lo), skip=0, scramble=True, seed=seed, method="cranley-patterson")
    else:
        u = np.random.default_rng(seed).random((n, len(lo)))
    return a + u * (b - a)


def frames_of(q, K):
    T = hjcdik.target_transforms(q[None, :])[0]
    return T[:, :3, 3], np.stack([quat_from_R(T[k][:3, :3]) for k in range(K)])


def write_yaml_flat(path, y_batch, y_time, y_pos, y_ori):
    """Same flat parallel-list schema hjcd_ik_bench emits, so benchmark/_results_io.py can read it."""
    with open(path, "w") as y:
        y.write("Batch-Size:\n")
        for v in y_batch:
            y.write(f"- {int(v)}\n")
        y.write("IK-time(ms):\n")
        for v in y_time:
            y.write(f"- {v:.9f}\n")
        y.write("Pos-Error:\n")
        for v in y_pos:
            y.write(f"- {v:.9g}\n")
        y.write("Ori-Error:\n")
        for v in y_ori:
            y.write(f"- {v:.9g}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- flags shared with hjcd_ik_bench (same names, same meanings) ---------------------------
    ap.add_argument("--skip-grid-codegen", action="store_true",
                    help="Accepted for parity. Codegen is ALWAYS skipped here: a multi-target build "
                         "needs the --target flags (see scripts/dev/g1_check.sh), so this harness "
                         "never regenerates grid.cuh for you.")
    ap.add_argument("--urdf", type=str, default=str(ROOT / "csrc" / "urdf" / "g1_29dof_rev_1_0.urdf"),
                    help="Recorded only; the compiled robot is whatever grid.cuh was generated from.")
    ap.add_argument("--yaml-out", type=str, default="results.yml", help="YAML output file name.")
    ap.add_argument("--batches", type=_parse_batches, default=_parse_batches("1,10,100,1000,2000"),
                    help="Batch sizes = candidate restarts per target (comma/space separated).")
    ap.add_argument("--num-solutions", type=int, default=1,
                    help="Number of returned solutions per target (best-N candidates).")
    ap.add_argument("--num-targets", type=int, default=100, help="Number of random targets.")
    ap.add_argument("--csv-out", type=str, default="",
                    help="CSV summary: solver,Batch-Size,time_ms,pos_err_mm,ori_err_rad,"
                         "collision_free(%%)  -- same columns as hjcd_ik_bench.")
    ap.add_argument("--solver", type=str, default="hjcdik-mt",
                    help="Solver name to emit in the CSV.")
    ap.add_argument("--print-solutions", action="store_true", help="Print joint configs.")
    ap.add_argument("--seed", type=int, default=0)

    # --- multi-target-only ----------------------------------------------------------------------
    ap.add_argument("--masks", type=str, default="all_four",
                    help=f"Comma list from {sorted(MASKS)}, or raw ints (0b0011 / 3). One table per "
                         f"mask. Multi-target only.")
    ap.add_argument("--position-tol", type=float, default=1e-4, help="metres (solved criterion)")
    ap.add_argument("--orientation-tol", type=float, default=1e-3, help="radians (solved criterion)")
    ap.add_argument("--coarse-mode", default="auto", choices=["auto", "none", "multi_target"])
    ap.add_argument("--coarse-iters", type=int, default=120)
    ap.add_argument("--lm-iters", type=int, default=60)
    ap.add_argument("--seed-mode", choices=["random", "nearby"], default="random",
                    help="random restarts across the joint range, or warm starts near the solution")
    ap.add_argument("--nearby-sigma", type=float, default=0.15, help="rad, for --seed-mode nearby")
    ap.add_argument("--sampler", choices=["halton", "uniform"], default="halton",
                    help="TARGET sampler. halton = the scrambled Halton hjcd_ik_bench uses.")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="Inset the target range away from the joint limits, as a fraction of span. "
                         "0.0 (default) = FULL range, matching hjcd_ik_bench. Positive makes targets "
                         "easier and is NOT comparable to it.")
    ap.add_argument("--reps", type=int, default=3, help="Timed repetitions (median).")
    args = ap.parse_args()

    N, K = hjcdik.num_joints(), hjcdik.num_targets()
    if K < 2:
        raise SystemExit(f"this build has K={K} target frame(s) -- it is a single-target build.\n"
                         f"Use benchmark/hjcd_ik_bench.py, or regenerate for a multi-target robot.")
    lim = hjcdik.joint_limits()
    lo, hi = lim[:, 0], lim[:, 1]

    masks = [(m.strip(), MASKS.get(m.strip()) or int(m.strip(), 0))
             for m in args.masks.split(",") if m.strip()]

    range_note = "FULL joint range" if args.margin == 0 else "INSET -- NOT comparable to hjcd_ik_bench"
    print(f"robot: {N} joints, {K} target frames ({', '.join(NAMES[:K])})")
    print(f"targets: {args.num_targets}, sampler={args.sampler}, margin={args.margin:g} ({range_note})")
    print(f"seeds: {args.seed_mode}   coarse_mode={args.coarse_mode}   "
          f"tol: {args.position_tol*1000:g} mm / {args.orientation_tol:g} rad\n")

    target_qs = sample_target_configs(args.num_targets, lo, hi, args.sampler, args.margin, args.seed)
    rng = np.random.default_rng(args.seed + 1)

    y_batch, y_time, y_pos, y_ori = [], [], [], []
    csv_rows = []

    for mname, mask in masks:
        act = [k for k in range(K) if (mask >> k) & 1]
        print(f"=== mask {mname} ({mask:0{K}b}) ===")
        hdr = (f"{'Batch-Size':>11}{'IK-time(ms)':>13}{'Pos-Error(mm)':>15}{'Ori-Error':>12}"
               f"{'solved%':>9}{'pos_ok%':>9}{'ori_ok%':>9}")
        print(hdr)
        print("-" * len(hdr))

        for B in args.batches:
            solved = pos_ok = ori_ok = 0
            pos_all, ori_all, times = [], [], []

            for q_true in target_qs:
                pos, quat = frames_of(q_true, K)
                P, Q = np.repeat(pos[None], B, axis=0), np.repeat(quat[None], B, axis=0)
                if args.seed_mode == "nearby":
                    seeds = np.clip(q_true + rng.normal(scale=args.nearby_sigma, size=(B, N)), lo, hi)
                else:
                    seeds = rng.uniform(lo, hi, size=(B, N))
                m = np.full(B, mask, dtype=np.uint32)
                kw = dict(active_target_mask=m,
                          position_tol=args.position_tol, orientation_tol=args.orientation_tol,
                          coarse_mode=args.coarse_mode, coarse_iters=args.coarse_iters,
                          lm_iters=args.lm_iters)

                hjcdik.solve(seeds, P, Q, **kw)                       # warm-up (untimed)
                ts = []
                for _ in range(args.reps):
                    t0 = time.perf_counter()
                    out = hjcdik.solve(seeds, P, Q, **kw)
                    ts.append((time.perf_counter() - t0) * 1e3)
                dt = statistics.median(ts)

                pe = out["position_errors"][:, act].max(axis=1)       # worst ACTIVE target
                oe = out["orientation_errors"][:, act].max(axis=1)
                if bool(out["success"].any()):
                    solved += 1
                if bool((pe <= args.position_tol).any()):
                    pos_ok += 1
                if bool((oe <= args.orientation_tol).any()):
                    ori_ok += 1

                # returned "solutions" = the best num_solutions candidates, as generate_solutions does
                order = np.argsort(pe)[:max(1, args.num_solutions)]
                for i in order:
                    pos_all.append(pe[i] * 1000.0)
                    ori_all.append(oe[i])
                    times.append(dt)          # one call time, replicated per returned solution
                    y_batch.append(B)
                    y_time.append(dt)
                    y_pos.append(pe[i] * 1000.0)
                    y_ori.append(oe[i])
                if args.print_solutions:
                    print(f"    q = {np.round(out['joint_config'][order[0]], 4)}")

            n = args.num_targets
            row = dict(solver=args.solver, mask=mname, batch=int(B),
                       time_ms=float(np.mean(times)),
                       pos_err_mm=float(np.mean(pos_all)), ori_err_rad=float(np.mean(ori_all)),
                       solved_pct=100.0*solved/n, pos_ok_pct=100.0*pos_ok/n, ori_ok_pct=100.0*ori_ok/n)
            csv_rows.append(row)
            print(f"{B:>11}{row['time_ms']:>13.6f}{row['pos_err_mm']:>15.6e}"
                  f"{row['ori_err_rad']:>12.3e}{row['solved_pct']:>8.0f}%"
                  f"{row['pos_ok_pct']:>8.0f}%{row['ori_ok_pct']:>8.0f}%")
        print()

    if args.yaml_out:
        write_yaml_flat(args.yaml_out, y_batch, y_time, y_pos, y_ori)
        print(f"[OK] wrote {args.yaml_out} with {len(y_batch)} entries "
              f"({args.num_targets} targets x {len(args.batches)} batches x "
              f"{len(masks)} masks x {args.num_solutions} solutions each).")

    if args.csv_out:
        with open(args.csv_out, "w", newline="") as f:
            w = csv.writer(f)
            # hjcd_ik_bench's columns, plus the multi-target ones it has no concept of.
            w.writerow(["solver", "Batch-Size", "time_ms", "pos_err_mm", "ori_err_rad",
                        "collision_free(%)", "mask", "solved(%)", "pos_ok(%)", "ori_ok(%)"])
            for r in csv_rows:
                w.writerow([r["solver"], r["batch"], f"{r['time_ms']:.9f}",
                            f"{r['pos_err_mm']:.9g}", f"{r['ori_err_rad']:.9g}", "",
                            r["mask"], f"{r['solved_pct']:.1f}", f"{r['pos_ok_pct']:.1f}",
                            f"{r['ori_ok_pct']:.1f}"])
        print(f"[OK] wrote {args.csv_out}")


if __name__ == "__main__":
    main()
