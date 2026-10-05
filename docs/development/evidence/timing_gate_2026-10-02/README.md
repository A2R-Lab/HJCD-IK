# Timing gate + HJCD-only paper rerun — 2026-10-02

**Erratum (2026-10-04):** the neutral A/B driver passed positional argument eight as `False`,
which forced `refine_fp64=0`. Thus its S=1 rows measure fp32, not default fp64. The raw CSVs
are preserved; the A/B cannot establish default-S1 performance. See `../../timing_gate.md`.

Archival evidence for the de-vendored GLASS tidy-up that landed on `main` as the fast-forward
`3f14ae6..a720514`. It is separate from the signed correctness receipt (`gpu-proof.json`,
recorded at `d3e6a75`): the receipt certifies test outcomes, not these timings. No binaries,
virtual environments or build trees are included; the two staged clones used as endpoints were
deleted after the run.

## Machine

RTX 5090 (sm_120), driver 615.71.09, CUDA UMD 13.4, nvcc 13.2.86, Python 3.12.3. Desktop graphics
active, clocks not locked. Before each run `nvidia-smi` showed no other compute process and the
CPU was ~95% idle; `paper_monitor.log` samples both every 3 s during the paper run (the only
compute processes seen are the run's own `.venv/bin/python`; CPU peaks of ~65% are its own
regenerate-and-rebuild steps, not the timed benchmarks).

## 1. main-vs-branch A/B (`ab_results*.csv`)

Driver: `scripts/perf/timing_driver.py` (tracked at `713e3d2`), run by `run_ab.sh` with one
isolated clone + venv per endpoint, 6 rounds, endpoint order alternating per round (ABBA) to
cancel boost-clock drift. Per cell: 5 warm-up calls, then per-call wall time of
`generate_solutions` over the 64 shared targets in `ab_targets.json` (open-world) or the 100
`box_panda` goal poses from `tests/mb_problems.json` (collision-free, `hard`). `ab_analyze.py`
reports the median over rounds of the per-round ratio branch/main of the per-round medians.
Both endpoints produce different candidates run to run (the coarse early-stop races across
blocks), so only latency is compared here; correctness equivalence was established separately
with a statistical A/B (same returned counts in 16 configuration groups, error distributions
within run-to-run noise).

Endpoints: main `3f14ae6`; branch `033e3d5` (before fix) and `32e134f` (after fix). The two
branch binaries differ only in `upload_replicated_target`.

**Before the fix** (`ab_results_before_fix.csv`), branch/main median ratio:

| leg | S | B | main ms | branch ms | ratio | round range |
|---|---|---|---|---|---|---|
| open | 1 | 1 | 1.5596 | 1.5489 | −0.67% | −0.70%..−0.59% |
| open | 1 | 10 | 1.5139 | 1.5002 | −0.91% | −0.96%..−0.85% |
| open | 1 | 100 | 1.3809 | 1.3741 | −0.46% | −1.00%..+0.12% |
| open | 1 | 1000 | 1.3731 | 1.3693 | −0.29% | −0.54%..+0.09% |
| open | 1 | 2000 | 1.4108 | 1.4099 | −0.10% | −0.37%..+0.16% |
| open | 1 | 8000 | 1.6894 | 1.7147 | **+1.59%** | +1.06%..+1.71% |
| open | 1 | 32000 | 4.5381 | 4.6260 | **+1.90%** | +0.88%..+2.58% |
| open | 4 | 1 | 1.5568 | 1.5461 | −0.68% | −0.81%..−0.19% |
| open | 4 | 10 | 1.5099 | 1.4975 | −0.83% | −1.06%..−0.73% |
| open | 4 | 100 | 1.4108 | 1.4019 | −0.57% | −0.72%..−0.26% |
| open | 4 | 1000 | 1.3929 | 1.3956 | +0.22% | −0.28%..+0.58% |
| open | 4 | 2000 | 1.4623 | 1.4682 | +0.43% | +0.32%..+0.60% |
| open | 4 | 8000 | 2.6502 | 2.6809 | **+1.14%** | −0.19%..+1.67% |
| open | 4 | 32000 | 7.3771 | 7.4684 | **+1.33%** | +0.67%..+2.35% |
| coll | 1 | 1000 | 2.0595 | 2.0549 | −0.16% | −0.36%..−0.04% |
| coll | 1 | 2000 | 2.3002 | 2.3029 | +0.12% | −0.19%..+0.41% |
| coll | 4 | 1000 | 2.1221 | 2.1248 | −0.01% | −0.11%..+0.51% |
| coll | 4 | 2000 | 2.4275 | 2.4360 | +0.36% | +0.16%..+0.57% |

Cause of the large-batch regression: the tidy-up replicated the 7-vector target with a single
32-thread block whose 7 active threads wrote all B rows serially (was one block per row), so the
cost grew linearly with B on one thread (~2.7 ns/row ≈ 90 µs at B=32000).

**After the fix** (`32e134f`: one thread per output element, `ab_results.csv`):

| leg | S | B | main ms | branch ms | ratio | round range |
|---|---|---|---|---|---|---|
| open | 1 | 1 | 1.5594 | 1.5491 | −0.67% | −1.05%..−0.53% |
| open | 1 | 10 | 1.5135 | 1.5007 | −0.86% | −0.99%..−0.76% |
| open | 1 | 100 | 1.3816 | 1.3782 | −0.24% | −0.41%..−0.14% |
| open | 1 | 1000 | 1.3713 | 1.3645 | −0.43% | −0.57%..−0.33% |
| open | 1 | 2000 | 1.4076 | 1.4016 | −0.42% | −0.53%..−0.31% |
| open | 1 | 8000 | 1.6899 | 1.6865 | −0.17% | −0.43%..+0.01% |
| open | 1 | 32000 | 4.5157 | 4.5032 | −0.27% | −0.63%..+0.01% |
| open | 4 | 1 | 1.5570 | 1.5449 | −0.78% | −0.92%..−0.75% |
| open | 4 | 10 | 1.5098 | 1.4956 | −0.96% | −1.11%..−0.85% |
| open | 4 | 100 | 1.4100 | 1.4022 | −0.52% | −0.91%..−0.36% |
| open | 4 | 1000 | 1.3936 | 1.3897 | −0.25% | −0.47%..−0.06% |
| open | 4 | 2000 | 1.4633 | 1.4579 | −0.27% | −0.88%..−0.12% |
| open | 4 | 8000 | 2.6603 | 2.6466 | −0.43% | −1.50%..+0.27% |
| open | 4 | 32000 | 7.3763 | 7.3431 | −0.53% | −0.70%..−0.10% |
| coll | 1 | 1000 | 2.0596 | 2.0537 | −0.32% | −0.43%..−0.10% |
| coll | 1 | 2000 | 2.3009 | 2.2984 | −0.06% | −0.21%..+0.18% |
| coll | 4 | 1000 | 2.1243 | 2.1193 | −0.27% | −0.41%..+0.05% |
| coll | 4 | 2000 | 2.4275 | 2.4243 | −0.06% | −0.24%..+0.04% |

Verdict: the landed code is 0.06–0.96% faster than the previous `main` in every cell, within the
1% bar everywhere. The small gain at small B is consistent with fewer kernel launches and
allocations in the host orchestration (no coarse→RT cast when RT is float, no `g_winner` reset).
Both `S=1` and `S=4` ran forced fp32 (see erratum above). `min_ms` columns are
single-call outliers of the early-stop race and are not a signal.

## 2. HJCD-only paper protocol (`paper_results/`, `paper_run.log`)

`scripts/bench/run_paper_experiments.sh` at `d3e6a75` with `HJCD_REGEN=1 RUN_FETCH=1 RUN_DOF=1
RUN_MMD=1` and all competitor baselines skipped (PyRoki, cuRobo, IKFlow and TRAC-IK are not
installed on this machine), run from an isolated clone by `run_paper.sh`. 273 s end to end,
most of it the seven regenerate+rebuild steps. `NUM_TARGETS=100`, batches 1,10,100,1000,2000,
`num_solutions=1`, time = per-call wall time after one warm-up call per (target, batch).

| Table | B | time ms | pos mm | ori rad |
|---|---|---|---|---|
| I Panda open-world | 1 / 10 / 100 / 1000 / 2000 | 3.336 / 2.074 / 1.667 / 1.711 / 1.743 | 2.797 / 5.8e-5 / 2.3e-5 / 4.8e-5 / 5.0e-5 | 9.1e-3 / 3.2e-7 / 7.0e-9 / 5.1e-9 / 5.4e-9 |
| I Fetch open-world | 1 / 10 / 100 / 1000 / 2000 | 1.105 / 0.946 / 0.834 / 0.814 / 0.860 | 5.3e-3 / 1.3e-6 / 9.0e-7 / 7.8e-7 / 8.9e-7 | 1.2e-5 / 1.9e-9 / 1.3e-9 / 1.3e-9 / 1.5e-9 |
| II Panda collision-free, box_panda (hard; 100% collision-free at every B) | 1 / 10 / 100 / 1000 / 2000 | 5.490 / 2.137 / 1.908 / 2.026 / 2.275 | 14.59 / 2.6e-5 / 5.1e-6 / 6.1e-5 / 8.8e-6 | 2.0e-2 / 1.3e-8 / 2.7e-9 / 3.1e-8 / 4.6e-9 |
| III DoF 7 / 12 / 18 / 24 at B=1000 | 1000 | 1.846 / 1.946 / 2.415 / 3.007 | 1.4e-3 / 2.6e-6 / 2.2e-6 / 1.6e-6 | 6.8e-7 / 2.2e-9 / 9.4e-10 / 5.8e-10 |

The B=1 rows are single-candidate solves that often do not converge; their averages are
dominated by misses and should be read against the paper's own B=1 row, not as a precision
figure. Table I Panda uses the `panda_hand` frame (shared open-world targets), Table II the
`panda_grasptarget_hand` collision build, Table III `panda_hand` on the DoF-extended URDFs.
Table IV was dumped (100 targets × 50 configurations) but not scored: no TRAC-IK ground truth
and no competitor dumps. The dump is not archived here.

The run exposed a defect that predates the tidy-up: `benchmark/hjcd_ik_bench.py` passed
`problem_idx=-1` in open-world solves, which the API has rejected since the audit-hardening
merge, so the open-world harness was non-functional on `main` from `3f14ae6` until `d3e6a75`.
The benchmark loop and the MMD dump path now pass `0` (ignored in open-world solves).

## Files

`run_ab.sh`, `run_paper.sh` and `ab_analyze.py` are the exact scripts used; the first two refer
to the staging directory layout (`<stage>/{main,branch}` clones, each with its own `.venv`) that
was deleted after the run. `ab_run*.log` are the driver's console output; `paper_monitor.log`
is the 3 s `nvidia-smi`/`top` sample during the paper run.
