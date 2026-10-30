# Host selection cost, 2026-10-30

Attribution and removal of the boundary-wave host costs measured in
[boundary_timing_2026-10-26](../boundary_timing_2026-10-26/README.md).

## Attribution

- The two staged endpoints (`9f3e017` reference, `70bbb70` candidate) have byte-identical
  device code: `cuobjdump -sass` of both packaged Panda modules matches line for line once
  address comments are stripped (one sm_120 cubin each, 237,581 lines). The costs are host code.
- Warm-start cells: the only warm-only addition was a per-call `cudaMemcpyFromSymbol` of the
  joint limits, a synchronous 112-byte device-to-host copy. Warm open B=1 was +5.4 to +7.4 us in
  every paired round while cold B=1 was unchanged, which matches that copy.
- No-warm cells: the candidate ranks with a pair key computed per comparison inside `std::sort`
  over the refined pool (Krep = top_k x 16, top_k scaled by S/2). `comparator_bench.cpp` times
  the three forms on one core of the development machine (Intel Core Ultra 9 285K, `c++ -O3
  -std=c++17`, 300 shuffled pools per size, medians):

| Krep (gate cell) | reference scalar | pair key per comparison | precomputed keys |
|---|---:|---:|---:|
| 512 (collision B=100, S=4) | 26.9 us | +11.7 us | -5.2 us |
| 768 (open B=2000, S=4) | 35.6 us | +14.4 us | -8.5 us |
| 2304 (open B=8000, S=4) | 118.2 us | +48.4 us | -24.1 us |
| 8448 (open B=32000, S=4) | 518.3 us | +219.9 us | -115.1 us |

  `comparator_bench_O3.txt` is that run (60 % converged candidates); `comparator_bench_O3_conv90.txt`
  repeats it with 90 % converged. The gate measured +9.5, +24 to 28 and +58 to 72 us for the three
  open cells: same direction and ordering, smaller because real pools carry many near-equal scores
  from the sixteen perturbed copies of each seed. The reference sort itself is a visible share of a
  solve (about 0.5 ms of 6.9 ms at B=32000), which the precomputed form reduces.

## Change (`csrc/kernel/hjcd_kernel.cu`, `csrc/kernel/selection_key.h`)

- `init_joint_limits_from_grid` keeps the exact `packed` vector it uploads to `c_joint_limits`
  in a translation-unit static; the warm-start clamp reads that copy. Same bytes, same module,
  no per-call device copy or implicit synchronisation.
- `select_from` computes each candidate's key once and sorts positions by key
  (`hjcd::sorted_positions`). `std::sort` decides only through the comparator and the key is pure,
  so the permutation is the one the per-comparison sort produced; `tests/native/test_selection.cpp`
  checks this on 200 random pools with ties, nonfinite scores and both warm modes.
  Selection semantics are unchanged: categorical convergence priority, clamped coarse-precision
  seeds and their proximity metric, nonfinite exclusion, refined-pool preference, coarse top-up
  and the duplicate tolerance.

## Validation

- Native selection test and the focused GPU regressions (`test_warm_ranking.py`,
  `test_world_validation.py`, `test_boundary_contracts.py`: 39 passed) on the rebuilt module.
- Full signed suite and receipt: see the receipt commit following the source commit.
- Latency: two paired quiet-window passes on 2026-10-30, below.

## Paired timing gate, 2026-10-30

Two complete, unchanged quiet-window passes. Quality counts match exactly and the measured host
costs are gone; the candidate is faster than the reference in every cell with a pool to sort.

### Provenance and protocol

The user announced the window with every agent quiet. Before the first pass the GPU had no
compute applications at 0 % utilisation and the one-minute load average was 0.01; before the
second pass the compute list was again empty. The launcher rejects foreign GPU PIDs before and
during workers. Both passes exited zero with no discarded rounds or endpoint edits. The first
pass showed no slowdown, so the protocol did not require a second; it was run anyway to keep the
two-pass form of the 2026-10-26 packet. During round 2 of the API leg of the second pass the
analysis script for the first pass ran for a few seconds on one core, because the wait loop had
picked up the wrong process id; that is the only known contention and it is disclosed rather
than excluded. No owned job remained running afterwards and the final GPU compute list was empty.

- Branch `review/general-joints-closeout`; evidence commit follows receipt `36003b3`.
- Reference source `9f3e017`, prepared hand-frame endpoint `fefecb8596cef1079404fc82046c6ac6dbd8f3f9`
  (the 2026-10-26 reference, reused in place).
- Candidate source `577debb`, prepared endpoint `67570e6a1fd7d1007e7c40e508fbc13d35245536`.
  Both endpoints carry the same generated-header hash `fa3241db...`; the installed binaries differ.
- Same five legs, workload, targets, high-DoF seed, rounds and precisions as 2026-10-26
  (`docs/open-tasks/boundary-gate-2026-10-30/`, assembled with `assemble_joint_gate.py` and
  rehearsed correctness-only beforehand). Per pass: four alternating paired rounds, 100
  targets per cell, S=1/fp64 and S=4/fp32; Panda open B=1/2000/8000/32000, collision B=100/2000,
  warm open B=1/2000 and collision B=100/2000, 24-DoF open B=1/2000/8000/32000; two cold
  runtime build and cache-hit rounds for Panda and 24-DoF.
- Raw output `/tmp/hjcd-boundary-timing-20261030-{a,b}`; the tracked `runs/` packet keeps
  manifests, CSVs, analyses, worker logs and build summaries without the build trees.

### Quality and accounting

`analyze.py` revalidated every expected cell, denominator, header identity and stored analysis.
Across both passes: 83,200 measured queries, 82,704 pose successes, zero empty outputs,
208,000 returned solutions; solved, empty and returned counts match in every paired cell of
every round and no cell raised the quality review flag. As on 2026-10-26, open B=1 cold Panda
solves 95/100 (S=1) and 96/100 (S=4) and warm Panda 98/100, identically on both endpoints in
every round; every other cell is 100/100.

### Latency

Medians of the eight paired per-round ratios, candidate relative to reference, negative is
faster; deltas are medians of the eight paired differences. `paired_results.txt` and
`summary.json` hold all 52 cells, both pass medians, every round and the absolute values.

| Path, open S=4 | B=2000 | B=8000 | B=32000 |
|---|---:|---:|---:|
| API | -0.925 %, -13.1 us | -1.856 %, -46.9 us | -2.809 %, -194 us |
| Native | -1.034 %, -14.4 us | -1.657 %, -41.3 us | -2.828 %, -195 us |
| Runtime Panda | -0.814 %, -11.5 us | -1.721 %, -43.4 us | -2.702 %, -186 us |

All nine are faster in 8/8 rounds, as are the S=1 cells at B=8000 (-1.2 to -1.4 %) and B=32000
(-3.0 to -3.2 %, -86 to -90 us), the collision cells at S=4 (-0.6 to -1.0 %, -9 to -16 us) and
the 24-DoF cells at B=8000 and B=32000 (-0.4 to -1.6 %, -16 to -159 us). The reference sort
was a visible share of a large-batch solve, which is why the gain exceeds the 2026-10-26 cost.

Warm starts: open S=4 B=2000 -3.319 % (-48 us), collision S=4 B=100 -1.425 % (-22 us) and
B=2000 -1.936 % (-32 us), open S=1 B=2000 -1.361 % (-18 us), all 8/8. Warm B=1, which carried
the +5 to +7 us copy on 2026-10-26, is now -0.053 % and -0.015 % with rounds split 4/8 and 3/8:
the per-call copy is gone and nothing measurable replaced it.

Six cells have a positive pooled median: Panda open B=1 (API S=1 +0.007 %, API S=4 +0.012 %,
runtime S=4 +0.011 %), API S=1 B=2000 (+0.005 %), 24-DoF B=1 (S=1 +0.021 %, +1.05 us; S=4
+0.009 %). Every one is at or below 0.02 % and 1.1 us with rounds split between 4/8 and 6/8;
these pools are one or sixteen candidates, where the sort is not on the path. They are noise,
not a cost, and no cell is slower by a repeatable margin.

### Compile and reuse

Four observations per endpoint and robot:

| Robot | Reference cold median (range), s | Candidate cold median (range), s | Reference / candidate cache hit, s |
|---|---:|---:|---:|
| Panda | 15.063 (14.928 to 15.237) | 14.450 (14.332 to 14.593) | 0.201 / 0.196 |
| 24-DoF | 55.267 (55.041 to 55.404) | 55.045 (54.753 to 55.327) | 0.426 / 0.429 |

Panda cold builds are about 0.6 s faster on the candidate, all four below all four reference
observations; the 2026-10-26 packet reported +0.33 s the other way for the boundary wave. The
host edit is small and the swing is not attributed to it; four samples and toolchain cache
state limit any compile-speed conclusion. 24-DoF ranges overlap; cache reuse is unchanged.

### Verdict

The boundary-wave host costs are removed, not accepted. Correctness is unchanged under the
383-test receipt `36003b3` and the 200-pool native equivalence test. The branch is ready for
the pull request to leave draft; merging remains the user's call.

### Reproduce the review without GPU work

```bash
.venv/bin/python docs/development/evidence/host_selection_2026-10-30/analyze.py \
  --repo . \
  --runs docs/development/evidence/host_selection_2026-10-30/runs/a \
         docs/development/evidence/host_selection_2026-10-30/runs/b \
  --summary /tmp/hjcd-host-selection-review.json
```

`SHA256SUMS` covers the tracked packet. Absolute paths in the manifests refer to the local
staging checkouts and are not portable.
