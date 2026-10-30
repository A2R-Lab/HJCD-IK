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
- Latency: requires a new paired quiet-window gate against the `9f3e017` reference; not run here.
