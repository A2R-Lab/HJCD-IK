# General-joint timing gate, 2026-10-25

## Conclusion

Two complete quiet-window passes compared pre-P2b `3f5ec5a` with the final general-joint
solver source `190915e`. Across 70,400 timed queries, every matched round/cell had identical
solved, empty and returned counts. No large latency or build-cost regression was observed.
There are small repeatable slowdowns, so this is not a claim of unchanged or universally
improved performance, and not unconditional merge approval.

The user explicitly authorized this window. Both launchers exited successfully, without a
foreign-GPU-process interruption. Measurements ran sequentially; no other work was launched
by this agent during timing. CPU activity was inspected before starting, not continuously
logged. GPU process monitoring does not establish a completely noise-free system.

## Protocol and provenance

- RTX 5090, driver 615.71.09, nvcc 13.2.86, Python 3.12.3, NumPy 2.5.1.
- Root harness at `7c3b70b`; no package/kernel changes since candidate source `190915e`.
- Reference prepared hand-frame clone: `9d3a53c36debe5bc48a03247e014cffd1565028a`.
- Candidate prepared hand-frame clone: `d40487623c7ded4877fda51066f9db65e32cecdf`.
- Both passes use the same frozen inputs, installed binaries and separate endpoint caches.
  Manifests record hashes, runtime versions, entry points and generated-header identities.
- Each pass: four alternating rounds, S=1/4, 100 targets/cell. Panda open-world
  B=1/2000/8000/32000; box_panda hard collision B=100/2000. High-DoF is open-world only.
- API, native entry, prepared runtime Panda, and prepared runtime 24-DoF are separate legs.
  Native entry timing is still end-to-end native solve time, not isolated device time.
- Two alternating cold-build/cache-hit rounds per pass, Panda and 24-DoF, fresh build caches.
  Four observations per endpoint/robot after both passes.
- Acceptance and interruption rules: [P2b protocol](../../p2b_gate.md). No criteria changed
  after seeing timed results. The second full pass was collected because the first showed
  small borderline differences. Both are preserved, not selectively pooled by outcome.

The launcher was `scripts/perf/run_joint_gate.py --quiet-window`, with stage
`docs/open-tasks/p2b-gate-2026-10-25`. Raw local output directories are
`docs/open-tasks/p2b-timing-results-2026-10-25-quiet` and the same name with `-repeat`.
The complete precompiled gate plus repeat finished within the allocated 30–45 minute window.

## Quality

Both endpoints solved 34,984 of 35,200 attempted queries under the timing protocol's
same-candidate 5 mm / 0.05 rad criterion. There were no empty outputs. The misses are
identical: Panda B=1 solves 95/100 at S=1 and 96/100 at S=4, in all three Panda entry-point
legs and every round. Larger Panda batches and all tested 24-DoF cells solve 100/100.
A returned result is not automatically accurate; these B=1 misses are retained in the denominator.

No matched cell differs in solved, empty or returned count, even before aggregation.
This shows quality parity for these workloads, not identical candidate configurations or
a guarantee that arbitrary targets/robots always solve. The mixed-axis/prismatic correctness
matrix is separately recorded in [general-joint correctness](../general_joints_2026-10-25/README.md).

## Latency

Each value below is the range across workload cells of the median of eight paired
candidate/reference ratios, expressed as percentage change. Positive means slower.
Per-round ratios and absolute deltas are in `summary.json`; the source CSVs retain
median, mean, p95 and query accounting.

| Entry point | Median-latency change across cells |
|---|---:|
| Packaged Panda API | -0.130% to +0.412% |
| Packaged Panda native | -0.095% to +0.276% |
| Runtime-built Panda API | -0.097% to +0.609% |
| Runtime-built 24-DoF API | -0.096% to +0.121% |

Review items, not dismissed as noise:

- Panda open-world B=1, S=1 is slower in all eight API pairs: +0.412%, or +8.35 µs
  around 2.03 ms. Runtime-built Panda is also slower in all eight: +0.609%, +12.39 µs.
  Native is +0.246%, +4.97 µs, slower in seven of eight pairs.
- Packaged API box_panda B=2000, S=4 is +0.239%, +3.89 µs, slower in all eight pairs.
  The same cell is not consistently slower for runtime-built Panda; native is +0.082%.
- Runtime Panda B=2000, S=1 has a pooled +0.562% median, but paired ratios cross one
  with appreciably greater round spread. This is distinct from the persistent B=1 signal.
- Most remaining cells have mixed-sign paired differences. All 24-DoF cells cross one.

For means, the pooled cell changes across all legs lie between -0.220% and +0.209%.
For p95, they lie between -0.246% and +0.812%. These are descriptive ranges of the measured
cells, not confidence intervals or a newly chosen acceptance threshold.

## Build and reuse costs

Values are medians of four cold/cache-hit observations per endpoint. Cold cost includes
runtime code generation, compilation and publication, not wheel construction or installation.

| Robot and operation | Reference | Candidate |
|---|---:|---:|
| Panda cold build | 14.817 s | 14.917 s |
| 24-DoF cold build | 54.601 s | 54.935 s |
| Panda cache hit | 0.1993 s | 0.1960 s |
| 24-DoF cache hit | 0.4211 s | 0.4218 s |

Panda cold-build ranges are 14.786–14.924 s versus 14.794–15.006 s.
24-DoF ranges are 54.352–55.064 s versus 54.145–55.126 s. Small median increases
overlap the observed ranges; there is no evidence here of a large compile-time regression.
Four samples are insufficient to establish a small compile-speed improvement or regression.

Runtime build metadata leaves `cuda_compiler_version` empty in these build-cost JSONs;
the environment's nvcc version above was checked directly. Full build logs and caches remain
in the original ignored output directories. Do not imply that the empty field establishes a version.

## Read-only investigation

After both timing passes, `cuobjdump --dump-sass` comparisons found identical instruction
lines for 24 of 26 packaged/runtime Panda function entries and 22 of 24 high-DoF entries.
The differing entries are the fp32/fp64 LM kernels. Their disassemblies include changed
call offsets and instruction layout; this is not evidence that every changed byte represents
additional work. Panda LM resource usage is unchanged:

| LM scalar | Registers | Stack bytes | Shared bytes |
|---|---:|---:|---:|
| fp32 | 245 | 64 | 1968 |
| fp64 | 176 | 176 | 2896 |

The source retains the all-+Z compile-time specialization but refactors Jacobian-column
construction through an inlined helper. Compiled-code differences make code generation a
plausible investigation target, not a proven cause of the measured end-to-end delta.
No code was changed to chase the sub-percent result. A causal attribution needs a controlled
helper-layout A/B and/or device/host profiling, not this disassembly observation alone.

## Disposition and follow-up

Quality acceptance is satisfied on the declared timing workloads. No substantial high-DoF,
latency or compilation regression was found, but the small repeatable Panda slowdowns remain
a review item. Discuss accepting the measured sub-percent cost versus doing a focused
compiler-layout/host-device attribution experiment before merge. Do not silently label it
performance-neutral. There is no paper-hardware or new-robot speed claim here.

No push or merge was performed. This report changes evidence only; the signed 317-test
correctness receipt still covers the measured implementation. Separate open items remain
GRiD's all-prismatic generator bug, Z1 collision-model fidelity and Fetch distribution licensing.

## Files

`run1/` and `run2/` preserve each leg's manifests, results CSV and original analysis JSON,
plus the compile sample JSONs. Worker logs are also copied locally (repository ignore rules
may exclude logs from Git); the original ignored output directories retain every log/cache.
`summary.json` combines all eight pairs without removing any round and verifies equality
of solved, empty and returned counts for each pair. Its ratios are recomputable from the CSVs.
