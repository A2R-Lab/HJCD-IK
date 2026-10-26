# Boundary-fix timing gate, 2026-10-26

Two complete, unchanged quiet-window passes finished successfully. Quality counts are
unchanged, but small repeated latency increases remain. **Performance acceptance is
pending; this is not a performance-neutral verdict or merge approval.**

## Provenance and protocol

The user authorized this CPU/GPU window. Initial external GPU check showed an RTX 5090
at 0% utilization with no compute applications; a process snapshot showed no busy CPU
worker. The launcher rejected foreign GPU PIDs before and during workers. Both runs
exited zero. No competitors, failed workers, discarded rounds or endpoint edits were
reported. This is not continuous CPU telemetry: user allocation plus the startup check
established CPU quietness. No owned job remains running, and the final GPU compute list
was empty. Other application processes were left alone.

- Working branch: `review/general-joints-closeout`, source/evidence checkpoint `1d795f9`.
- Reference source `9f3e017`, prepared hand-frame endpoint `fefecb8596cef1079404fc82046c6ac6dbd8f3f9`.
- Candidate implementation `70bbb70`, prepared endpoint `a7c0dc5cb44a1aecbd34d4e91ab5b427fdbf7cf2`.
- Subsequent migration/evidence commits do not change solver/build/harness implementation.
- Separate packaged Panda, runtime Panda and 24-DoF binaries/caches were precompiled.
  Installed binary/header hashes, runtime versions and workload identities are in every manifest.
- Five legs: API, native, runtime Panda, 24-DoF and bounded-midpoint warm-start API.
  Each run has four alternating paired rounds, 100 targets/cell, S=1/fp64 and S=4/fp32
  under auto precision. Panda open B=1/2000/8000/32000; collision B=100/2000;
  warm open B=1/2000. High-DoF is open only. Models, tolerances and targets are unchanged.
- Each pass includes two alternating rounds of cold runtime builds/cache hits for Panda
  and 24-DoF. This includes codegen, compilation and publication, not wheel installation.
- Raw output directories: `/tmp/hjcd-boundary-timing-20261026-a` and
  `/tmp/hjcd-boundary-timing-20261026-b`. Build caches/binaries remain there locally.
  The compact tracked `runs/` packet retains manifests, CSVs, analyses, worker logs and
  build summaries. The local launcher is
  `docs/open-tasks/boundary-gate-2026-10-25/run.sh --quiet-window NEW_OUTPUT_DIRECTORY`.

The second pass was run because the first showed small slowdowns, as required by the
predeclared [gate protocol](../../p2b_gate.md). Neither workload nor acceptance criteria
changed. Both runs are retained. These are same-machine regression measurements, not a
rerun of the historical paper or physical-robot validation.

## Quality and accounting

The analysis revalidated all expected cells, query denominators, header identities,
precision settings and stored analyses. Across both runs: **83,200 measured queries**
(41,600 per pass), 82,704 pose successes, zero empty outputs and 208,000 returned solutions.
Solved, empty and returned counts match exactly in **every paired cell of every round**.
Each endpoint therefore has 41,600 queries and 41,352 pose successes across the two passes.

This does not mean every target succeeded. At the intentionally minimal open B=1:
cold Panda solves 95/100 with S=1 and 96/100 with S=4; warm Panda solves 98/100 for both.
Those same misses occur on both endpoints in every round. All other measured cells
have 100/100 pose success, including the measured collision and high-DoF cells.
These are aggregate matched counts, not claims of identical candidate configurations
or a guarantee on arbitrary targets. Independent-FK checks ran in each worker.

## Latency results

Values below are medians of eight paired per-round median-latency ratios, candidate
relative to reference. Positive means slower. Absolute deltas are medians of the eight
paired differences; they are not obtained by multiplying separately pooled medians.
`paired_results.txt` and `summary.json` retain all 52 cells, both pass medians, every
round, baseline/candidate milliseconds and absolute deltas. No formal significance or
universal performance claim is made from these finite samples.

| Path, open S=4 | B=2000 | B=8000 | B=32000 |
|---|---:|---:|---:|
| API | +0.671%, +9.48 us | +1.110%, +27.91 us | +0.843%, +58.27 us |
| Native | +0.773%, +10.74 us | +0.874%, +21.87 us | +1.046%, +71.92 us |
| Runtime Panda | +0.697%, +9.85 us | +0.945%, +23.83 us | +1.011%, +69.86 us |

All nine cells above are slower in **8/8 paired rounds**, so the gate does not dismiss
them as a one-run fluctuation. API open S=1/B=32000 is also slower in 8/8 (+0.508%,
+14.33 us), though native/runtime S=1 results vary and do not support a universal S=1
slowdown claim. Most remaining cold Panda collision differences are smaller; runtime
S=4/B=100 collision is +0.210% (+3.25 us), also positive in 8/8 rounds.

Warm-start pooled changes range from +0.286% to +0.805%, about +5.18 to +11.75 us.
Seven of eight warm cells are slower in all eight rounds. Warm S=1/open/B=2000 is the
exception: its two pass medians are +0.746% and -0.032%, with 5/8 slower rounds; its
pooled +0.581% is not evidence of a consistently slower individual cell.

24-DoF pooled changes range from -0.009% to +0.224%; **every** cell's eight-round range
crosses zero. Several have mostly-positive rounds, so this is weaker/noisier evidence,
not proof of exact performance equivalence. No blanket speedup is observed or claimed.

## Compile and reuse cost

Four observations per endpoint/robot (two in each complete pass):

| Robot | Reference cold median (range), s | Candidate cold median (range), s | Reference / candidate cache-hit median, s |
|---|---:|---:|---:|
| Panda | 14.902 (14.897–14.937) | 15.233 (15.130–15.289) | 0.199 / 0.198 |
| 24-DoF | 55.163 (54.175–55.226) | 55.055 (54.946–55.179) | 0.426 / 0.425 |

Panda cold-build median is about +0.332 s (+2.2%) and all four candidate observations
exceed all four reference observations. Report that cost rather than calling compilation
unchanged. High-DoF ranges overlap; cache reuse remains about 0.2/0.43 s. Small sample
counts and order/toolchain effects limit compile-speed conclusions.

## Interpretation and next step

Correctness remains closed, with **383 passed** in the signed suite; its receipt was
reverified after timing on clean `1d795f9`. An additional verification attempt while
this evidence directory was untracked correctly failed the clean-tree policy (retained
as `receipt_dirty_check.log`); signature, fingerprint and ancestry still passed. Reverify
again on the committed evidence checkpoint. The boundary source diff changes host seed preparation and
candidate ranking, not the device solver math. The no-warm path now uses a categorical
pair score, and warm calls add a joint-limit device-to-host copy plus finite checks.
These are plausible attribution targets, **not measured root causes**; compiler layout
can also affect behavior. No tuning, profiler run or source change was made during timing.

Recommended next step, requiring a separate implementation decision: isolate host
selection cost first, preserve the scalar no-warm comparison where semantics permit,
and investigate obtaining effective warm limits without an extra per-call transfer.
Retain categorical convergence priority, bounded coarse-precision seed semantics,
nonfinite exclusion, pool preference and duplicate policy. Do not weaken fixes or
thresholds to recover a favorable number. Any proposed change needs the existing
regressions, full signed proof, re-staged final endpoint and a new paired timing gate.
Alternatively, the user can explicitly accept these measured costs. The previous
accepted P2b tradeoff is not authorization for this additional cost.

No push, PR, merge, dependency update, paper-table edit or website update occurred.
The box is released; no further timing is scheduled. Publication remains a separate
approval after the performance decision.

## Reproduce the review without GPU work

From the repository root, using the existing development environment:

```bash
.venv/bin/python docs/development/evidence/boundary_timing_2026-10-26/analyze.py \
  --repo . \
  --run-a docs/development/evidence/boundary_timing_2026-10-26/runs/a \
  --run-b docs/development/evidence/boundary_timing_2026-10-26/runs/b \
  --summary /tmp/hjcd-boundary-review.json
```

`SHA256SUMS` covers the compact preserved packet. The generated summary is descriptive;
the original CSVs and manifests remain the source evidence. Absolute paths in manifests
refer to the original local staging checkout and are not portable binaries.
