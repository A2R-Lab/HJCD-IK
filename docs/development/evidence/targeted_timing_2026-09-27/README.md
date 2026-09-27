# Frozen targeted-runtime evidence — 2026-09-27

This is an archival review bundle, **not a new supported benchmark launcher**.
The exact campaign harness and its CPU-only tests are preserved without changing
their machine-local staging assumptions. They are separate from the signed
correctness receipt: that receipt does not certify these timings or this harness.
No binaries, virtual environments, generated build trees or raw configuration
dumps are included here.

## Result and interpretation

See [the full ten-case table](summary.md) and [machine-readable results](summary.json).
All 60 workers completed in 167.48 seconds: 7,680 measured calls, all solved,
28,416 returned configurations, no environment-collision or joint-limit failures.
Maximum independent FK position error was 0.061544 mm, maximum orientation error
1.64555e-5 rad, and maximum returned-position/FK discrepancy 4.981e-7 m.
There were 183 telemetry samples, no observed foreign compute process, and at
most 6.518% aggregate CPU busy time (the rejection threshold was 20%).

Open controls stayed within 1% across paired rounds. Full-document repeated-scene
median latency fell about 14%; changing-scene latency fell about 93%. Compact
changing-scene latency fell about 29%. The `hard` and `both` policies agreed in
this pattern. The large full-document improvement reflects avoiding host JSON
reparsing/copying, **not a comparable acceleration of the GPU IK kernel**.

Pooled median/p95 are computed over 384 calls per endpoint/case; the percentage
range uses three separately paired round medians. Repetitions reuse 32 targets
or scenes; they are not independent new tasks or a statistical confidence interval.
The full-document repeat schedule includes a scene change every four calls, so
its p95 includes cache misses. Open one-output fp64 retains an approximately 9 ms
p95 despite its approximately 1.29 ms median. First-process calls were recorded
separately (about 179–201 ms in full-document cases), not included in warm latency.

The independent NumPy oracle selects the URDF-derived `hjcd` Panda geometry and
checks environment collisions only. It does not independently certify self
collision, physical deployment, all targets, other hardware, or the paper's
competitor/MMD results. Desktop graphics remained active, clocks were not locked,
and telemetry sampling cannot exclude interference between observations.

## Exact protocol and provenance

- [run.json](run.json) records endpoints, hashes, toolchain, settings, completion
  order and UTC timestamps. Its absolute paths describe the original machine.
- [run.py](run.py) and [run.sh](run.sh) are byte-for-byte copies of the executed
  harness/launcher. [test_harness.py](test_harness.py) and
  [test_orchestration.py](test_orchestration.py) are their 14 CPU-only tests.
- [panda.json](panda.json) preserves all 32 frozen open-world targets. The full
  scene input is already tracked as [tests/mb_problems.json](../../../../tests/mb_problems.json);
  its hash must match `run.json`. Compact JSON is derived inside the harness from
  the first 32 `box_panda` entries, preserving original indices and goal poses.
- [raw_artifact_sha256.txt](raw_artifact_sha256.txt) identifies the retained raw
  per-worker outputs, telemetry, harness and reports. Hashes identify artifacts;
  they are not signatures or a substitute for inspecting the raw data.

Both endpoints use Release, native `sm_120`, CUDA 13.2.86, GCC 13.3, Python 3.12.3,
NumPy 2.5.1, no diagnostic CUDA flags, RTX 5090 and driver 615.71.09. Baseline is
`209be95`, latest is `516b988` (implementation unchanged through `cc71978`). Each
snapshot retains its dependency pins. This is an end-to-end snapshot comparison,
not isolation of one individual source change.

There are ten cases: two open controls (one output/fp64 and four outputs/fp32),
then `hard`/`both` × full/compact JSON × repeated/changing scenes (four outputs,
fp32). Every case uses batch 2000, W=1, position/orientation LM tolerances 1e-8,
and statistics disabled. Three rounds alternate A/B and B/A by case and round.
Each endpoint/case/round has a fresh worker. After one separately recorded cold
call, warmup is at least four calls and 0.3 seconds. Each worker measures 128
synchronous Python calls: four per target, grouped or interleaved by schedule.
Validation and serialization happen after all measured calls.

Acceptance requires reviewing both quality and latency. The frozen harness stops
on worker/guard failures and checks structural validity, joint limits, environment
collisions and FK/returned-position agreement. **Its exit code alone does not
enforce solved rate or a latency threshold**; inspect the solved/zero-count fields
and paired variability before accepting a future campaign.

## Reproduction and retained local artifacts

Original working directory was `/tmp`. The following paths are relative to the
repository and deliberately remain ignored:

```text
docs/open-tasks/
  targeted-timing-2026-09-27/{run.py,run.sh,test_harness.py,test_orchestration.py}
  targeted-timing-2026-09-27/results-20260927T043021361833Z/
  timing-prebuild-2026-09-26/runtime/bin/python
  timing-prebuild-2026-09-26/current-panda/site/
  timing-prebuild-2026-09-26/inputs/{panda.json,mb_problems.json}
  premerge-2026-09-27/timing-site/
  premerge-2026-09-27/receipt-verification/
```

The last directory is the frozen `cc71978` oracle checkout, including Panda URDF
and foam geometry. Restore the archived harness/tests into the original layout
to use its relative paths; do not execute `run.sh` directly from this archive.
Restore inputs from this bundle and the tracked scene JSON. Original installed
modules must match the hashes in `run.json`. For a new build/machine, reproduce
the endpoint commits, pinned dependencies and release settings in isolated
checkouts, explicitly rebaseline build hashes in a **new** harness, and record
new provenance. Recompilation is not expected to recreate identical binary bytes.

With the original staging restored, host-only readiness is:

```bash
bash docs/open-tasks/targeted-timing-2026-09-27/run.sh --check
PYTHONSAFEPATH=1 OPENBLAS_NUM_THREADS=1 \
  docs/open-tasks/timing-prebuild-2026-09-26/runtime/bin/python \
  -m unittest discover -s docs/open-tasks/targeted-timing-2026-09-27 -p 'test_*.py' -v
```

Only in an explicitly allocated quiet GPU/CPU window, replace `--check` with
`--run`. The default budget is 30 minutes; this completed campaign took under
three. Output goes to a fresh timestamped directory beside the active harness.
Ctrl-C/SIGTERM stops its own worker and saves partial evidence; restarting creates
a new campaign, never an in-place resume or splice of separate windows. Its lock
prevents duplicate HJCD campaigns only; coordinate with other projects separately.
Do not repeat timing merely to update documentation.
