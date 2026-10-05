# Audited precompiled A/B gate — October 5, 2026

Measured native source: pre-refinement `1d2cba2` versus audit-fixed `2302d22`, and the latter with
`HJCD_CC_STOP=0`. This is a same-machine regression gate, **not** a new comparison against the paper or
competitor solvers. The later upstream pin update is separate; do not relabel the recorded binary hashes.

## Independently verified

- 360 unique summary rows, 36,000 measured query attempts, six rotating rounds, 72 successful workers.
- Both endpoints: Release builds, Panda hand frame, foam geometry, Python 3.12.3, NumPy 2.5.3, CUDA 13.2.86.
- Automatic refinement precision: S=1 fp64; S=4 fp32. Same inputs and full error/collision validation.
- Recomputed analysis equals `analysis.json`; driver/helper/input/installed-binary hashes match the manifest.
- All matched per-round cells have identical solved, empty and returned counts across all three endpoints.
  Each endpoint solved 5,970/6,000 S=1 queries and 5,976/6,000 S=4 queries. No empty outputs; respectively
  6,000 and 24,000 configurations returned. These counts are not a universal accuracy/safety guarantee.

## Latency interpretation

Ranges below are across workload/batch cells of the median paired-round new/old latency ratio:

| Workload | Requested outputs | Ratio range | Interpretation |
| --- | ---: | ---: | --- |
| Open-world | 1 | 0.8505–0.9956 | Generally lower; B=8000 is approximately unchanged |
| Open-world | 4 | 0.9383–0.9489 | 5.1–6.2% lower |
| box_panda | 1 | 0.8253–0.9354 | 6.5–17.5% lower |
| box_panda | 4 | 0.7647–0.8732 | 12.7–23.5% lower |

All 20 final/reference cell medians are below one. Nineteen cells have every paired round below one.
Open-world S=1 B=8000 spans 0.9894–1.0055 across rounds: do not call that a separated improvement.
This gate supports no observed regression on this corpus; it cannot attribute all gains to one change.

The no-stop diagnostic is slightly faster on box_panda at equal measured quality: no-stop/final paired
median ratios 0.9679–0.9781 for S=1 and 0.9810–0.9918 for S=4. Keep collision-aware stopping enabled:
this easy scene does not exercise the hard-scene success benefits. No default was changed.

## Provenance and limitations

The timing agent reported an exclusive window on the RTX 5090 / Core Ultra 9 285K, performance CPU
governors, no clock/power changes, and no concurrent compute/compiles. It identified initial graphics
activity from the desktop app, minimized that window, and observed four consecutive 0%-GPU samples
before starting under the shared lock. The unchanged launcher checks foreign compute PIDs throughout.
That operational account is preserved in local `docs/open-tasks/timing_review_2026-10-05.md`; raw worker
logs remain in `docs/open-tasks/audit_gate_2026-10-04/timing-results-20261005/`.

The archived manifest/CSV/analysis here preserve the exact endpoint hashes and full per-cell counts.
No paper/baseline campaign ran in this window. Historical b10/b15 missing-query groups still require
recollection before republishing their rates. Later dependency/tool adoption needs its own validation;
an unchanged generated header body alone does not prove an identical compiled executable.
