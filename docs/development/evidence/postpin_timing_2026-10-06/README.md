# Post-pin confirmation gate — October 6, 2026 (reviewed by Claude, October 7)

Focused same-machine A/B confirming that the final dependency pins (GRiD `8dccbfa` + GLASS `9e57178`,
consumed by HJCD `121982d`) did not change quality or latency relative to the already-audited pre-pin build
(`2302d22`, measured in `audit_timing_2026-10-05/`). It is **not** a comparison against the paper's hardware
or competitor solvers, and it does not re-measure the October 5 refinement gains.

## Endpoints (frozen, provenance re-checked read-only on October 7 with `run.sh --check`)

| Label | Meaning | Commit | Header sha (build_info) | Binary sha |
| --- | --- | --- | --- | --- |
| `main` | pre-pin audit build (`docs/open-tasks/audit_gate_2026-10-04/branch`, own venv) | `2302d22` | `efc2c758…` | `a1f5a163…` |
| `branch` | post-pin primary repo (`docs/open-tasks/upstream_2026-10-05/.venv`, editable) | `121982d` | `602db063…` | `2c91ca0a…` |

Both Release, CUDA 13.2.86, Python 3.12.3, NumPy 2.5.3, `panda_hand_joint` frame, identical foam spheres,
hard collision mode, automatic refinement precision (S=1 fp64, S=4 fp32). Workload: four alternating rounds
(endpoint order rotates per round), S ∈ {1, 4}, open-world B ∈ {1, 2000, 8000, 32000} on the shared Halton
targets (`targets.json` sha `89d2d46e…`), `box_panda` B ∈ {100, 2000} (`tests/mb_problems.json` sha
`69a3eae8…`), 100 targets per cell. Launcher/driver/helper hashes in `SHA256SUMS` and `manifest.json` match
the tree at `121982d`. Installed binaries re-hashed October 7: unchanged.

## Independently verified (October 7)

- 96 unique CSV rows (48 per endpoint), 32 worker logs with 100 targets each, 9,600 measured attempts;
  no worker errors; `analysis.json` recomputed from `results.csv` and equal to the stored file to 1e-9.
- **Every matched per-round cell has identical solved / empty / returned counts.** Open-world S=1: 95/100
  at B=1 and 100/100 at B≥2000 (both endpoints, every round); S=4: 96/100 at B=1, else 100/100, 400
  configurations returned; `box_panda`: 100/100 and 100/400 everywhere; zero empty outputs. "Solved" means
  at least one returned candidate satisfies pos < 5 mm AND ori < 0.05 rad on the same candidate under the
  driver's independent FK, with reported errors cross-checked; in collision mode every returned candidate
  was also verified free under the `hjcd` sphere oracle (the driver raises otherwise).
- Launcher log: GPU at 0 % utilization, 29 °C, 6.5 W, idle clocks at start; the gate aborts on any foreign
  compute PID before and during every worker (2 s polling) and none appeared. Run wall time 64 s
  (04:12:39–04:13:42 UTC). Exit was clean (`analysis.json` is only written after `analyze()` succeeds).
  The coordinator-side account of CPU/graphics quiet was not available to the reviewer.

## Paired latency ratios (branch / main, per-round medians; 100 targets per cell)

| Workload | S | B | main median ms | branch median ms | paired ratio | round range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| open-world | 1 | 1 | 2.0067 | 2.0063 | 1.0001 | 0.9989–1.0005 |
| open-world | 1 | 2000 | 1.2610 | 1.2602 | 0.9946 | 0.9856–1.0382 |
| open-world | 1 | 8000 | 1.6822 | 1.6800 | 0.9988 | 0.9915–1.0066 |
| open-world | 1 | 32000 | 2.7604 | 2.7688 | 1.0046 | 0.9941–1.0078 |
| open-world | 4 | 1 | 1.4154 | 1.4153 | 0.9997 | 0.9992–1.0006 |
| open-world | 4 | 2000 | 1.3883 | 1.3879 | 1.0009 | 0.9977–1.0021 |
| open-world | 4 | 8000 | 2.4880 | 2.4891 | 1.0013 | 0.9970–1.0034 |
| open-world | 4 | 32000 | 6.8365 | 6.8428 | 1.0013 | 0.9993–1.0033 |
| box_panda | 1 | 100 | 2.0903 | 2.0932 | 1.0016 | 0.9997–1.0031 |
| box_panda | 1 | 2000 | 2.0894 | 2.0928 | 1.0016 | 0.9979–1.0048 |
| box_panda | 4 | 100 | 1.8667 | 1.8653 | 0.9990 | 0.9966–1.0040 |
| box_panda | 4 | 2000 | 1.9408 | 1.9423 | 1.0005 | 0.9973–1.0040 |

Verdict: **unchanged**. All twelve paired medians lie within 0.9946–1.0046; no cell is consistently
slower or faster across rounds. The one round outside ±1 % (open-world S=1 B=2000, one round at +3.8 %
while the cell's other rounds are at −1.4 … +0.2 %) is single-round noise, not a trend; it does not meet
the "median > 1.03 or consistently slower" review trigger. Ratios near one are "no change", not a win.
This confirms the pin update is performance-neutral on this corpus; it establishes nothing about other
hardware, the competitor baselines or the paper tables.

## Files

`results.csv`, `manifest.json`, `analysis.json`, `launcher.log`, `worker_logs/` (verbatim), `gate.json`,
`run.sh`, `SHA256SUMS` (the frozen launcher inputs). Original outputs remain untouched under
`docs/open-tasks/postpin_timing_2026-10-05/results-20261006T041238798290236Z/`.
