# Full paper-protocol campaign with competitor baselines — RTX 5090, 2026-10-07 (quiet window)

The corrected, complete campaign on the shipped code: `main` at `f4dc1b5` (solver unchanged since the pushed
`dd4c350`; GRiD `8dccbfa`, GLASS `9e57178`), every baseline installed in the isolated staging clone
(`docs/open-tasks/paper-2026-10-03/hjcd/.venv`): PyRoki @ main (collision-aware, foam spheres through its own
`RobotCollision.from_sphere_decomposition`), cuRobo v2 @ main (cuda-core backend, bundled `franka.yml` spheres
re-targeted to the evaluation frame), IKFlow 0.0.8, tracikpy for the MMD ground truth. Separate from the signed
correctness receipt; one machine, one window; not the camera-ready paper's numbers (RTX 4060, cuRobo v0.7).

## Window account

Launched 15:52:03 UTC by `window_campaign.sh --run` (preflight: staging clone clean at the expected commit,
baseline imports, GPU idle; then the shared `/tmp/a2rlab-timing.lock` and three 0 %-utilization samples).
`run_paper_experiments.sh` with `HJCD_REGEN=1 RUN_FETCH=1 RUN_DOF=1 RUN_MMD=1 RUN_HARD=1`, 100 targets /
100 problems per cell, batches 1/10/100/1000/2000 (DoF at 1000): **exit 0, 1604 s, no solver stage skipped,
no retry taken**. `monitor.log` (3 s samples, 495 rows) shows no compute process other than the run's own
`.venv/bin/python` workers; load average 0.02 at launch; nobody else on the box (user-announced window).
The staging build was restored to the default `panda_grasptarget_hand` at the end (`b2707112…`).
Two harness fixes made the night before are part of this run: the wrapper no longer pre-creates the
fail-closed `OUT_DIR`, and `LD_LIBRARY_PATH` carries the venv's `nvidia/cu13/lib` so cuRobo's NVRTC finds
`libnvrtc-builtins.so.13.0` (the frame check now reports PASS for all three baselines).

## Headline numbers (time = mean per-call wall time, ms; position error mm; 100 queries per cell)

Open-world Panda (`panda_hand`, shared Halton targets):

| B | HJCD-IK | cuRobo v2 | PyRoki | IKFlow |
| ---: | --- | --- | --- | --- |
| 1 | 3.30 / 2.80 (95/100 within 5 mm) | 8.52 / 1.8e-2 | 3.30 / 371 | 2.60 / 16.0 |
| 10 | 2.04 / 5.8e-5 | 5.90 / 1.7e-2 | 5.62 / 5.2 | 2.48 / 3.3 |
| 100 | **1.65** / 1.8e-5 | 5.93 / 1.8e-2 | 6.02 / 0.39 | 2.76 / 1.7 |
| 1000 | **1.62** / 4.2e-5 | 6.14 / 2.1e-2 | 7.04 / 1.1e-4 | 5.10 / 0.77 |
| 2000 | **1.65** / 2.5e-5 | 6.35 / 2.1e-2 | 7.19 / 1.2e-4 | 6.15 / 0.70 |

Fetch open-world (`ee_link`): HJCD-IK 0.81–1.10 ms (100 % within 5 mm at every B), cuRobo 1.81–2.10, PyRoki
2.8–6.0, IKFlow 2.5–6.2 (30–160 mm error). DoF scaling at B = 1000 (7/12/18/24): HJCD-IK 1.76 / 1.87 / 2.31 /
2.85 ms; cuRobo 2.38 (median; the mean 8.74 contains one 591 ms kernel JIT on the first problem) / 2.19 /
2.45 / 2.67; PyRoki 6.8 / 8.2 / 10.7 / 12.7. MMD (lower = better): HJCD-IK **0.098**, PyRoki 0.124, cuRobo
0.143, IKFlow 0.211.

`box_panda`, B = 2000: paper protocol (TCP, cylinder-snapped) HJCD-IK 1.82 ms, cuRobo 2.45, PyRoki 29.1
(collision-aware); dataset protocol (`panda_hand`, goal as posed) HJCD-IK 2.10, cuRobo 2.44, PyRoki 30.3.

Dataset protocol, all ten sets at B = 2000 — mean ms, and success (pose < 5 mm/0.05 rad AND free under the
visual-mesh judge, 1 mm) from `results/table_oracles_collfree_hand.md`:

| set | HJCD-IK ms / % | cuRobo v2 ms / % | PyRoki ms / % |
| --- | --- | --- | --- |
| bookshelf_small | 2.10 / 100 | 2.49 / 98 | 29.1 / 100 |
| bookshelf_tall | 2.07 / 100 | 2.46 / 100 | 30.6 / 100 |
| bookshelf_thin | 1.82 / 100 | 2.45 / 100 | 32.9 / 100 |
| box | 2.10 / 100 | 2.44 / 100 | 30.3 / 100 |
| box_flipped | — / 100 | — / 100 | — / 100 |
| cage | 1.86 / 100 | 2.46 / 100 | 26.9 / 100 |
| table_pick | 2.64 / 100 | 2.43 / 99 | 32.5 / 100 |
| table_under_pick | 2.34 / 100 | 2.44 / 98 | 30.9 / 100 |
| kitchen | 1.83 / 99 | 2.44 / 99 | 37.7 / 99 |
| table_bars | 3.22 / 100 | 2.47 / 100 | 27.1 / 100 |

(`box_flipped` latency is in `results/table_collfree_hand_box_panda_flipped.md`.) At B = 100 HJCD-IK is
98–100 % on every set; cuRobo 97–100 %; PyRoki 95–100 %. Under the convex-hull judge the same configurations
score lower for everyone (cage 77–87 %, table_bars 33–78 %), as documented in the fairness evidence.
At B = 1 (one candidate) HJCD-IK's collision-free success is 5–95 % by set — the single-candidate regime
is not where this solver is meant to run.

## Files

`results/` = the complete `OUT_DIR` (per-solver CSV/YML, per-set markdown tables, Pareto PNGs, the stored
configurations `configs_collfree_{paper,hand}/*.jsonl` — 15 000 records with empties, and the multi-judge
`table_oracles_collfree_*.md`); `run.log` (campaign stdout/stderr, NULs stripped), `monitor.log`,
`run_paper_full.sh` + `window_campaign.sh` as run.
