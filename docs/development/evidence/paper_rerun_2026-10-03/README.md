# Paper-protocol rerun with competitor baselines — 2026-10-03

Archival evidence for the full `scripts/bench/run_paper_experiments.sh` campaign on `main` at `2fc1316`
(HJCD-IK kernel unchanged since `32e134f`), with all competitor baselines installed. It is separate from the
signed correctness receipt. Numbers are from one quiet window on one machine; they are not the camera-ready
paper's numbers (RTX 4060, CUDA 12.5, cuRobo v0.7) and should not be mixed with them.


> **Correction, same day (afternoon).** The morning campaign's cuRobo Table II columns were produced by a cuRobo
> robot built from our mesh-less `panda.urdf` via `RobotBuilder`, which has **no collision spheres**, so cuRobo's
> "collision-free" IK was in fact unconstrained in every Table II row (both protocols). This affected the paper's
> harness design as well, not cuRobo. Fixed in `486b547` (cuRobo's bundled, sphere-equipped `franka.yml` is
> re-targeted to the requested frame); cuRobo's Table II passes were re-run (`results/collfree_*_curobo.csv`,
> 121 s, quiet GPU) and every table below is regenerated from the corrected files. The superseded CSVs are kept in
> `results_superseded_curobo_nospheres/`. A second bug fixed at the same time: the per-set table glob merged
> `box_panda_flipped` into the `box_panda` dataset-protocol table (`a69cf2d`-era script); regenerated from
> explicit file lists. PyRoki's collision-free path is plain IK with no obstacle term (as in the paper), so its
> `cf` column is purely the oracle filtering its pose solutions.

## Machine and run

RTX 5090 (sm_120), driver 615.71.09, nvcc 13.2.86, Python 3.12.3. Isolated clone with its own venv:
hjcdik (editable), PyRoki (jax 0.11.2 + cuda12 plugin, jaxls, pyroki @ main), cuRobo v2 (`NVlabs/curobo@main`,
cuda-core 1.2.1 backend, torch 2.14.1+cu130), IKFlow 0.0.8 with the paper's Panda model
`panda__full__lp191_5.25m` and the Fetch model loaded offline, tracikpy (ROS-free build) for the MMD ground
truth. `run_paper_full.sh` ran `HJCD_REGEN=1 RUN_FETCH=1 RUN_DOF=1 RUN_MMD=1 RUN_HARD=1`: 874 s end to end,
exit 0, no solver stage skipped, no CUDA-graph retry taken. `monitor.log` (3 s samples) shows no compute
process other than the run's own `.venv/bin/python` and CPU peaks only during the eight regenerate+rebuild
steps. The informational EE-frame check (`check_ee_frames.py`) reported `curobo SKIP` on its first kinematics
kernel JIT in this process; the same probe passes when run on its own (4/4) and cuRobo's solves ran normally.

Protocol: 100 Halton open-world targets per robot (`panda_hand` frame; Fetch `ee_link`), 100 problems per
MotionBenchMaker set, batch/seed sizes 1, 10, 100, 1000, 2000, one solution per call, per-call wall time
after warm-up. HJCD reports `n=1` (best candidate) per call; the baselines report their own per-call metric.
Every solver's collision-free column is the shared NumPy sphere oracle (`benchmark/panda_collision.py`,
`hjcd` geometry = the URDF-derived model HJCD compiles). See the caveat at the end.

A first attempt (`dry_runs/attempt1_aborted_run.log`) aborted at minute 2 when cuRobo's Fetch run at 2000
seeds failed CUDA-graph capture (`cudaErrorStreamCaptureInvalidated`); fixed in `2fc1316` by freeing the solver
between seed counts (as the Table II loop already did) with a no-graph retry as backstop, and by guarding every
baseline call so a solver failure cannot abort the campaign. The rerun above is the complete one.

## Table I — open-world, Panda (`panda_hand`, 100 shared Halton targets)


| Batch | curobo Time(ms) | curobo Pos(mm) | curobo Ori(rad) | hjcdik Time(ms) | hjcdik Pos(mm) | hjcdik Ori(rad) | ikflow Time(ms) | ikflow Pos(mm) | ikflow Ori(rad) | pyroki Time(ms) | pyroki Pos(mm) | pyroki Ori(rad) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8.547 | 0.01823 | 5.914e-05 | 3.338 | 2.797 | 0.00906 | 2.694 | 13.34 | 0.1321 | 3.259 | 370.7 | 0.298 |
| 10 | 5.969 | 0.01729 | 6.227e-05 | 2.076 | 5.772e-05 | 3.249e-07 | 2.572 | 3.701 | 0.04959 | 5.403 | 5.216 | 0.004334 |
| 100 | 6.019 | 0.01783 | 5.541e-05 | 1.665 | 2.276e-05 | 6.919e-09 | 2.856 | 1.493 | 0.06374 | 5.634 | 0.3903 | 2.451e-04 |
| 1000 | 6.246 | 0.0209 | 1.243e-04 | 1.718 | 4.799e-05 | 5.475e-09 | 5.138 | 0.732 | 0.0439 | 6.863 | 1.080e-04 | 2.171e-07 |
| 2000 | 6.433 | 0.02089 | 1.242e-04 | 1.745 | 4.987e-05 | 5.798e-09 | 6.19 | 0.6529 | 0.04486 | 6.866 | 1.132e-04 | 2.269e-07 |

_solvers: curobo: n=100/batch, hjcdik: n=1/batch, ikflow: n=100/batch, pyroki: n=100/batch_

## Table I — open-world, Fetch (`ee_link`)


| Batch | curobo Time(ms) | curobo Pos(mm) | curobo Ori(rad) | hjcdik Time(ms) | hjcdik Pos(mm) | hjcdik Ori(rad) | ikflow Time(ms) | ikflow Pos(mm) | ikflow Ori(rad) | pyroki Time(ms) | pyroki Pos(mm) | pyroki Ori(rad) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.845 | 2.571e-05 | 2.028e-08 | 1.106 | 0.005296 | 1.155e-05 | 2.687 | 137 | 0.4807 | 3.063 | 410 | 0.7618 |
| 10 | 1.847 | 2.571e-05 | 2.028e-08 | 0.9463 | 1.261e-06 | 1.883e-09 | 2.573 | 60.9 | 0.276 | 5.16 | 3.271e-05 | 2.856e-08 |
| 100 | 1.935 | 9.542e-06 | 1.348e-08 | 0.8372 | 9.162e-07 | 1.325e-09 | 2.833 | 43.37 | 0.3205 | 5.148 | 2.819e-05 | 2.855e-08 |
| 1000 | 2.092 | 1.682e-06 | 1.095e-08 | 0.8156 | 9.848e-07 | 1.486e-09 | 5.137 | 32.24 | 0.3668 | 5.971 | 2.994e-05 | 2.855e-08 |
| 2000 | 2.219 | 1.608e-06 | 1.042e-08 | 0.8593 | 1.018e-06 | 1.415e-09 | 6.191 | 31.12 | 0.3452 | 6.072 | 2.920e-05 | 2.895e-08 |

_solvers: curobo: n=100/batch, hjcdik: n=1/batch, ikflow: n=100/batch, pyroki: n=100/batch_

## Table II — collision-free, box_panda, paper protocol

Target = closest cylinder's xy + goal_pose z/orientation, solved for the TCP (HJCD `panda_grasptarget_hand`,
cuRobo `panda_grasptarget`, PyRoki `panda_hand_tcp` at 103.4 mm vs our 105 mm). Continuity row for the
published Table II.


| Batch | curobo Time(ms) | curobo Pos(mm) | curobo Ori(rad) | hjcdik Time(ms) | hjcdik Pos(mm) | hjcdik Ori(rad) | pyroki Time(ms) | pyroki Pos(mm) | pyroki Ori(rad) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 9.079 | 0.3238 | 0.001226 | 5.488 | 14.59 | 0.02032 | 3.326 | 499.4 | 0.3599 |
| 10 | 2.142 | 5.934e-05 | 6.582e-08 | 2.118 | 2.571e-05 | 1.284e-08 | 5.742 | 1.402e-04 | 1.985e-07 |
| 100 | 2.143 | 3.790e-05 | 4.555e-08 | 1.884 | 5.002e-06 | 2.627e-09 | 5.909 | 1.421e-04 | 2.093e-07 |
| 1000 | 2.342 | 1.742e-05 | 3.426e-08 | 2.002 | 2.176e-05 | 1.249e-08 | 7.108 | 1.266e-04 | 1.935e-07 |
| 2000 | 2.472 | 1.615e-05 | 3.293e-08 | 2.258 | 9.886e-06 | 5.345e-09 | 7.22 | 1.202e-04 | 2.033e-07 |

_solvers: curobo: n=100/batch, hjcdik: n=1/batch, pyroki: n=100/batch_

## Table II — collision-free, dataset protocol (`panda_hand`, goal_pose as posed), all 8 sets

MotionBenchMaker goals are `panda_hand` poses (FK of the dataset's `goal_ik` lands on them within 0.01–5 mm;
105 mm off at the TCP) and this is the only protocol valid on every set (cage has no cylinders; the flipped
box goal is 38 cm from any cylinder). box_panda:


| Batch | curobo Time(ms) | curobo Pos(mm) | curobo Ori(rad) | hjcdik Time(ms) | hjcdik Pos(mm) | hjcdik Ori(rad) | pyroki Time(ms) | pyroki Pos(mm) | pyroki Ori(rad) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8.549 | 0.1052 | 1.343e-04 | 6.485 | 22.15 | 0.03437 | 3.363 | 522.7 | 0.2789 |
| 10 | 2.131 | 0.02121 | 1.208e-05 | 2.427 | 5.263e-06 | 7.833e-10 | 5.851 | 0.4027 | 2.374e-04 |
| 100 | 2.203 | 0.004937 | 1.628e-07 | 2.226 | 4.971e-06 | 7.924e-10 | 6.584 | 1.172e-04 | 1.943e-07 |
| 1000 | 2.377 | 0.004717 | 3.935e-08 | 2.298 | 4.808e-06 | 7.800e-10 | 7.567 | 1.129e-04 | 1.963e-07 |
| 2000 | 2.513 | 0.002099 | 1.461e-07 | 2.553 | 4.702e-06 | 8.112e-10 | 7.73 | 1.143e-04 | 1.831e-07 |

_solvers: curobo: n=100/batch, hjcdik: n=1/batch, pyroki: n=100/batch_

Per-set success. For the baselines `succ` is their own pose success (<5 mm, <0.05 rad), `cf` the shared
oracle on the returned configuration, `both` = succ AND cf; times are medians. For HJCD, `accurate` uses the
same thresholds and `strict` is <1 mm, <1e-3 rad; HJCD's `hard` filter returns only candidates its compiled
model deems collision-free, so its oracle column is 100 by construction (same geometry).

```
Baselines, dataset protocol (panda_hand, goal_pose), 100 problems/set; cuRobo WITH its collision spheres (corrected 10-03 pm).
succ = pose success (<5 mm, <0.05 rad); cf = hjcd sphere oracle on the returned configuration; both = succ AND cf; median ms
set                          B |   cuRobo succ/cf/both%     ms |   PyRoki succ/cf/both%     ms
bookshelf_small_panda      100 |     100/ 96/ 96 (n=100)   2.18 |     100/ 82/ 82 (n=100)   5.76 |
bookshelf_small_panda     1000 |     100/ 94/ 94 (n=100)   2.29 |     100/ 82/ 82 (n=100)   6.88 |
bookshelf_small_panda     2000 |     100/ 96/ 96 (n=100)   2.42 |     100/ 88/ 88 (n=100)   7.20 |
bookshelf_tall_panda       100 |     100/ 99/ 99 (n=100)   2.19 |     100/ 97/ 97 (n=100)   5.66 |
bookshelf_tall_panda      1000 |     100/ 98/ 98 (n=100)   2.29 |     100/ 95/ 95 (n=100)   6.77 |
bookshelf_tall_panda      2000 |     100/ 99/ 99 (n=100)   2.42 |     100/ 98/ 98 (n=100)   6.89 |
bookshelf_thin_panda       100 |     100/100/100 (n=100)   2.16 |     100/100/100 (n=100)   5.84 |
bookshelf_thin_panda      1000 |     100/100/100 (n=100)   2.39 |     100/100/100 (n=100)   7.11 |
bookshelf_thin_panda      2000 |     100/100/100 (n=100)   2.53 |     100/100/100 (n=100)   7.35 |
box_panda                  100 |     100/100/100 (n=100)   2.20 |     100/100/100 (n=100)   6.56 |
box_panda                 1000 |     100/100/100 (n=100)   2.37 |     100/100/100 (n=100)   7.54 |
box_panda                 2000 |     100/100/100 (n=100)   2.51 |     100/ 98/ 98 (n=100)   7.67 |
box_panda_flipped          100 |     100/100/100 (n=100)   2.23 |     100/100/100 (n=100)   6.13 |
box_panda_flipped         1000 |     100/100/100 (n=100)   2.35 |     100/ 99/ 99 (n=100)   7.16 |
box_panda_flipped         2000 |     100/100/100 (n=100)   2.47 |     100/ 99/ 99 (n=100)   7.61 |
cage_panda                 100 |     100/ 99/ 99 (n=100)   2.15 |     100/ 92/ 92 (n=100)   6.59 |
cage_panda                1000 |     100/100/100 (n=100)   2.33 |     100/ 86/ 86 (n=100)   7.28 |
cage_panda                2000 |     100/ 99/ 99 (n=100)   2.46 |     100/ 83/ 83 (n=100)   7.16 |
table_pick_panda           100 |     100/ 97/ 97 (n=100)   2.12 |     100/ 49/ 49 (n=100)   5.53 |
table_pick_panda          1000 |     100/ 94/ 94 (n=100)   2.30 |     100/ 43/ 43 (n=100)   6.74 |
table_pick_panda          2000 |     100/ 93/ 93 (n=100)   2.43 |     100/ 45/ 45 (n=100)   6.86 |
table_under_pick_panda     100 |     100/ 91/ 91 (n=100)   2.11 |     100/ 44/ 44 (n=100)   5.82 |
table_under_pick_panda    1000 |     100/ 89/ 89 (n=100)   2.29 |     100/ 40/ 40 (n=100)   7.06 |
table_under_pick_panda    2000 |     100/ 90/ 90 (n=100)   2.42 |     100/ 41/ 41 (n=100)   7.13 |
```

```
HJCD dataset protocol (panda_hand build, goal_pose, hard): per set, 100 problems, S=1; accurate = pos<5 mm & ori<0.05 rad (baseline thresholds) / strict = pos<1 mm & ori<1e-3; oracle-free of returned configs
set                          B accurate%  strict% oracle_free% median ms
bookshelf_small_panda      100        98       94          100      1.86
bookshelf_small_panda     1000       100       99          100      1.99
bookshelf_small_panda     2000       100       99          100      2.14
bookshelf_tall_panda       100       100      100          100      2.01
bookshelf_tall_panda      1000       100      100          100      2.16
bookshelf_tall_panda      2000       100      100          100      2.45
bookshelf_thin_panda       100       100      100          100      1.98
bookshelf_thin_panda      1000       100      100          100      2.18
bookshelf_thin_panda      2000       100      100          100      2.52
box_panda                  100       100      100          100      2.23
box_panda                 1000       100      100          100      2.27
box_panda                 2000       100      100          100      2.52
box_panda_flipped          100       100      100          100      1.61
box_panda_flipped         1000       100      100          100      1.75
box_panda_flipped         2000       100      100          100      2.08
cage_panda                 100        99       99          100      1.87
cage_panda                1000        90       88          100      1.97
cage_panda                2000        97       93          100      2.25
table_pick_panda           100        92       89          100      2.13
table_pick_panda          1000        91       85          100      2.24
table_pick_panda          2000        94       87          100      2.57
table_under_pick_panda     100        97       91          100      2.01
table_under_pick_panda    1000        94       88          100      2.11
table_under_pick_panda    2000        95       90          100      2.45
```

## Table III — DoF scalability (open-world, B = 1000, `panda_hand`)

| DoF | cuRobo ms | cuRobo pos mm | HJCD ms | HJCD pos mm | PyRoki ms | PyRoki pos mm |
| --- | --- | --- | --- | --- | --- | --- |
| 7 | 2.024 | 2.6e-4 | 1.795 | 1.5e-3 | 7.045 | 2.5e-4 |
| 12 | 2.172 | 2.3e-5 | 1.893 | 2.9e-6 | 8.799 | 2.6e-4 |
| 18 | 2.493 | 2.3e-5 | 2.393 | 2.3e-6 | 10.35 | 4.8e-4 |
| 24 | 2.683 | 2.8e-5 | 2.992 | 1.5e-6 | 12.97 | 7.2e-4 |

(Per-DoF files: `table_dof{7,12,18,24}.md`.)

## Table IV — MMD (open-world Panda, 50 best of 2000, TRAC-IK ground truth)


| Metric | hjcdik | pyroki | curobo | ikflow |
| --- | --- | --- | --- | --- |
| MMD ↓ | 0.10290 | 0.12813 | 0.14360 | 0.22169 |
| MMD² ↓ | 0.01059 | 0.01642 | 0.02062 | 0.04915 |

_canonical IMQ MMD (beta=0.5, scales=0.2,0.5,1,2,5), per-pose then averaged; n_targets per solver: hjcdik=100, pyroki=100, curobo=100, ikflow=100_

## Multi-oracle collision study (`oracle_study/`)

Every solver's returned configuration (dataset protocol, B = 100 and 2000, 100 problems per set, 4800
configurations) was stored (`--configs-out` / `--configs_out`, added `3490d28`) and re-scored offline with
`benchmark/score_collision_oracles.py` under three judges: the `hjcd` spheres HJCD compiles, the paper's 65 mm
`paper` spheres, and the `panda_description` collision **meshes** checked with FCL (fingers open, penetration
tolerance 1 mm and 5 mm). `cf` means pose reached AND the oracle agrees the configuration is clear.

## Pose-success AND collision-free (%), per oracle

| set | solver | B | n | pose ok | pose & hjcd | pose & paper | pose & mesh<=1mm | pose & mesh<=5mm | all oracles |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bookshelf_small_panda | curobo | 100 | 100 | 100 | 96 | 96 | 94 | 94 | 93 |
| bookshelf_small_panda | curobo | 2000 | 100 | 100 | 96 | 96 | 93 | 93 | 91 |
| bookshelf_small_panda | hjcdik | 100 | 100 | 98 | 98 | 98 | 92 | 93 | 92 |
| bookshelf_small_panda | hjcdik | 2000 | 100 | 100 | 100 | 100 | 94 | 94 | 94 |
| bookshelf_small_panda | pyroki | 100 | 100 | 100 | 82 | 82 | 83 | 83 | 82 |
| bookshelf_small_panda | pyroki | 2000 | 100 | 100 | 88 | 88 | 89 | 90 | 87 |
| bookshelf_tall_panda | curobo | 100 | 100 | 100 | 99 | 99 | 97 | 97 | 97 |
| bookshelf_tall_panda | curobo | 2000 | 100 | 100 | 99 | 99 | 98 | 99 | 98 |
| bookshelf_tall_panda | hjcdik | 100 | 100 | 100 | 100 | 100 | 98 | 99 | 98 |
| bookshelf_tall_panda | hjcdik | 2000 | 100 | 100 | 100 | 100 | 98 | 99 | 98 |
| bookshelf_tall_panda | pyroki | 100 | 100 | 100 | 97 | 97 | 97 | 97 | 97 |
| bookshelf_tall_panda | pyroki | 2000 | 100 | 100 | 98 | 98 | 98 | 99 | 98 |
| bookshelf_thin_panda | curobo | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| bookshelf_thin_panda | curobo | 2000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| bookshelf_thin_panda | hjcdik | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| bookshelf_thin_panda | hjcdik | 2000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| bookshelf_thin_panda | pyroki | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| bookshelf_thin_panda | pyroki | 2000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| box_panda | curobo | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| box_panda | curobo | 2000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| box_panda | hjcdik | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| box_panda | hjcdik | 2000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| box_panda | pyroki | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| box_panda | pyroki | 2000 | 100 | 100 | 98 | 98 | 98 | 99 | 98 |
| box_panda_flipped | curobo | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| box_panda_flipped | curobo | 2000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| box_panda_flipped | hjcdik | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| box_panda_flipped | hjcdik | 2000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| box_panda_flipped | pyroki | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| box_panda_flipped | pyroki | 2000 | 100 | 100 | 99 | 99 | 100 | 100 | 99 |
| cage_panda | curobo | 100 | 100 | 100 | 99 | 99 | 83 | 90 | 83 |
| cage_panda | curobo | 2000 | 100 | 100 | 99 | 99 | 74 | 83 | 74 |
| cage_panda | hjcdik | 100 | 100 | 99 | 99 | 99 | 82 | 88 | 82 |
| cage_panda | hjcdik | 2000 | 100 | 97 | 97 | 97 | 77 | 84 | 77 |
| cage_panda | pyroki | 100 | 100 | 100 | 92 | 92 | 84 | 87 | 84 |
| cage_panda | pyroki | 2000 | 100 | 100 | 83 | 83 | 73 | 77 | 73 |
| table_pick_panda | curobo | 100 | 100 | 100 | 97 | 97 | 95 | 96 | 93 |
| table_pick_panda | curobo | 2000 | 100 | 100 | 93 | 93 | 95 | 97 | 90 |
| table_pick_panda | hjcdik | 100 | 100 | 92 | 92 | 92 | 89 | 89 | 89 |
| table_pick_panda | hjcdik | 2000 | 100 | 95 | 95 | 95 | 90 | 93 | 90 |
| table_pick_panda | pyroki | 100 | 100 | 100 | 50 | 50 | 50 | 54 | 48 |
| table_pick_panda | pyroki | 2000 | 100 | 100 | 43 | 43 | 44 | 46 | 42 |
| table_under_pick_panda | curobo | 100 | 100 | 100 | 91 | 91 | 95 | 95 | 87 |
| table_under_pick_panda | curobo | 2000 | 100 | 100 | 90 | 90 | 94 | 97 | 86 |
| table_under_pick_panda | hjcdik | 100 | 100 | 98 | 98 | 98 | 97 | 97 | 97 |
| table_under_pick_panda | hjcdik | 2000 | 100 | 95 | 95 | 95 | 95 | 95 | 95 |
| table_under_pick_panda | pyroki | 100 | 100 | 100 | 44 | 44 | 45 | 49 | 43 |
| table_under_pick_panda | pyroki | 2000 | 100 | 100 | 39 | 39 | 40 | 43 | 38 |

_pose ok = pos < 5 mm and ori < 0.05 rad on the solver's own report; oracles ignore the base link, check environment obstacles only, touching permitted; `all oracles` = every column agrees free._

Takeaways: the two sphere models agree to the problem; the mesh oracle is stricter in the cage (HJCD 97 -> 77,
cuRobo 99 -> 74: the hand's spheres under-cover the meshes against the bars) and marginally more lenient under
the table. With collision spheres active, cuRobo and HJCD are within a few points of each other on every set
under every oracle; PyRoki (unconstrained IK) is the outlier on the table sets. `curobo_nospheres.jsonl` is the
superseded unconstrained collection for reference.

## Reading the results

- HJCD reproduces its own numbers: Panda open-world 1.67–1.75 ms (B=100–2000), Fetch 0.82–0.95 ms,
  box collision-free 1.9–2.6 ms, lowest MMD. B=1 rows are single-candidate solves: 5/100 Panda targets miss
  entirely, which is what makes the B=1 mean 2.8 mm; at B>=10 all 100 are sub-micron.
- **cuRobo v2 is a far stronger baseline than the v0.7 in the paper**: with collision spheres 2.1–2.5 ms on the
  collision scenes (HJCD 1.9–2.6 ms), 1.8–2.2 ms on Fetch (faster than HJCD), 6–8.5 ms on Panda open-world via
  the bundled `franka.yml`, and faster than HJCD at 24 DoF (2.68 vs 2.99 ms). Pose accuracy 1e-5–1e-2 mm.
- PyRoki: 5–7.7 ms, sub-0.1 mm from B>=100. IKFlow: 2.6–6.2 ms with mm-level error.
- Hard sets (success = pose AND collision-free): HJCD 94–100% and cuRobo 90–100% under the sphere oracles,
  within a few points of each other on every set; under the mesh oracle both drop in the cage (77 / 74).
  PyRoki, which has no obstacle term, is 39–49% on the table sets. HJCD's accuracy is not monotone in B on the
  hard sets (cage 99/90/97 at B=100/1000/2000) — an open item.

## Caveats

1. The sphere oracles are not independent of HJCD (it filters with the `hjcd` model), so its 100% there is by
   construction; the mesh oracle is the independent judge and it is the one that lowers HJCD in the cage.
2. PyRoki's collision column measures unconstrained IK filtered by the oracle, not a collision-aware solver.
3. Timing is per-call wall time including each library's host overhead, as in the paper.
4. The paper-protocol row keeps PyRoki's 1.6 mm TCP mismatch; the dataset protocol has none.
5. One machine, one window, 100 targets/problems; no confidence intervals.

## Files

`results/`: per-solver CSVs (`open_*`, `fetch_open_*`, `collfree_paper_*`, `collfree_hand_<set>_*`,
`dof<N>_*`), merged tables (`table_*.md`, `table4_mmd.md`), Pareto plots, `*.metadata.json` sidecars,
`run.log` (NUL bytes stripped), `monitor.log`, and the two per-set summaries. `oracle_study/`: configuration dumps,
the multi-oracle table and the collection script. `results_superseded_curobo_nospheres/`: the morning's
unconstrained cuRobo Table II files and the re-run logs. `dry_runs/`: the aborted first
attempt, the 3-target functional dry run, and the baseline installer log. `run_paper_full.sh`: the runner.
