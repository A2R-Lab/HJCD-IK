# Fairness pass + harder sets — RTX 5090, 2026-10-03 (correctness only; GPU shared)

**Audit erratum (2026-10-04):** the conservative b10/b15 ladder dumps contain 2775/2771 records,
not the expected 2776, because empty outputs were omitted. Their returned-record rates need recollection
before use as per-query success. Default-model dumps are complete. The “3–4× latency” observation below
is withdrawn as a performance claim because the GPU was shared. Visual meshes are an environment-only
approximation, not physical ground truth. Archived raw data remain unchanged.

Companion to `paper_rerun_2026-10-03/` (the latency campaign). Everything here is success-rate data collected
on a GPU shared with other jobs; **no latency in this directory is meaningful**. Summarised in
`docs/source/user_guide/benchmarks/results.rst`, section *All MotionBenchMaker sets* and *Clearance ladder*.

## What was changed before collecting

| Item | Change |
| --- | --- |
| PyRoki | collision-aware IK: PyRoki's `world_collision_cost` + `self_collision_cost` on the foam spheres via `RobotCollision.from_sphere_decomposition` (`baseline_bench.py --pyroki-collision on`, default). Its URDF capsule model was rejected: the dataset's own collision-free `goal_ik` violate it by 16–70 mm on cage/table sets. |
| Judges | `benchmark/collision_oracles.py`: `hull` = `panda_description` *collision* meshes (franka's convex hulls — MoveIt/MBM geometry), `visual` = the *visual* meshes (true shape), tolerance by shrinking obstacles; new `curobo` sphere model (cuRobo's `franka.yml` spheres on our chain). Fixed: world cache keyed by `id()`. |
| HJCD on cuRobo spheres | `benchmark/make_curobo_sphere_urdf.py` → `benchmark/reference/panda_curobo_spherized.urdf`; HJCD regenerated with it (`-t panda_hand_joint --collision --spherized-urdf …`, header sha `af92cbf7…`); solver label `hjcdik_cusph`. 59 non-base spheres compile (2 base spheres dropped, as with foam). |
| New sets | `benchmark/mbm_export.py` on MotionBenchMaker's public Panda dataset (`problems/download.sh panda`): `kitchen_panda`, `table_bars_panda` → `benchmark/problems/mb_extra_panda.json`. Mesh furniture decomposed exactly (boxes / 32-gon prisms → cylinders / rectilinear unions → boxes; one 4 %-overfull OBB on a bevelled counter piece). Validation against the original meshes: 100/100 request goals free under both; random configurations agree 99.7 % (kitchen) / 99.95 % (table_bars). |
| Ladder | `benchmark/make_clearance_ladder.py --include-level0 --deltas-cm 0.5 1 1.5 2 3` on cage / table_pick / table_under_pick; cuboids grow by *D* per face, cylinders and goals untouched; a problem survives if ≥ 1 of its `goal_ik` is free under the hull judge (1 mm). Levels with < 20 survivors skipped. `ladder/levels_used.json` lists the kept counts. The ladder was generated before the `id()` cache fix; regenerating after it changes only marginal cases (cage D=0 gains problem 36; the skipped ≥ 2 cm cage levels), so the collected results stand. |

## Kernel variants (same night)

Solver labels in the dumps: `hjcdik` = kernel before the collision-aware refinement (foam spheres);
`hjcdik_ccstop` = after it (collision-aware early stop + seed ranking + repair round + 5 mm band fallback;
foam spheres, header `7f4a82d9…`); `hjcdik_cusph_ccstop` = same kernel on cuRobo's spheres (`af92cbf7…`);
`hjcdik_b10_ccstop` / `hjcdik_b15_ccstop` = same kernel on the bounded-bulge full-cover models fitted to the
visual meshes by `benchmark/make_bounded_bulge_spheres.py` (377 / 200 spheres; the URDFs are in this
directory, they are NOT the default build). `sphere_model_fidelity.txt` is the `--report` of all four models.
`run_hjcd_variant.sh` collected every HJCD variant (B = 100, 2000, all three problem files, ~3 min each).

Headline deltas (B = 2000, true-mesh judge): cage/table_pick/table_under_pick 96/94/95 → **100/100/100**;
ladder table levels +5–9 points (now within 0–4 of cuRobo); cage −1 cm 49 → 64, −1.5 cm 0 → 4 on foam's
spheres, 95 / 96 on cuRobo's spheres (= cuRobo). b10/b15: 3–4× latency and lower success (cage −0.5 cm:
52 / 44 %).

## Layout

- `eight_sets/` — the 8 dataset sets, dataset protocol (`panda_hand`, `goal_pose` as posed), B = 100 and 2000:
  stored configurations per solver (`*.jsonl`), `oracle_table.md` / `scores.json` from
  `score_collision_oracles.py` with every judge. `hjcdik.jsonl` and `curobo.jsonl` are the
  `paper_rerun_2026-10-03/oracle_study` dumps re-scored; `pyroki_coll.jsonl` is the new collision-aware PyRoki.
- `ladder/` — the clearance ladder: dumps, per-judge table, `ladder.md` + `ladder.png`
  (`plot_clearance_ladder.py`, true-geometry judge).
- `new_sets/` — kitchen + table_bars, same layout.
- `run_*.sh` — the collection scripts as run (paths point at the staging clone).

## Headline numbers (B = 2000, pose < 5 mm / 0.05 rad AND free under the visual meshes, 1 mm; before → after the kernel refinement)

| set | HJCD-IK | HJCD-IK/cuRobo spheres | cuRobo v2 | PyRoki (collision-aware) |
| --- | --- | --- | --- | --- |
| cage | 96 → 100 | 96 → 100 | 100 | 100 |
| table_pick | 94 → 100 | 94 → 97 | 99 | 100 |
| table_under_pick | 95 → 100 | 92 → 97 | 98 | 100 |
| kitchen | 99 | 100 | 99 | 99 |
| table_bars | 100 | 96 | 100 | 100 |
| bookshelf / box sets | 99–100 | 98–100 | 98–100 | 100 |

Clearance ladder, cage, D = 1 / 1.5 cm: HJCD-IK 49 / 0, HJCD-IK on cuRobo spheres 91 / 82, cuRobo 95 / 96,
PyRoki 59 / 0. Foam's spheres accept 0 % of the hull-feasible dataset `goal_ik` at D = 1.5 cm (cuRobo's: 68 %),
i.e. the collision *model* sets the ceiling there. Table sets, D = 2 cm: HJCD-IK (either spheres) 76–80,
cuRobo 92, PyRoki 90–92 — a solver gap (post-solve filtering vs. collision cost in the loop).

## Judge disagreement, for the record

Under the convex-hull judge the same configurations score cage 77–81 % and table_bars 31–72 % for every solver
(spheres under-cover the hulls by up to 58 mm on link 5 — the hull is 50 % larger than the link). Under the
sphere judges table_bars is 100 % for everyone. The true-geometry judge is the headline for that reason.
