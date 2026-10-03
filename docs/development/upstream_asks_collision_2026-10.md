# Upstream asks — collision primitives and models (GRiD, GLASS), October 2026

Written for: the GRiD and GLASS agents/maintainers, and whoever lands the HJCD-IK follow-up.

**Status: proposal, gated on the HJCD-IK quiet-window timing gate** (`docs/open-tasks/ab_2026-10-03/`). Nothing here
starts until that gate confirms the collision-aware refinement (`33d312f`) costs nothing open-world and an acceptable
amount in collision mode. Decision record: user, 2026-10-04 — "after timing, if this is good, upstream the collision
models and improvements to GRiD, and any relevant GLASS improvements, to keep things modular".

## Why

HJCD-IK's collision-aware refinement needed three things GRiD's `grid_collision` does not expose today, so they were
built HJCD-side as a stopgap (`csrc/kernel/hjcd_kernel.cu`, `csrc/generated/hjcd_collision_tables.cuh`):

1. a **warp-scoped** collision verdict from already-computed joint world transforms (GRiD's `config_free` is
   block-cooperative: it runs the multi-target FK with `__syncthreads`, so a single warp of a multi-warp block cannot
   call it);
2. the **sphere batch tables** (anchor slot / local offset / radius) as addressable device data (GRiD bakes them as
   function-local `static const` arrays inside the FK extractor);
3. a way to **measure and choose sphere models** (bulge vs. coverage against the true link meshes) and to build
   conservative ones (`benchmark/make_bounded_bulge_spheres.py`).

Evidence that it matters (`docs/development/evidence/fairness_hardsets_2026-10-03/`, results.rst *Collision-aware
refinement*): hard-set success 96/94/95 → 100/100/100 %; on identical spheres HJCD-IK = cuRobo on the whole clearance
ladder; the sphere model, not the search, decides tight scenes.

## GRiD asks (grid_collision namespace, codegen + runtime)

| # | Ask | Shape | Replaces in HJCD-IK |
| --- | --- | --- | --- |
| G1 | **Expose the sphere batch** as `__constant__`/`__device__ const` arrays in the collision namespace: `sphere_anchor[N]` (movable-joint slot of `s_Xworld`), `sphere_offset[3N]`, `sphere_radius[N]`, plus `NUM_SPHERES`. Same order as the FK extractor's batch, base spheres dropped. | codegen (`GRiDCodeGenerator`, collision emitter) | the sidecar `hjcd_collision_tables.cuh` and its `generate_grid.py` emitter |
| G2 | **Warp-scoped verdict** `grid_collision::warp::config_free<T>(const T* s_Xworld, const Environment<float>& env, float* w_scratch)`: lanes stride the spheres (place from G1 tables, `grid_cc_sphere_in_environment`), `__syncwarp`, lanes stride the self-collision ranges, `__any_sync` reduce; every lane returns the verdict. Entered by a full warp; no block barrier. Also `warp::collision_distance` (per-sphere min signed distance + normal, lanes over spheres) for a future gradient/nullspace repair step. | runtime header (`grid_collision_geometry.cuh` or a new `grid_collision_warp.cuh`) | `warp_config_free` in `hjcd_kernel.cu` |
| G3 | **Broad→fine cascade in the warp and block paths**: per-anchor bounding sphere derived from the fine rows (`_broad_tier_from_rows` already does this at bake time) tested first against the environment and for self pairs (anchor-pair hit mask), fine spheres only where the broad one hits. This is what makes a 200–400-sphere conservative model affordable (today 3–4× latency on HJCD's loops). | codegen + runtime | nothing yet (HJCD has no cascade) |
| G4 | **Parallelise the block path**: `config_free` / `collision_distance` currently run the env and self loops serially on every thread ("W3 perf TODO" in the header). Thread-per-sphere + block any-reduce; thread-per-range for self. | runtime | the post-solve `mark_collisions` cost scaling with sphere count |
| G5 | **Bounded-bulge mesh spherizer** as a mode of `_spherize.py` next to the voxel fill: voxelise + EDT, candidates = inscribed radius + bulge budget, greedy surface cover (port of `benchmark/make_bounded_bulge_spheres.py`); and a `--report` that evaluates ANY spherized URDF against the mesh (count, max/p99 bulge, uncovered surface %). Fidelity is the number a paper needs beside a sphere count. | codegen (`algorithms/_spherize.py`, CLI) | `benchmark/make_bounded_bulge_spheres.py` |
| G6 | **Named Panda sphere presets** in GRiD's assets with provenance + fidelity: foam `smaller_panda_spherized` (58; bulge 40 mm, 4–35 % uncovered), cuRobo `franka.yml` spheres (61; Apache-2.0, NVIDIA; bulge 75 mm on link 5, 20–71 % uncovered), bounded-bulge b10/b15 from the `panda_description` visual meshes. `generate_grid.py --spherized-urdf` then takes a preset name. | assets + codegen | `benchmark/reference/panda_curobo_spherized.urdf`, `make_curobo_sphere_urdf.py` |

Non-asks (stay in HJCD-IK, they are solver policy): the collision-aware early stop, the colliding-seed rank penalty,
the repair round, the success-band fallback, `HJCD_CC_STOP` / `HJCD_REPAIR_ATTEMPTS`. After G1–G2 land they become
~40 lines calling `grid_collision::warp::config_free`.

Interface detail that must not drift: HJCD's `s_jointX[16*jid]` for `jid < NUM_JOINTS` is GRiD's `s_Xworld[16*jid]`
(column-major 4×4; `ee_pose_inner_warp` fills it), so G2 can take exactly that pointer. The base link (`anchor < 0`) is
excluded at bake time on both sides.

## GLASS asks (small; only if they are genuinely generic)

| # | Ask | Note |
| --- | --- | --- |
| L1 | `glass::warp::transform_points<T>(const T* X_colmajor16, const float* pts, float* out, int n)` — lanes stride points, `R·p + t`. | Used by G2 to place spheres; also by any multi-target extractor. Trivial, but it is the one piece of G2 that is pure linear algebra. |
| L2 | `glass::warp::any(bool)` / `all(bool)` thin wrappers over `__any_sync`/`__all_sync` with the FULL mask, for symmetry with `warp::reduce` / `argmin_pair`. | Keeps domain code free of raw intrinsics; no perf effect. |

Nothing else in this work is linear algebra; the sphere/primitive SDFs are GRiD's.

## HJCD-IK follow-up once G1–G2 (and optionally G3) land

1. Bump `external/GRiD` (and the nested GLASS pin together with `external/GLASS` — `bootstrap.sh` checks they match),
   regenerate `grid.cuh`, delete the sidecar emitter and `warp_config_free`, call the GRiD API.
2. Re-run `tests` + `scripts/bench/capture_baseline.py` (statistical regression), the clearance ladder
   (`docs/open-tasks/paper-2026-10-03/run_hjcd_variant.sh`) and the timing A/B (`docs/open-tasks/ab_2026-10-03/`):
   verdicts must be identical, latency within noise.
3. With G3: re-evaluate the conservative models (b10/b15) — if the cascade brings them to foam's latency, the website's
   sphere-fidelity table gets a row that is BOTH conservative and fast, which neither foam nor cuRobo's model is.
4. Receipt, docs (`CLAUDE.md` Conventions paragraph, `upgrading.md`), push.

## Where the agents pick it up

GRiD: `~/Desktop/GRiD/docs/open-tasks/hjcd_asks_collision_primitives_2026-10-04.md`;
GLASS: `~/Desktop/GLASS/docs/open-tasks/hjcd_asks_warp_helpers_2026-10-04.md` (both local open-task ledgers, 2026-10-04).

## Sequencing

timing gate (HJCD, quiet window) → G1 + G2 + L1/L2 (small, unblock the HJCD cleanup) → G5 + G6 (tooling/assets,
independent) → G3 + G4 (performance; measure on HJCD's ladder + A/B) → HJCD follow-up.
