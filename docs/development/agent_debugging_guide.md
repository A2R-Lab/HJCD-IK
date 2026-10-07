# HJCD-IK agent debugging guide

Hard-won, HJCD-IK-specific institutional knowledge. **Read before changing the kernel, codegen, or robot
config.** Coarse search has one candidate per block and shares candidate state across warps;
LM refinement has one independent candidate per warp. Their synchronization scopes are intentionally different.
Companion docs: [`CLAUDE.md`](../../CLAUDE.md), [`STARTUP_PROMPT.md`](STARTUP_PROMPT.md).
The running roadmap / open-work list is `docs/open-tasks/TODO.md` (local, ignored by Git).

## 0. Validation checklist (before committing)

1. **Regenerate `grid.cuh` and rebuild** if the URDF or EE target changed:
   `python scripts/codegen/generate_grid.py csrc/urdf/panda.urdf -t panda_grasptarget_hand --collision --spherized-urdf external/foam/assets/panda/smaller_panda_spherized.urdf` then
   `python -m pip install -e .`. Stale `grid.cuh` = silently wrong FK/Jacobian.
2. **`FLANGE_IDX` / target agreement** across `grid.cuh`, `csrc/kernel/hjcd_kernel.cu`, and any benchmark problem.
3. **Build clean** and import: `python -c "import hjcdik; print(hjcdik.num_joints())"`.
4. **Run the benchmark** and compare to the committed baseline (do not eyeball):
   `python benchmark/hjcd_ik_bench.py --skip-grid-codegen` → solved-rate, pos/ori error, timing.
5. **Thread/warp sweep** if you touched the solver loop: validate at 1 warp (32) and multi-warp block sizes;
   confirm results are batch-size-invariant (divergence at larger blocks ⇒ missing sync).
6. **No GPU contention during timing.** Other agents run heavy GPU work on this machine — isolate timing runs.

## 1. Recurring bug classes

### 1a. Stale `grid.cuh` (wrong kinematics)
**Symptom:** loss decreases but the final EE pose is wrong / convergence is erratic.
**Cause:** URDF or target changed without re-running codegen.
**Fix:** regenerate + rebuild (checklist #1). The build caches; force a clean rebuild if in doubt.

### 1b. `FLANGE_IDX` / EE-target mismatch
**Symptom:** IK "solves" but FK of the solution puts the EE at the wrong link.
**Cause:** the fixed target index from codegen (`panda_grasptarget_hand`) disagrees with the kernel.
**Fix:** capture the index GRiD assigns; verify every `FLANGE_IDX`/`EE_IDX` use in `hjcd_kernel.cu`.
**Note:** GRiD's named-fixed-target dispatch has had bugs (named handles calling the unsuffixed all-leaf
kernels → off by the flange offset). When validating FK, compare against a Python reference, not just "it ran".

### 1c. Warp-vs-block sync in the solver loop
**Symptom:** diverges erratically; results differ by block/batch size.
**Cause:** coarse candidate state is block-shared, but LM candidates and coarse pair-evaluation
scratch are warp-local. A warp barrier cannot publish data across warps, and block barriers inside
independently diverging LM iterations can deadlock (including partially populated final blocks).
**Fix:** use block barriers for shared coarse candidate state; LM's `SYNC()` is always
`__syncwarp(mask)`. Use `compute-sanitizer --tool racecheck`, not only numerical comparisons.

### 1d. Robot constants hardcoded
**Symptom:** wrong sizes / OOB after swapping robots.
**Cause:** `NUM_JOINTS`, `TOPOLOGY_HELPERS_COUNT`, transform counts are **per-URDF**. The GRiD example header
is a 19-DOF robot; Panda regenerates to `NUM_JOINTS=7`.
**Fix:** read counts from the generated `grid::` symbols; never hardcode.

### 1e. Submodule not initialized / GLASS pin drift
**Symptom:** CMake can't find GRiD/GLASS; codegen script missing; or `grid.cuh` fails to compile with
unknown `glass::` symbols.
**Cause:** `external/GRiD` is the codegen source and `external/GLASS` provides the linear-algebra
primitives. Since the header is generated with `vendor_glass=False`, `grid.cuh` includes the
**top-level** `external/GLASS` instead of vendoring its own copy, so GRiD's nested GLASS pin and
`external/GLASS` must agree.
**Fix:** `bash scripts/setup/bootstrap.sh` (pins all three submodules). After bumping either pin,
regenerate with `scripts/codegen/generate_grid.py` and run `scripts/codegen/check_grid_cuh_fresh.py`.

### 1f. Collision geometry mismatch
**Symptom:** free targets flagged in-collision (or vice versa).
**Cause:** the baked collision spheres don't match the robot — `grid.cuh` was generated without
`--collision` (no `grid_collision` namespace → the Python API rejects collision-free requests; see the
`HJCD_HAS_COLLISION` guard), or the spherization source is wrong (Panda needs the foam
`--spherized-urdf`, not on-disk meshes).
**Fix:** regenerate with `--collision` (and, for Panda, `--spherized-urdf …/smaller_panda_spherized.urdf`);
or run without `--collision-free`.

### 1g. Scratch reuse and stop-flag read/write overlap
**Symptom:** numerical tests and memcheck pass, but Racecheck reports shared-memory WAR hazards.
**Cause:** a shuffle reduction exchanges registers; it is not a shared-memory fence. Lane 0 must not
overwrite per-warp anchor FK scratch until all lanes finish reading the prior anchor. Similarly, a
block-shared stop flag must not be overwritten while another warp is still making its previous exit decision.
In LM's final output, every lane must finish reading the shared restore condition before lane 0 overwrites
the error used by that condition; the non-restoring path also needs a warp fence.
**Fix:** fence anchor-buffer reuse with `__syncwarp`; publish each coarse stop decision once and reuse
that stable value until the next synchronized update. Do not add block barriers to warp-local LM math.
Build an isolated diagnostic wheel with `-Ccmake.define.CMAKE_CUDA_FLAGS=-lineinfo` to obtain source lines.
A useful regression target is `tests/test_multiwarp.py::test_partial_last_block_no_crash` under
`compute-sanitizer --tool racecheck --error-exitcode=1`; require zero warnings as well as zero errors.

## 2. Debugging methodology
- **Shrink:** `--num-targets 1 --batches "1"`, print inputs/outputs, check the numbers are sane.
- **Separate FK from the solver:** call FK on the returned solution and compare to a Python reference
  (`grid_rbd` / RBDReference) before blaming the optimizer.
- **Unconstrained first:** disable collision, confirm IK converges, then re-enable.

## 3. GRiD codegen gotchas
- Symbolic codegen (sympy) is slow on first run; subsequent runs cache.
- The fixed-target name (`-t`) must be a real frame in the URDF.
- Editing GRiD codegen invalidates the `.so` cache only if the cache key hashes the codegen source — when in
  doubt, force a clean rebuild.

## 4. Warp-locality discipline (integration work)
HJCD-IK's speed comes from warp-per-candidate parallelism. When refactoring math onto GRiD/GLASS:
- Use **warp-scoped** primitives (`grid::ee_pose_inner_warp`, `glass::warp::*`) — never block-scoped
  (`glass::` default) or cooperative-groups (`glass::cgrps::`) or vendor (`glass::nvidia::`) at these tiny,
  warp-dispatched sizes.
- **A/B every swap** at production thread counts against the hand-rolled version; keep the hand-rolled code if a
  primitive can't match, and refine the upstream primitive instead of regressing HJCD.

## 5. Performance learnings (measured on RTX 5090 sm_120, CUDA 13)
The raw multi-warp / perf-attribution sweeps (June 2026) were local notes and are not tracked; the
tracked timing evidence is `docs/development/evidence/targeted_timing_2026-09-27/` and the
audit record `docs/development/audit_hardening_validation.md`.

- **⚠️ STALE-BINARY TRAP (the #1 perf-methodology bug — it invalidated an entire timing/correctness pass):**
  `ninja -C build` rebuilds `build/_hjcdik*.so`, but Python imports the **editable-install copy** under
  `.venv/.../site-packages/hjcdik/`, which `ninja` NEVER touches. So `ninja`-only "rebuilds" leave the
  RUNNING binary stale — env knobs (`HJCD_LM_WARPS`, `HJCD_LM_EPS_*`) and code changes silently have no
  effect. **Always rebuild with `scripts/setup/rebuild.sh`** (ninja + copy `.so` into site-packages + import
  check) or `pip install -e . --no-build-isolation`. *Symptoms that you're on a stale binary: a new env
  knob does nothing; a temporary `printf` in the kernel never prints; results don't change when they
  obviously should. Verify with `python -c "import hjcdik._hjcdik as m; print(m.__file__)"` + its mtime vs
  your last build.* This trap made multi-warp look "flat" and the tol knob look "dead" — both false.
- **Multi-warp (W LM candidates/block) is SLOWER on a big GPU — measured (correct binary): W=1 fastest,
  W=8 up to 41% (ns=1) / 63% (ns=4) slower.** `Krep` candidates is FIXED regardless of W, so regrouping into
  `ceil(Krep/W)` blocks × W warps does NOT raise total warps/SM — and the bigger blocks (32·W threads,
  W·~4 KB smem) cut co-resident blocks/SM, hurting latency hiding. The workload (Krep ≈ 160–2200) is far
  below saturation, so there's no occupancy deficit to fill. *Lesson: "pack more warps/block" raises
  occupancy only if it raises total resident warps/SM AND you were warp-starved; when the work-item count is
  fixed and ≪ saturation it does nothing and the bigger blocks cost you. Check `Krep/numSMs` vs the
  warp/block caps first.* Default = **W=1**; W>1 retained only as an opt-in for low-SM devices (Jetson —
  plausible there, untested).
- **fp32-vs-fp64 refine — regime-dependent, the most important perf knob:**
  - `num_solutions≥2` (early-stop OFF ⇒ **throughput-bound**, all Krep candidates run full): fp32 is
    **5–7× FASTER** (the 5090's 1/64 fp64 throughput penalty dominates). Sub-micron.
  - `num_solutions=1` at the tight 1e-8 m tol: fp32 ~1.2× *slower* (can't reach the tol below its float
    floor, grinds all iters). **BUT with a precision-appropriate looser tol (~1e-6 m) fp32 EARLY-STOPS and
    is ~2× FASTER than fp64@1e-8, still sub-micron (~330 nm)** — so the tolerance IS a lever here.
  *Lesson: classify latency- vs throughput-bound before judging mixed precision (the 1/64 penalty is
  THROUGHPUT); and a too-tight tol can mask a precision win by forcing full iteration counts. A "negative"
  result in one regime/tol can be a multi-× win in another.* The convergence early-stop itself is correct
  (verified: a 1 m tol collapses to coarse quality immediately).
- **Timing-harness gotcha (self-contention false-positive):** a quiet-GPU guard that samples
  `nvidia-smi utilization.gpu` *between configs of its own sweep* will read its OWN recently-run kernels
  (util is time-averaged) and abort as if another process were contending. Likewise the compute-apps list
  includes the sweep's own ~0.5 GB CUDA context. Fix (in `scripts/perf/time_multiwarp_sweep.py`): exclude
  `os.getpid()` from the foreign-apps check, and only util-gate at STARTUP (before any solving) — mid-sweep,
  gate on foreign *apps* only. *Lesson: a process can't use util to tell if IT is the one keeping the GPU
  busy; gate on foreign pids, not aggregate util, once you're running.*

- **⚠️ nsys-SPLIT before attributing a high-DoF / scaling cost to a kernel.** When 24-DoF was ~13× slower
  than 7-DoF, the "obvious" suspect was the fp64 O(DoF³) warp-Cholesky in `lm_tuner` — **wrong.** A
  fp32-vs-fp64 A/B (fp32 came out *slower* at every DoF) refuted it, and the per-kernel nsys split
  (`scripts/perf/dof_scaling_ab.sh --nsys`) showed `coarse_search` was **88%** of the 24-DoF wall and scaled
  ~O(N³), while `lm_tuner` only grew 2.5×. Root cause (since fixed): the greedy candidate loop ran O(N³)
  work **serialized on `lane==0`** with **two full O(N) `ee_fk_thread` chains per candidate**. The fix is
  the current lane-parallel sweep + `ee_fk_suffix_thread` (recompute only the suffix from the perturbed
  joint). *Lesson: a kernel's name and the most-numerically-scary line are not evidence; split the wall by
  kernel (nsys `cuda_gpu_kern_sum`) and A/B the suspected lever before believing a cause. The fix for a
  warp kernel that scales badly is almost always restoring warp-parallelism, not changing precision.*
- **Tolerance is a per-regime lever, and looser tol can be a trap.** The LM early-stop (`eps_pos`/`eps_ori` in `solve_lm_batched`; env `HJCD_LM_EPS_POS/ORI`) only
  shortens `lm_tuner`; at high DoF where `coarse_search` dominates, a looser tol buys ~1.05× *and* craters
  accuracy (11–140 mm — it returns coarse-quality solutions). Keep tight 1e-8 unless you've confirmed via
  nsys that the LM loop is the bottleneck for your DoF/num_solutions.

### Statistics output comparisons and coarse early-stop

Requesting multiple solutions disables LM early-stop, not coarse early-stop. Coarse
blocks still publish a first-success flag, so scheduling (including sanitizer
instrumentation) can change the candidate pool and returned joint configurations
between otherwise identical solves. A test requiring identical arrays with and
without statistics should use one coarse block and multiple outputs, removing
inter-block early-stop while retaining the strict numerical comparison. Larger-batch
tests should check solution quality, collision policy and statistics consistency,
not exact candidate identity.

## 6. Lessons log
*(Append new bug classes / tricks here as they emerge — keep this guide the single source of truth.)*

- **Candidate identity is not run-to-run stable, by design (ruled 2026-10-02).** Coarse blocks race on the
  cross-block early-stop flag (`g_stop`), and the LM refine does the same when one solution is requested, so
  two identical calls can return different (equally valid) candidates. Accepted behaviour: do not add a
  deterministic mode. Consequences: compare builds statistically (returned counts, error distributions,
  `scripts/bench/capture_baseline.py` aggregates), never bitwise; regression tests use slack bands over many
  targets; do not write tests that pin a specific returned configuration.

- **Bit-identical refactors vs. tuned numerics (2026-10).** The LM loop applies robust row weights
  twice on a trial step: once folded into `row_s` at the iteration start, and again (fresh, at the
  trial's own position error) when scoring the backtracking trial — while the dogleg / coordinate
  fallbacks score with `row_s` only. This asymmetry is part of the tuned paper behaviour and was kept
  verbatim in the 2026-10 tidy-up; `robust_row_weights` / `ee_residual6` just name the shared pieces.
  Changing it is a numerics experiment that needs the GPU regression + a paper-protocol rerun, not a cleanup.
- **A change under `csrc/` cannot be "done" without a GPU.** The CPU-only CI only verifies the signed
  `gpu-proof.json`, whose fingerprint covers `csrc/`, `tests/`, `docs/source/`, scripts and the submodule
  gitlinks. Compile-only checks (`nvcc -c`, available via the `nvidia-cuda-nvcc` pip wheel without a GPU)
  catch syntax and template errors, but every such change still needs `pytest tests` + a re-recorded
  receipt on a real GPU before merge.
- **Collision benchmarking is a comparison of collision MODELS first, solvers second (2026-10-03).** Things
  that bit us, with the fix in parentheses: (a) cuRobo's `RobotBuilder` robot from a mesh-less URDF has NO
  spheres → "collision-free" IK silently unconstrained (use the bundled `franka.yml` re-targeted);
  (b) PyRoki's IK example has no obstacle term and its URDF capsule model is 16-70 mm too coarse for the
  MotionBenchMaker gaps (feed it the foam spheres via `RobotCollision.from_sphere_decomposition`,
  `--pyroki-collision on`); (c) `panda_description`'s *collision* meshes are convex hulls (link5's is 50 %
  larger than the link) — a "mesh oracle" built on them is MoveIt's conservative geometry, not ground truth;
  the *visual* meshes are (`MeshOracle(geometry="visual")`); (d) every sphere model under-covers the hulls
  and over-covers the true link somewhere, so thin obstacles (table_bars) flip verdicts between judges —
  always report several judges (`benchmark/score_collision_oracles.py`) and say which one is the headline;
  (e) on the clearance ladder the foam spheres accept 0 % of the dataset's own hull-feasible `goal_ik` at
  cage −1.5 cm vs 68 % for cuRobo's spheres: when a solver "fails" a tight scene, first check whether its
  own model admits ANY solution (`benchmark/make_curobo_sphere_urdf.py` builds HJCD on cuRobo's spheres to
  separate model from search). (f) `MeshOracle` once cached worlds by `id(world_dict)`; transient dicts reuse
  ids → stale geometry for a different problem. Cache by content.
- **MotionBenchMaker problems are reproducible from the public dataset.** `tests/mb_problems.json` is exactly
  MBM's `problems/download.sh panda` YAML (scene primitives in the panda_link0 frame, xyzw→wxyz, plus a robot
  stand under the base; verified bit-for-bit on cage). `benchmark/mbm_export.py` converts any MBM folder,
  including the mesh scenes (kitchen, table_bars) by exact rectilinear box decomposition / prism→cylinder.
  `goal_pose` of an exported set is FK(request joint goal) at panda_hand (MBM's own IK tolerance puts the
  request goal up to 14 mm from its pose query, so the dataset's pose and our FK differ by that much).
- **The early stop was collision-blind (fixed 2026-10-03).** `g_stop` was raised by the first pose-accurate
  candidate whether or not it collided, so cluttered scenes stopped the batch on a candidate the hard filter
  then discarded. The fix is warp-scoped (`grid_collision::warp::config_free` over generated tables on the per-warp
  `s_jointX`) and must stay so: `grid_collision::config_free` is block-cooperative (multi-target FK with
  `__syncthreads`) and cannot be called from one warp of a multi-warp LM block. Traps hit on the way:
  (a) the LM loop counter `it` is a per-lane register — a restart decided by lane 0 must be signalled
  through warp-shared scratch (`accurate == 2`) and applied by every lane, or the warp desynchronises;
  (b) `warp_base` (coarse per-warp dynamic smem) is reused as sphere scratch only where the anchor FK
  buffers are dead, and the host sizes it as max(FK buffers, 3·NUM_SPHERES floats); (c) `lm_eps` is 1e-8 m,
  so "accurate" in the LM means fully converged — the practically successful candidates live in the 5 mm
  band below it, which is why the old kernel's cage "successes" at 1.4–3.5 mm vanished when the stop became
  collision-aware (warps kept converging into the obstacle) until the band fallback was added. Diagnose
  stage by stage with `HJCD_CC_STOP=0/1/2/3` and `HJCD_REPAIR_ATTEMPTS=0`.
- **Before improving the repair step, check what the misses ARE (2026-10-07).** The scorer's `pose ok` vs
  `pose & own spheres` columns tell you whether any accurate-but-colliding candidates are being returned at
  all. With the collision-aware stop + band fallback they are not (equal in every ladder cell), so a smarter
  repair (nullspace step along contact normals — implemented, measured, archived in
  `evidence/repair_nullspace_2026-10-07/`, not shipped) has nothing to act on; the residue is the sphere
  model. Success is monotone in B now (`evidence/c1_batch_monotonic_2026-10-07/`).
- **Finer conservative spheres are not "better" on this benchmark.** Measured with
  `benchmark/make_bounded_bulge_spheres.py --report`: foam bulges 40 mm and leaves 4–35 % of the true
  surface uncovered; cuRobo's model bulges 75 mm on link 5 and leaves 20–71 % uncovered; a full-cover model
  with ≤13.5 mm bulge needs 377 spheres and scored lower in archived runs. Shared-GPU latency observations
  are not valid speed comparisons; conservative-model groups with missing queries require recollection. Compare
  solvers on identical spheres; if a conservative model is ever required, it needs GRiD's broad→fine cascade.

- **Coarse constant-copy WAR hazard (2026-10-04).** Copying every transform cell from shared source matrices
  inside independently progressing warp loops reads cells that FK immediately overwrites. nvcc can reuse
  the dead source cells' shared slots for scores written by another warp. Initialize each warp's constant
  transform cells once, before the loops, followed by a block barrier. Racecheck the diagnostic binary,
  including fp32/fp64, partial multi-warp blocks, and collision repair/stop settings; numerical tests alone
  missed this. Do not put a block barrier inside the divergent loop.
- **Benchmark accounting and precision (2026-10-04).** One record per attempted query, including empty output.
  Score pose via independent FK with explicit target/frame, reject incomplete groups, and compare both errors
  on one candidate. Never infer accuracy from count. The eighth positional Python solver argument is
  `refine_fp64`, not `write_stats`; name all options. See `timing_gate.md` for the corrected gate and errata.
