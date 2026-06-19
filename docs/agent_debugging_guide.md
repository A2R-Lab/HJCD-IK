# HJCD-IK agent debugging guide

Hard-won, HJCD-IK-specific institutional knowledge. **Read before changing the kernel, codegen, or robot
config.** HJCD-IK is a batched GPU IK solver: one CUDA block per problem, warp-per-candidate, warp-scoped
math throughout. Companion docs: [`CLAUDE.md`](../CLAUDE.md), [`STARTUP_PROMPT.md`](STARTUP_PROMPT.md),
[`HANDOFF.md`](HANDOFF.md).

## 0. Validation checklist (before committing)

1. **Regenerate `grid.cuh` and rebuild** if the URDF or EE target changed:
   `python scripts/generate_grid.py include/test_urdf/panda.urdf -t panda_grasptarget_hand` then
   `python -m pip install -e .`. Stale `grid.cuh` = silently wrong FK/Jacobian.
2. **`FLANGE_IDX` / target agreement** across `grid.cuh`, `src/hjcd_kernel.cu`, and any benchmark problem.
3. **Build clean** and import: `python -c "import hjcdik; print(hjcdik.num_joints())"`.
4. **Run the benchmark** and compare to the committed baseline (do not eyeball):
   `python benchmark/ik_benchmark.py --skip-grid-codegen` → solved-rate, pos/ori error, timing.
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
**Cause:** the solver is single-warp in places and multi-warp in others; the `SYNC()` macro picks
`__syncwarp()` vs `__syncthreads()` by block size. Mixing them wrong races the Jacobian/solve steps.
**Fix:** every cross-lane/cross-warp dependency needs the right barrier; warp-only sections use `__syncwarp(mask)`.
Use `compute-sanitizer --tool racecheck` when available.

### 1d. Robot constants hardcoded
**Symptom:** wrong sizes / OOB after swapping robots.
**Cause:** `NUM_JOINTS`, `TOPOLOGY_HELPERS_COUNT`, transform counts are **per-URDF**. The GRiD example header
is a 19-DOF robot; Panda regenerates to `NUM_JOINTS=7`.
**Fix:** read counts from the generated `grid::` symbols; never hardcode.

### 1e. Submodule not initialized
**Symptom:** CMake can't find GRiD/GLASS; codegen script missing.
**Fix:** `git submodule update --init --recursive`. Note `external/GRiD` is the codegen source and
`external/GLASS` provides the warp linear-algebra primitives.

### 1f. Collision geometry mismatch
**Symptom:** free targets flagged in-collision (or vice versa).
**Cause:** per-robot collision spheres (`src/robots/{panda,fetch}.cuh`) are hand-tuned and Panda/Fetch-only.
**Fix:** for a new robot, add its collision primitives or run without `--collision-free`.

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
Full data: [`open-tasks/multiwarp_timing_result.md`](open-tasks/multiwarp_timing_result.md),
[`open-tasks/perf_attribution_2026-06-16.md`](open-tasks/perf_attribution_2026-06-16.md).

- **⚠️ STALE-BINARY TRAP (the #1 perf-methodology bug — it invalidated an entire timing/correctness pass):**
  `ninja -C build` rebuilds `build/_hjcdik*.so`, but Python imports the **editable-install copy** under
  `.venv/.../site-packages/hjcdik/`, which `ninja` NEVER touches. So `ninja`-only "rebuilds" leave the
  RUNNING binary stale — env knobs (`HJCD_LM_WARPS`, `HJCD_LM_EPS_*`) and code changes silently have no
  effect. **Always rebuild with `scripts/rebuild.sh`** (ninja + copy `.so` into site-packages + import
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
  (`scripts/perf/dof_scaling_ab.sh --nsys`) showed `coarse_search` is **88%** of the 24-DoF wall and scales
  ~O(N³), while `lm_tuner` only grows 2.5×. Root cause: the greedy candidate loop (`hjcd_kernel.cu:1231-
  1289`) runs O(N³) work **serialized on `lane==0`** (the FK scratch `l_C`/`l_tmp` is per-warp, so 31/32
  lanes idle) with **two full O(N) `ee_fk_thread` chains per candidate** — the suffix/partial-FK recompute
  that was deferred during the GRiD-FK migration. *Lesson: a kernel's name and the most-numerically-scary
  line are not evidence; split the wall by kernel (nsys `cuda_gpu_kern_sum`) and A/B the suspected lever
  before believing a cause. The fix for a warp kernel that scales badly is almost always restoring
  warp-parallelism (here: parallelize the inner loop + partial FK), not changing precision.*
- **Tolerance is a per-regime lever, and looser tol can be a trap.** The LM early-stop (`:796/802`) only
  shortens `lm_tuner`; at high DoF where `coarse_search` dominates, a looser tol buys ~1.05× *and* craters
  accuracy (11–140 mm — it returns coarse-quality solutions). Keep tight 1e-8 unless you've confirmed via
  nsys that the LM loop is the bottleneck for your DoF/num_solutions.

## 6. Lessons log
*(Append new bug classes / tricks here as they emerge — keep this guide the single source of truth.)*
