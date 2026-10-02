# CLAUDE.md — orientation for agents (and humans) on HJCD-IK

**HJCD-IK** = **Hybrid Jacobian Coordinate Descent Inverse Kinematics**: a GPU-accelerated, *batched*
IK solver (paper: [arXiv:2510.07514](https://arxiv.org/abs/2510.07514)). It generates many candidate IK
solutions in parallel for a 6-DOF end-effector target, with optional collision avoidance, built on top of
[GRiD](https://github.com/A2R-Lab/GRiD) (robot kinematics codegen) and
[GLASS](https://github.com/A2R-Lab/GLASS) (single-block CUDA linear algebra).

> **Before changing the kernel or codegen, read [`docs/development/agent_debugging_guide.md`](docs/development/agent_debugging_guide.md).**
> It is the runbook for HJCD-IK's recurring traps: stale `grid.cuh`, `FLANGE_IDX`/target mismatch,
> warp-vs-block sync in the solver loop, and submodule init.
> **The user-facing API contract is [`docs/source/user_guide/upgrading.md`](docs/source/user_guide/upgrading.md)**
> (hard collision filtering by default, `count` may be zero, errors raise, native `Result<T>` is move-only).

## Mental model

Each solve handles one target and a batch of candidate configurations. **Coarse search assigns one
candidate per block**, with warps evaluating joint-pair perturbations. **LM refinement assigns one
candidate per warp**, optionally packing several independent candidates into a block.
LM math and per-warp coarse scratch stay warp-scoped; coarse state shared across warps needs block
barriers. Two phases (`csrc/kernel/hjcd_kernel.cu`):
1. **Coarse search** (`coarse_search`): random restarts + greedy pairwise coordinate descent. The candidate
   sweep over the second joint runs **lane-parallel across the warp** (`for j = lane; j < N; j += WARP_SIZE`
   + warp min-reduce); each candidate recomputes only the **FK suffix** from its perturbed joint
   (`ee_fk_suffix_thread`, built on `grid::update_XmatHom_joint`) rather than a full chain — this is what
   makes high-DoF scale (was O(N³) serial-on-lane-0; see `docs/development/agent_debugging_guide.md` §5).
2. **LM refine** (`solve_lm_batched` / `lm_tuner`): single-warp Levenberg–Marquardt — build the 6×N geometric
   Jacobian (cross-products), form & solve the normal equations `(JᵀJ + λ·diag)Δq = Jᵀr` via a hand-rolled
   `__shfl` warp-Cholesky, with dogleg/line-search backtracking.

Forward kinematics produces the **world-frame joint transforms** `s_jointXforms[16·jid]` (4×4 each); the EE
pose error is computed as a **quaternion** error (`mat_to_quat` / `quat_err_rotvec`). For Panda: `N = 7`
joints. Target indices and transform counts are generated constants; the default
`panda_grasptarget_hand` build currently has `grid::EE_FIXED_FRAME_IDX = 10`.

## Key files

| Path | What it is |
|---|---|
| `csrc/kernel/hjcd_kernel.cu` | The solver: block-cooperative coarse search + warp-scoped LM refine + host orchestration (`generate_ik_solutions`). **The file you'll edit most.** |
| `csrc/kernel/hjcd_settings.h` | `HJCDSettings<T>` (coarse/LM tolerances, `lambda_init`), `mat4_mul`, FK helpers (`ee_fk_warp`/`ee_fk_thread`/`ee_fk_suffix_thread`), `#include "grid.cuh"`, `N`/`FLANGE_JID`/`GRASP_FIXED_IDX`. |
| `csrc/kernel/hjcd_kernel.h` | Native host API: move-only `Result<T>`, `generate_ik_solutions<T,RT>(target, batch, ...)`, `sample_random_target_poses`. |
| `csrc/kernel/main.cpp` | Native CLI (`single`/`sweep`/`from_csv`), open-world only; CTest-covered by `tests/native/`. |
| `csrc/generated/grid.cuh` | **Generated** GRiD kinematics + collision header. Do **not** hand-edit. Generated with `vendor_glass=False`, so it `#include "glass.cuh"`s the top-level GLASS instead of vendoring a copy (~6k lines, was ~15k). |
| `external/GRiD/` | Submodule: GRiD codegen (emits `grid.cuh` from a URDF). Its nested GLASS pin must equal `external/GLASS`. |
| `external/GLASS/` | Submodule: GLASS single-block / warp / thread linear algebra (`glass::warp::`, `glass::thread::`, `glass::block::`). |
| `external/foam/` | Submodule: pre-spherized Panda collision URDF (`assets/panda/smaller_panda_spherized.urdf`). |
| `csrc/kernel/grid_env.cuh` | Parses a MotionBenchMaker problem JSON → `grid_collision::Environment` (device obstacle set) for the collision kernels; `problem_document.h` caches the parsed document. |
| `csrc/bindings/pybind_module.cpp` | Python bindings → `generate_solutions`, `sample_targets`, `num_joints`, `collision_enabled`, `build_info`. `hjcdik/__init__.pyi` is the typed stub. |
| `tests/` | pytest suite (API contracts, FK equivalence, collision policy, codegen, stats, regression vs `baseline_metrics.json`); `tests/native/` = CTest native API + CLI contracts. `gpu-proof-tests.txt` is the signed-receipt manifest. |
| `gpu-proof.json` | Signed pytest-gpu-proof receipt of the last full GPU run; verified by CPU-only CI. |
| `benchmark/hjcd_ik_bench.py` | HJCD-IK benchmark harness: solved-rate, position/orientation error, timing. |
| `benchmark/baseline_bench.py` | Competitor baselines (PyRoki/cuRobo, `--mode`); optional, see `docs/source/user_guide/benchmarks/results.rst`. |
| `benchmark/baseline_ikflow.py` | IKFlow baseline (standalone, torch); same CSV/MMD-dump schema. |
| `benchmark/check_ee_frames.py` | Gated smoke test: do all solvers agree on the EE (panda_hand) frame? |
| `benchmark/gen_targets.py` | Neutral Halton + numpy-FK shared open-world targets (fair cross-solver compare). |
| `benchmark/{make_tables,plot_pareto}.py` | Merge per-solver CSVs → paper tables / accuracy-latency Pareto (Figs 4/5). |
| `benchmark/{mmd,run_mmd,gen_groundtruth_tracik}.py` | MMD/MMD² (Table IV): config dumps + TRAC-IK ground truth. |
| `tests/{mb,wall}_problems.json` | MotionBenchMaker / wall collision problem sets. |

## Build & test

CMake 3.24+ / CUDA 12.x or 13.x / pybind11 (scikit-build-core). GRiD codegen runs at configure time when enabled.

```bash
sudo apt install -y nlohmann-json3-dev   # collision environment JSON header dependency
git submodule update --init --recursive          # GRiD + GLASS
python -m pip install -e .                        # builds the _hjcdik extension (CUDA arch auto-detected)
python benchmark/hjcd_ik_bench.py --skip-grid-codegen   # run the solver
```

`./scripts/setup/setup_dev.sh` does all of the above (system deps + pinned submodules + venv + codegen + build).
`scripts/setup/bootstrap.sh` alone pins the three submodules (GRiD, GLASS, foam) at the committed revisions.

**Testing and the GPU-proof gate.** `python -m pytest tests` needs a CUDA GPU and the collision-enabled Panda
build. There is no GPU CI runner: `.github/workflows/verify-gpu-proof.yml` only verifies the committed, signed
`gpu-proof.json` receipt, whose fingerprint covers `csrc/`, `hjcdik/`, `tests/`, `benchmark/`, `scripts/`,
`examples/`, `docs/source/`, the build files and the submodule gitlinks. **Any change to those paths makes the
receipt stale**: run the full suite and `scripts/setup/run_gpu_proof.sh` on a GPU (Python ≥ 3.11) and commit the
new receipt, after `scripts/setup/update_gpu_proof_manifest.py` if test IDs changed. A compile-only check without a
GPU is possible with the `nvidia-cuda-nvcc` pip wheel (`nvcc -c csrc/kernel/hjcd_kernel.cu -arch=sm_80 ...`).
Native checks: `cmake -S . -B build-native -DBUILD_PYTHON=OFF -DHJCDIK_BUILD_NATIVE_TESTS=ON && cmake --build
build-native && ctest --test-dir build-native`. `hjcdik.build_info()` reports the compiled header's SHA-256 so a
stale install or wrong-robot wheel can be detected without initializing CUDA.

Python API:
```python
from hjcdik import generate_solutions, sample_targets, num_joints
targets = sample_targets(num_targets=10, seed=0)         # list of [x,y,z, qw,qx,qy,qz]
out = generate_solutions(targets[0], batch_size=2000, num_solutions=4)
# out = {joint_config, pose, pos_errors, ori_errors, count}; count may be < num_solutions (even 0)
# pos_errors are mm, ori_errors are rad; refine_fp64=-1 (default) = fp64 for one solution, fp32 for several
```

## Conventions / discipline

- **Never hand-edit `grid.cuh`.** It is GRiD codegen output. To change the robot or EE target, run
  `python scripts/codegen/generate_grid.py <urdf> -t <target_frame>` and rebuild. Robot constants
  (`NUM_JOINTS`, topology counts) are baked per-URDF — read them from the generated symbols, never hardcode.
- **Collision is URDF-driven (grid_collision), not hand-coded.** Passing `--collision` to
  `generate_grid.py` bakes GRiD's `grid_collision` namespace (per-robot spheres + self-collision ranges)
  into `grid.cuh`; the kernel scores it via `grid_collision::collision_distance` (see
  `csrc/kernel/hjcd_kernel.cu` `score_environment_costs`). Sphere source: `--collision-res R` spherizes
  the URDF's own collision geometry, OR `--spherized-urdf <foam.urdf>` reads a pre-spherized (foam-format)
  URDF directly — use the latter when the URDF's collision meshes don't resolve on disk. **Panda uses the
  checked-in foam sphere geometry** (`external/foam/assets/panda/smaller_panda_spherized.urdf`,
  58 non-base spheres), bound to the kinematic URDF's fixed frames. Its +/-40 mm finger origins
  differ from the legacy paper reference's +/-65 mm origins. The build/codegen wires this automatically
  (see `CMakeLists.txt`). This is
  the **bring-your-own-URDF** path: `generate_grid.py <robot.urdf> --collision [...]` provides FK and
  collision for supported fixed-base serial arms with no hand-written per-robot header.
- **Collision policy.** Python exposes `collision_mode="hard"|"soft"|"both"`; `hard` is the
  default and strictly excludes colliding candidates (self **+** environment). `soft` is a penetration-cost
  ranking mode and does not guarantee collision freedom; `both` ranks and filters. Neither the API nor the
  native solver reads `HJCD_CC_MODE`; only the benchmark CLI keeps it as the default of `--collision-mode`.
  All three are post-solve, off the hot warp loop. Obstacle JSON uses `pose` (`[x,y,z,qw,qx,qy,qz]`); the legacy
  Euler `box` schema is gone.
- **`FLANGE_IDX` discipline.** The fixed EE target (`panda_grasptarget_hand`) and its index must agree across
  codegen, the kernel, and any benchmark problem. A mismatch silently solves to the wrong frame.
- **Warp-locality is the performance contract.** New math must stay warp-scoped (`__shfl_*_sync`, `__syncwarp`).
  Do not drop the solver onto block-scoped primitives.
- **Prefer GLASS/GRiD primitives over hand-rolled math.** Quaternion ops, norms, 4x4 products, reductions and the
  normal-equation solve all come from `glass::thread::` / `glass::warp::` / `glass::block::`; the hand-rolled
  helpers that remain (`solve_pos`/`solve_ori`, `ee_residual6`, `robust_row_weights`, `rank_score`) are HJCD-specific.
  Generic kernels that outgrow HJCD belong upstream (kinematics → GRiD, linear algebra → GLASS).
- **Tidy-ups must be numerics-preserving unless validated on a GPU.** The LM cost path has deliberate asymmetries
  (see the debugging guide §6); refactor by naming shared pieces, not by "fixing" them.
- **Short, single-line commit messages; no `Co-Authored-By` footer.**

## History — where the code came from

**2026-07 — re-based on GRiD/GLASS (PR #2).** The bespoke Panda-only FK (`X_warp` / `X_single_thread`) was
replaced by GRiD's stock warp FK (`grid::ee_pose_inner_warp`), and the hand-rolled math (4x4 products, warp
reduce, warp Cholesky, quaternion error) moved onto GLASS's `glass::warp::` / `glass::thread::` tiers. The
end-effector frame is **per-robot** (codegen resolves `grid::EE_FIXED_FRAME_IDX` from the named target and
injects it; `hjcd_settings.h` consumes it) — see
[`docs/source/user_guide/benchmarks/results.rst`](docs/source/user_guide/benchmarks/results.rst)
for the per-robot EE map + how to regenerate the paper sweeps.

**2026-09/10 — audit hardening (PR #3).** Strict `hard` collision filtering became the default; the native API
got a move-only `Result<T>`, checked CUDA/GRiD initialization that raises instead of aborting, a solver lock, and
a per-device parsed-document + environment cache (scene switches ~14x faster on the host, kernel unchanged).
The build compiles the CUDA core once (`hjcdik_core`) for both front ends; native CTest + CLI contract tests,
`build_info()`, the typed stub and the signed GPU-proof manifest were added. Full record:
[`docs/development/audit_hardening_validation.md`](docs/development/audit_hardening_validation.md).

**2026-10 — de-vendored GLASS + tidy-up.** `grid.cuh` is generated with `vendor_glass=False` (one GLASS per
translation unit, header ~6k lines instead of ~15k). Dead helpers, duplicate host logic, the ignored native
`d_robotModel` argument, the `HJCD_CC_MODE` / `"auto"` shim and the legacy Euler obstacle schema were removed;
the device and host candidate ranking now share one `rank_score` (the host previously used a different
orientation weight).

**Collision migrated to `grid_collision`.** The former bespoke pRRTC stack (`csrc/collision/` +
`csrc/robots/{panda,fetch}.cuh`) is gone; collision is now GRiD's URDF-driven `grid_collision` baked into
`grid.cuh` (`--collision`), scored post-solve by `mark_collisions` for strict filtering and optionally by `score_environment_costs`
for soft ranking (the hot warp solver never touches collision). Strict filtering can return fewer
solutions than the historical soft-ranking path. The paper reference model lives frozen under
`benchmark/reference/panda_collision_model.cuh`; it is NOT identical to the compiled geometry
because the fixed finger openings differ. `benchmark/panda_model.py` provides explicit `paper`
and URDF-derived `hjcd` models. Paper comparisons retain `paper`; implementation tests use `hjcd`.
Both Python oracles check environment collisions only, not the kernel's self-collision policy.

The collision code path is compiled in only when `grid.cuh` was generated with `--collision` — codegen emits
a `#define HJCD_HAS_COLLISION 1` sentinel and the kernel + `grid_env.cuh` guard all `grid_collision::` use on
it. A no-collision header (e.g. the DoF-scaling regens, or any BYO-URDF built without `--collision`) still
compiles and runs open-world; the Python API rejects a collision-free request in that build.

**Performance status.** Published numbers are the camera-ready paper's (RTX 4060) and live in
`docs/source/user_guide/benchmarks/results.rst`; they have **not** been re-run end to end on the current code.
The tracked post-migration evidence is the audit timing gate
(`docs/development/evidence/targeted_timing_2026-09-27/`, RTX 5090: open-world B=2000 ≈ 1.3 ms fp64, kernel
within 1% before/after the audit). `scripts/bench/run_paper_experiments.sh` (with `HJCD_REGEN=1`) regenerates the
paper protocol; `scripts/perf/run_all_timing_sweeps.sh` is the HJCD-only timing capture.

> **Build/test gotcha:** `ninja -C build` does NOT update the imported `.so` (it's the editable copy in
> site-packages). Always rebuild with **`scripts/setup/rebuild.sh`** (or `pip install -e . --no-build-isolation`).

## What next

The running roadmap is `docs/open-tasks/TODO.md` (local, gitignored agent scratch — recreate it from this list
if missing). Tracked project docs: this file, `docs/development/agent_debugging_guide.md`,
`docs/development/STARTUP_PROMPT.md`, `docs/source/user_guide/upgrading.md`,
`docs/source/user_guide/benchmarks/results.rst`, and the sphinx docs. In priority order:

1. **Re-record the GPU proof** after any `csrc/`/`tests/`/docs change (see *Testing and the GPU-proof gate*).
2. **Rerun the paper protocol on the current pins** (`HJCD_REGEN=1 RUN_FETCH=1 RUN_DOF=1 RUN_MMD=1
   scripts/bench/run_paper_experiments.sh`) and record it as dated evidence under `docs/development/evidence/`.
3. **Collision-free regression test** over `tests/mb_problems.json` (still a TODO in `tests/test_regression.py`).
4. **Upstream candidates:** grasptarget-offset FK (`ee_fk_warp`/`ee_fk_thread`/`ee_fk_suffix_thread`) and the
   batched pose-7 FK kernel → GRiD; the warp dogleg step and a `gn_step` variant that exposes diag(A)/g → GLASS.
5. **Branched-chain support** in `ee_fk_suffix_thread` (needs the parent table; the GRiD primitive is general).
6. **Self-hosted GPU runner** so `.github/workflows/test.yml` can leave manual-only mode.
