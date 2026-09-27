# CLAUDE.md — orientation for agents (and humans) on HJCD-IK

**HJCD-IK** = **Hybrid Jacobian Coordinate Descent Inverse Kinematics**: a GPU-accelerated, *batched*
IK solver (paper: [arXiv:2510.07514](https://arxiv.org/abs/2510.07514)). It generates many candidate IK
solutions in parallel for a 6-DOF end-effector target, with optional collision avoidance, built on top of
[GRiD](https://github.com/A2R-Lab/GRiD) (robot kinematics codegen) and
[GLASS](https://github.com/A2R-Lab/GLASS) (single-block CUDA linear algebra).

> **Before changing the kernel or codegen, read [`docs/development/agent_debugging_guide.md`](docs/development/agent_debugging_guide.md).**
> It is the runbook for HJCD-IK's recurring traps: stale `grid.cuh`, `FLANGE_IDX`/target mismatch,
> warp-vs-block sync in the solver loop, and submodule init.

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
joints, `EE_IDX = 7`, `FLANGE_IDX = 8`, `NX = 9` stored frames.

## Key files

| Path | What it is |
|---|---|
| `csrc/kernel/hjcd_kernel.cu` | The solver: block-cooperative coarse search + warp-scoped LM refine. **The file you'll edit most.** |
| `csrc/kernel/hjcd_settings.h` | `HJCDSettings<T>`, `mat4_mul`, FK helpers (`ee_fk_warp`/`ee_fk_thread`/`ee_fk_suffix_thread`), `#include "grid.cuh"`, `N`/`FLANGE_JID`/`GRASP_FIXED_IDX`. |
| `csrc/generated/grid.cuh` | **Generated** GRiD kinematics header (FK, robot model). Do **not** hand-edit. |
| `external/GRiD/` | Submodule: GRiD codegen (emits `grid.cuh` from a URDF). |
| `external/GLASS/` | Submodule: GLASS single-block / warp linear algebra. |
| `csrc/kernel/grid_env.cuh` | Parses a MotionBenchMaker problem JSON → `grid_collision::Environment` (device obstacle set) for the collision-scoring kernel. |
| `csrc/bindings/pybind_module.cpp` | Python bindings → `generate_solutions`, `sample_targets`, `num_joints`. |
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

Python API:
```python
from hjcdik import generate_solutions, sample_targets, num_joints
targets = sample_targets(num_targets=10, seed=0)         # list of [x,y,z, qw,qx,qy,qz]
out = generate_solutions(targets[0], batch_size=2000, num_solutions=4)
# out = {joint_config, pose, pos_errors, ori_errors, count}
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
  ranking mode and does not guarantee collision freedom; `both` ranks and filters. `HJCD_CC_MODE` remains
  only as the benchmark compatibility fallback (`collision_mode="auto"`). All three are post-solve, off the hot warp loop.
- **`FLANGE_IDX` discipline.** The fixed EE target (`panda_grasptarget_hand`) and its index must agree across
  codegen, the kernel, and any benchmark problem. A mismatch silently solves to the wrong frame.
- **Warp-locality is the performance contract.** New math must stay warp-scoped (`__shfl_*_sync`, `__syncwarp`).
  Do not drop the solver onto block-scoped primitives.
- **Short, single-line commit messages; no `Co-Authored-By` footer.**

## Integration — re-based on GRiD/GLASS (merged to `main`, 2026-07-11)

HJCD-IK is re-based onto the latest GRiD (`modernizing-tests`) + GLASS (`main`) for modularity and
upstreamable performance. The bespoke Panda-only FK (`X_warp` / `X_single_thread`) was replaced by GRiD's
stock warp FK (`grid::ee_pose_inner_warp`), and the hand-rolled math (`mat4_mul`, warp reduce, warp Cholesky)
moved onto GLASS's `glass::warp::` sub-namespace. The end-effector frame is now **per-robot** (codegen
resolves `grid::EE_FIXED_FRAME_IDX` from the named target and injects it; `hjcd_settings.h` consumes it) —
see [`docs/source/user_guide/benchmarks/results.rst`](docs/source/user_guide/benchmarks/results.rst)
for the per-robot EE map + how to regenerate the paper sweeps.

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
compiles and runs open-world; the Python API rejects a collision-free request in that build. **Timing (2026-07-10, RTX
5090) confirms no regression** from the migration: open-world B=2000 ≈ 1.86 ms (matches pre-migration), and
the collision-free leg reports a per-mode `soft`/`hard` column (`scripts/perf/run_all_timing_sweeps.sh`).

> **Build/test gotcha:** `ninja -C build` does NOT update the imported `.so` (it's the editable copy in
> site-packages). Always rebuild with **`scripts/setup/rebuild.sh`** (or `pip install -e . --no-build-isolation`).

**Detailed working state lives in local, untracked notes** (`docs/HANDOFF.md` + `docs/open-tasks/`, gitignored —
they're agent scratch, not project artifacts). Tracked project docs: this file, `docs/development/agent_debugging_guide.md`,
`docs/source/user_guide/benchmarks/results.rst`, and the sphinx docs.
