# CLAUDE.md — orientation for agents (and humans) on HJCD-IK

**HJCD-IK** = **Hybrid Jacobian Coordinate Descent Inverse Kinematics**: a GPU-accelerated, *batched*
IK solver (paper: [arXiv:2510.07514](https://arxiv.org/abs/2510.07514)). It generates many candidate IK
solutions in parallel for a 6-DOF end-effector target, with optional collision avoidance, built on top of
[GRiD](https://github.com/A2R-Lab/GRiD) (robot kinematics codegen) and
[GLASS](https://github.com/A2R-Lab/GLASS) (single-block CUDA linear algebra).

> **Before changing the kernel or codegen, read [`docs/agent_debugging_guide.md`](docs/agent_debugging_guide.md).**
> It is the runbook for HJCD-IK's recurring traps: stale `grid.cuh`, `FLANGE_IDX`/target mismatch,
> warp-vs-block sync in the solver loop, and submodule init.

## Mental model

**One CUDA block per IK problem; warp-per-candidate inside.** The solver is **warp-scoped throughout**
(`warp_id = threadIdx.x >> 5`, `lane = threadIdx.x & 31`), not block-scoped — this is the core performance
contract. Two phases (`src/hjcd_kernel.cu`):
1. **Coarse search** (`coarse_search`): random restarts + greedy per-joint coordinate descent to get near the target.
2. **LM refine** (`solve_lm_batched` / `lm_tuner`): single-warp Levenberg–Marquardt — build the 6×N geometric
   Jacobian (cross-products), form & solve the normal equations `(JᵀJ + λ·diag)Δq = Jᵀr` via a hand-rolled
   `__shfl` warp-Cholesky, with dogleg/line-search backtracking.

Forward kinematics produces the **world-frame joint transforms** `s_jointXforms[16·jid]` (4×4 each); the EE
pose error is computed as a **quaternion** error (`mat_to_quat` / `quat_err_rotvec`). For Panda: `N = 7`
joints, `EE_IDX = 7`, `FLANGE_IDX = 8`, `NX = 9` stored frames.

## Key files

| Path | What it is |
|---|---|
| `src/hjcd_kernel.cu` | The solver: coarse search + LM refine, all warp-scoped. **The file you'll edit most.** |
| `include/hjcd_settings.h` | `HJCDSettings<T>` (epsilon/nu/k_max), `mat4_mul`, `#include "grid.cuh"`, `N = grid::NUM_JOINTS`. |
| `include/test_cuh/grid.cuh` | **Generated** GRiD kinematics header (FK, robot model). Do **not** hand-edit. |
| `external/GRiD/` | Submodule: GRiD codegen (emits `grid.cuh` from a URDF). |
| `external/GLASS/` | Submodule: GLASS single-block / warp linear algebra. |
| `src/robots/{panda,fetch}.cuh` | Per-robot collision spheres + fixed transforms (Panda/Fetch only). |
| `src/collision/` | pRRTC per-block collision checking. |
| `src/pybind_module.cpp` | Python bindings → `generate_solutions`, `sample_targets`, `num_joints`. |
| `benchmark/ik_benchmark.py` | Benchmark harness: solved-rate, position/orientation error, timing. |
| `tests/{mb,wall}_problems.json` | MotionBenchMaker / wall collision problem sets. |

## Build & test

CMake 3.23+ / CUDA 12.x / pybind11 (scikit-build-core). GRiD codegen runs at configure time when enabled.

```bash
sudo apt install -y libeigen3-dev nlohmann-json3-dev   # system header deps (Eigen3 + nlohmann-json)
git submodule update --init --recursive          # GRiD + GLASS
python -m pip install -e .                        # builds the _hjcdik extension (CUDA arch auto-detected)
python benchmark/ik_benchmark.py --skip-grid-codegen   # run the solver
```

`./scripts/setup_dev.sh` does all of the above (system deps + submodules on our branches + venv + codegen + build).

Python API:
```python
from hjcdik import generate_solutions, sample_targets, num_joints
targets = sample_targets(num_targets=10, seed=0)         # list of [x,y,z, qw,qx,qy,qz]
out = generate_solutions(targets[0], batch_size=2000, num_solutions=4)
# out = {joint_config, pose, pos_errors, ori_errors, count}
```

## Conventions / discipline

- **Never hand-edit `grid.cuh`.** It is GRiD codegen output. To change the robot or EE target, run
  `python scripts/generate_grid.py <urdf> -t <target_frame>` and rebuild. Robot constants
  (`NUM_JOINTS`, topology counts) are baked per-URDF — read them from the generated symbols, never hardcode.
- **`FLANGE_IDX` discipline.** The fixed EE target (`panda_grasptarget_hand`) and its index must agree across
  codegen, the kernel, and any benchmark problem. A mismatch silently solves to the wrong frame.
- **Warp-locality is the performance contract.** New math must stay warp-scoped (`__shfl_*_sync`, `__syncwarp`).
  Do not drop the solver onto block-scoped primitives.
- **Short, single-line commit messages; no `Co-Authored-By` footer.**

## Integration in progress — `grid-glass-integration` branch

This branch re-bases HJCD-IK onto the latest GRiD (`modernizing-tests`) + GLASS (`main`) for modularity and
upstreamable performance. The bespoke Panda-only FK (`X_warp` / `X_single_thread`, currently hand-written into
the vendored `grid.cuh`) is being replaced by GRiD's stock warp FK (`grid::ee_pose_inner_warp`), and the
hand-rolled math (`mat4_mul`, warp reduce, warp Cholesky) moved onto a new `glass::warp::` sub-namespace.
The end-effector frame is now **per-robot** (codegen resolves `grid::EE_FIXED_FRAME_IDX` from the named
target and injects it; `hjcd_settings.h` consumes it) — see [`docs/reproduce_paper_sweeps.md`](docs/reproduce_paper_sweeps.md)
for the per-robot EE map + how to regenerate the paper sweeps.

> **Build/test gotcha:** `ninja -C build` does NOT update the imported `.so` (it's the editable copy in
> site-packages). Always rebuild with **`scripts/rebuild.sh`** (or `pip install -e . --no-build-isolation`).

**Detailed working state lives in local, untracked notes** (`docs/HANDOFF.md` + `docs/open-tasks/`, gitignored —
they're agent scratch, not project artifacts). Tracked project docs: this file, `docs/agent_debugging_guide.md`,
`docs/reproduce_paper_sweeps.md`, and the sphinx docs.
