# Installation & Quickstart

HJCD-IK is a GPU-accelerated, *batched* inverse kinematics solver: it explores many candidate joint
configurations in parallel (one CUDA block per IK problem, one candidate per warp) and refines the
promising ones, with optional collision avoidance. Kinematics come from [GRiD](https://github.com/A2R-Lab/GRiD)
(a per-URDF generated `grid.cuh`); the warp-scoped linear algebra comes from
[GLASS](https://github.com/A2R-Lab/GLASS).

## Requirements

- CUDA 12.x or 13.x toolkit (`nvcc`) and an NVIDIA GPU
- CMake ≥ 3.24, a C++17 host compiler
- Python ≥ 3.9
- System header library: **nlohmann-json** (the collision environment parser includes it)

### System dependencies (Debian/Ubuntu)

```bash
sudo apt install -y nlohmann-json3-dev
```

On other platforms install `nlohmann-json` via your package manager.

## Build

```bash
git clone --recursive https://github.com/A2R-Lab/HJCD-IK
cd HJCD-IK
# or, if already cloned:  git submodule update --init --recursive
python -m pip install -e .
```

This builds the `_hjcdik` extension. The CUDA architecture is auto-detected for the GPU present at
configure time (`CMAKE_CUDA_ARCHITECTURES=native`). For a fresh build targeting other GPUs,
set `CUDAARCHS` (for example `CUDAARCHS="86;89" python -m pip install -e .`), or explicitly override
a cached setting with `python -m pip install -e . -Ccmake.define.CMAKE_CUDA_ARCHITECTURES="86;89"`.
The checked-in collision-enabled Panda `grid.cuh` is used by default; codegen is not needed
for that build. See the custom-robot tutorial to generate a different model.

```{tip}
**One-shot dev setup** — system deps + pinned submodules + a `.venv` + the docs toolchain +
codegen + build:  `./scripts/setup/setup_dev.sh`  (`SKIP_APT=1` / `SKIP_BUILD=1` / `SKIP_SUBMODULES=1` to
skip steps).
```

Source archives retain the implementation, codegen sources, licenses, tests, and pre-spherized
robot URDFs, but omit upstream mesh collections, upstream tests/docs, and showcase media.
Use a recursive repository clone if you need those full asset collections.

The top-level submodules are `external/GRiD` (kinematics codegen → `grid.cuh`),
`external/GLASS` (warp CUDA linear algebra), and `external/foam` (the default Panda's
pre-spherized collision model). Development setup and GPU-proof tooling require Python 3.11+;
the base package supports Python 3.9+.

## Quickstart

```python
from hjcdik import generate_solutions, sample_targets, num_joints

print("DOF:", num_joints())

# Sample a reachable target: [x, y, z, qw, qx, qy, qz]
target = sample_targets(num_targets=1, seed=0)[0]

# Generate a batch of candidate IK solutions
out = generate_solutions(
    target,
    batch_size=2000,     # candidates explored in parallel
    num_solutions=4,     # distinct solutions to return
)
print("returned:", out["count"])
print("best position error:", out["pos_errors"].min())
print("joint configs shape:", out["joint_config"].shape)
```

For collision-free solving, pass `collision_free=True` with a MotionBenchMaker problem set — see the
{doc}`../benchmarks/results` page (runnable examples + benchmarks). To target a different robot or
end-effector frame, see {doc}`../tutorials/custom_robot`.
