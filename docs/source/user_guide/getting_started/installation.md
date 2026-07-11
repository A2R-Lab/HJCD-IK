# Installation & Quickstart

HJCD-IK is a GPU-accelerated, *batched* inverse kinematics solver: it explores many candidate joint
configurations in parallel (one CUDA block per IK problem, one candidate per warp) and refines the
promising ones, with optional collision avoidance. Kinematics come from [GRiD](https://github.com/A2R-Lab/GRiD)
(a per-URDF generated `grid.cuh`); the warp-scoped linear algebra comes from
[GLASS](https://github.com/A2R-Lab/GLASS).

## Requirements

- CUDA 12.x or 13.x toolkit (`nvcc`) and an NVIDIA GPU
- CMake ≥ 3.23, a C++17 host compiler
- Python ≥ 3.9
- System header libraries: **Eigen3** and **nlohmann-json** (the collision env parser includes it)

### System dependencies (Debian/Ubuntu)

```bash
sudo apt install -y libeigen3-dev nlohmann-json3-dev
```

On other platforms install the equivalents (`eigen`, `nlohmann-json`) via your package manager.

## Build

```bash
git clone --recursive https://github.com/A2R-Lab/HJCD-IK
cd HJCD-IK
# or, if already cloned:  git submodule update --init --recursive
python -m pip install -e .
```

This builds the `_hjcdik` extension. The CUDA architecture is auto-detected for the GPU present at
configure time (`CMAKE_CUDA_ARCHITECTURES=native`); override it (e.g. `-DCMAKE_CUDA_ARCHITECTURES=86;89`)
when cross-compiling. The checked-in `grid.cuh` is used by default; set `-DHJCDIK_AUTO_CODEGEN=ON` to
regenerate it during configure.

```{tip}
**One-shot dev setup** — system deps + submodules (on our branches) + a `.venv` + the docs toolchain +
codegen + build:  `./scripts/setup/setup_dev.sh`  (`SKIP_APT=1` / `SKIP_BUILD=1` / `SKIP_SUBMODULES=1` to
skip steps).
```

The two submodules are `external/GRiD` (kinematics codegen → `grid.cuh`) and `external/GLASS`
(single-block / warp CUDA linear algebra).

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
