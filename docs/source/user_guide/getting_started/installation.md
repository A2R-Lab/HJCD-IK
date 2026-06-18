# Installation

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
`scripts/setup_dev.sh` installs these automatically on apt-based systems (set `SKIP_APT=1` to skip).

## Build
```bash
git clone --recursive https://github.com/A2R-Lab/HJCD-IK
cd HJCD-IK
# or, if already cloned:  git submodule update --init --recursive
python -m pip install -e .
```

This builds the `_hjcdik` extension. By default the CUDA architecture is auto-detected for the GPU
present at configure time (`CMAKE_CUDA_ARCHITECTURES=native`); override it (e.g.
`-DCMAKE_CUDA_ARCHITECTURES=86;89`) when cross-compiling or building on a host without a GPU. GRiD
codegen (`grid.cuh`) is used from the checked-in header by default; set `-DHJCDIK_AUTO_CODEGEN=ON` to
regenerate it during configure.

> One-shot dev setup (system deps + submodules on our branches + venv + codegen + build):
> `./scripts/setup_dev.sh`

## Submodules
- `external/GRiD` — robot-kinematics code generator (emits `grid.cuh`).
- `external/GLASS` — single-block / warp CUDA linear algebra.
