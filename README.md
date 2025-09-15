# HJCD-IK (Hybrid Jacobian Coordinate Descent)

CUDA-based inverse kinematics solver built on [GRiD](https://github.com/A2R-Lab/GRiD).

## Requirements

- CUDA Toolkit 12.x
- CMake &ge; 3.22
- yaml-cpp
- Visual Studio 2022 (Windows) or GCC/Clang (Linux)

## Clone
```bash
git clone --recurse-submodules https://github.com/A2R-Lab/HJCD-IK.git
cd HJCD-IK
```

## Create GRiD header
```bash
cd external/GRiD
python generateGRiD.py /path/to/urdf
cd ../..
```
### Test .cuh files
test.cuh files are provided for Franka Panda and Fetch Arm if you would like to build and run solver. Uncomment the includes in hjcd_kernel.cu to run.

## Build
### Windows
```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE="C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake" -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### Linux
```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

## Run
### Windows
```bash
.\build\Release\hjcdik.exe
```

### Linux
```bash
./build/hjcdik
```

## Usage

This executable (`./build/hjcdik`) runs the GPU-accelerated IK solver on the Franka Panda or Fetch Robot Arm.
It supports three modes of operation:
- **single**: Run on one random target pose.
- **sweep**: Run on many random target poses.
- **from_csv**: Replay a set of target poses from an existing CSV file.

```bash
./build/hjcdik [OPTIONS]
```

### Options
- **`--mode=single|sweep|from_csv`**
- **`--batch_size=N`**
  - Number of IK seeds
- **`--num_solutions=S`**
  - Number of final IK solutions
- **`--num_targets=T`**
  - Number of random targets to evaluate (only used in sweep mode)
- **`--yaml_out=FILE.yml`**
  - Path for YAML summary output
- **`--csv_in=FILE.csv`**
  - Input CSV with target poses (only in from_csv mode)
- **`--csv_out=FILE.csv`**
  - Output CSV with per-solution results (only in from_csv mode)
- **`-h`, `--help`**
  - Print help and exit