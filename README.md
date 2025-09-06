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

Outputs results to hjcd_ik_per_batch.yml