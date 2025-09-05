# GLOBE_IK

CUDA-based inverse kinematics solver built on [GRiD](https://github.com/A2R-Lab/GRiD).

## Requirements

- CUDA Toolkit 12.x
- CMake &ge; 3.22
- yaml-cpp
- pybind11
- Visual Studio 2022 (Windows) or GCC/Clang (Linux)

## Clone
```bash
git clone --recurse-submodules https://github.com/A2R-Lab/GLOBE_IK.git
cd GLOBE_IK
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
.\build\Release\globeik.exe
```

### Linux
```bash
./build/globeik
```

Outputs results to globeik_ik_per_batch.yml
