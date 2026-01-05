# HJCD-IK: Hybrid Jacobian Coordinate Descent Inverse Kinematics

[![arXiv:2510.07514](https://img.shields.io/badge/arXiv-2510.07514-b31b1b.svg)](https://arxiv.org/abs/2510.07514)

This repository contains the code from ["HJCD-IK: GPU-Accelerated Inverse Kinematics through Batched Hybrid Jacobian Coordinate Descent"](https://arxiv.org/abs/2510.07514)

## Requirements

- NVIDIA GPU + **CUDA Toolkit 12.x**
- **Python &ge; 3.9**
- **CMake &ge; 3.23**
- **Visual Studio 2022** (Windows) or **GCC/Clang** (Linux)

## Installation
```bash
git clone https://github.com/A2R-Lab/HJCD-IK.git
cd HJCD-IK
```

HJCD-IK relies on [GRiD](https://github.com/A2R-Lab/GRiD), a GPU-accelerated library for rigid body dynamics and analytical gradients.

(Mac/Linux)
```bash
chmod +x scripts/bootstrap.sh
./scripts/boostrap.sh
```
(Windows)
```bash
.\scripts\boostrap_windows.bat
```

You can install `hjcdik` with `pip` on Python &ge; 3.9:
```bash
python -m pip install -e . --config-settings=cmake.define.HJCDIK_AUTO_CODEGEN=ON
```
<!--
## Initial Start
After installation, configure an initial GRiD header file for the robot:
```bash
cd external/GRiD
python generateGRiD.py /path/to/urdf
cd ../..
```
For testing we provide `panda` and `fetch` urdf files in `include/test_urdf`. 
-->
## Benchmark
Once done installing, run:
```bash
python benchmarks/ik_benchmark.py --skip-grid-codegen
```
To run the Panda Arm benchmark on batch sizes of `1,10,100,1000,2000`. Results are written to a `results.yml` file.

### Usage
* `--num-targets <int>`
  * How many target poses to sample. Default: `100`
* `--batches "<list>"`
  * Batch sizes to test (comma or space separated). Default: `"1,10,100,1000,2000"`
* `--num-solutions <int>`
  * How many IK solutions to return per call. Default: `1`
* `--yaml-out <path>`
  * Output result file. Default: `results.yml`
* `--urdf <path>`
  * URDF path used if running GRiD codegen. Default: `include/test_urdf/panda.urdf`
* `--skip-grid-codegen`
  * Skips creating GRiD header file and immediately runs benchmarks. Default: off
* `--seed <int>`
  * Seed for target sampling. Default: `0`

### Usage Examples
* Custom batches/targets/solutions, write a custom file:
```bash
python benchmarks/ik_benchmark.py \
  --batches "1,32,256,2048" \
  --num-targets 250 \
  --num-solutions 4 \
  --yaml-out results.yml \ 
  --skip-grid-codegen
```
* Run with GRiD code-gen using specific URDF:
```bash
python benchmarks/ik_benchmark.py --urdf include/test_urdf/panda.urdf
```

## Cite
```bibtex
@article{yasutake2025hjcd,
  title={HJCD-IK: GPU-Accelerated Inverse Kinematics through Batched Hybrid Jacobian Coordinate Descent},
  author={Yasutake, Cael and Kingston, Zachary and Plancher, Brian},
  journal={arXiv preprint arXiv:2510.07514},
  year={2025}
}
```