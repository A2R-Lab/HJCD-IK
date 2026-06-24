# HJCD-IK: Hybrid Jacobian Coordinate Descent Inverse Kinematics

[![arXiv:2510.07514](https://img.shields.io/badge/arXiv-2510.07514-b31b1b.svg)](https://arxiv.org/abs/2510.07514)

This repository contains the code from ["HJCD-IK: GPU-Accelerated Inverse Kinematics through Batched Hybrid Jacobian Coordinate Descent"](https://arxiv.org/abs/2510.07514)

## Requirements

- NVIDIA GPU + **CUDA Toolkit 12.x or 13.x**
- **Python &ge; 3.9**
- **CMake &ge; 3.23**
- **Visual Studio 2022** (Windows) or **GCC/Clang** (Linux)
- System header libraries **Eigen3** and **nlohmann-json**. On Debian/Ubuntu:
  ```bash
  sudo apt install -y libeigen3-dev nlohmann-json3-dev
  ```
  (`scripts/setup/setup_dev.sh` installs these for you on apt-based systems.)

## Installation
```bash
git clone https://github.com/A2R-Lab/HJCD-IK.git
cd HJCD-IK
```

HJCD-IK relies on [GRiD](https://github.com/A2R-Lab/GRiD), a GPU-accelerated library for rigid body dynamics and analytical gradients.

(Linux)
```bash
chmod +x scripts/setup/bootstrap.sh
./scripts/setup/bootstrap.sh
```
Note: may need to run ```dos2unix scripts/setup/bootstrap.sh ``` before ```./scripts/setup/bootstrap.sh``` first

(Windows)
```bash
.\scripts\bootstrap_windows.bat
```

You can install `hjcdik` with `pip` on Python &ge; 3.9:
```bash
python -m pip install -e .
```

### Optional: competitor baselines (PyRoki / cuRobo)
The base install is lightweight and needs none of the baselines. To benchmark HJCD-IK against the
paper's competitors, install them with the helper (each stage is skippable — some are heavy):
```bash
./scripts/setup/install_baselines.sh                 # PyRoki + cuRobo
SKIP_CUROBO=1 ./scripts/setup/install_baselines.sh   # PyRoki only
```
See [`docs/source/user_guide/benchmarks/baselines.md`](docs/source/user_guide/benchmarks/baselines.md) for the full install/run guide and reproduction coverage.

## Using different robots
At installation, HJCD-IK creates a new GRiD header file for the Franka Panda Arm and sets `panda_grasptarget_hand` as its end-effector flange. To use a different robot, you must first create a new `grid.cuh` header file using:
```bash
python scripts/codegen/generate_grid.py <PATH_TO_URDF> -t <FIXED_TARGET_NAME>
```
* `PATH_TO_URDF`: the path to the new robot URDF file
* `FIXED_TARGET_NAME`: the name of the robot end-effector flange (e.g. Franka: `panda_grasptarget_hand`)
  * Note: GRiD prints out possible fixed joint names found (if any) during code generation

## Benchmark
To run IK benchmark, use:
```bash
python benchmark/hjcd_ik_bench.py --skip-grid-codegen
```
which performs IK using the Panda Arm with batches of `1, 10, 100, 1000, 2000`. Results are written to a `results.yml`.

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
* `--grid-target <FIXED_TARGET_NAME>`
  * The name of the robot end-effector flange offset
* `--skip-grid-codegen`
  * Skips creating GRiD header file and immediately runs benchmarks. Default: off
* `--seed <int>`
  * Seed for target sampling. Default: `0`

### Usage Examples
* Custom batches/targets/solutions, out file name:
```bash
python benchmark/hjcd_ik_bench.py \
  --batches "1,32,256,2048" \
  --num-targets 250 \
  --num-solutions 4 \
  --yaml-out results.yml \ 
  --skip-grid-codegen
```
* To generate a new GRiD header on a different robot, run:
```bash
python benchmark/hjcd_ik_bench.py --urdf include/test_urdf/fetch.urdf
```

### Collision-Free Benchmark
To run collision-free benchmark on the [MotionBenchMaker](https://github.com/KavrakiLab/motion_bench_maker) dataset, run:
```bash
python benchmark/hjcd_ik_bench.py --skip-grid-codegen --collision-free --problems-json tests/mb_problems.json --problem-set bookshelf_thin_panda
```

### Collision-Free Benchmark Additional Usage
* `--collision-free`
  * Enable collision filter on solutions.
* `--problems-json <path>` 
  * Path to json problem file for collision-free benchmarking.
* `--problem-set <str>`
  * Problem set within json file to run benchmarking.
* `--problem-idx <int>`
  * Run collision-free benchmarking on specific problem index within problem set.
  
## Creating Collision Environments
Collision environments are specified in the Motion Benchmarker-style JSON problem format. Each problem contains a `goal_pose`, `start` configuration, `world_frame`, and an optional `obstacles` field. Examples of environments can be found in the `tests` folder.

Obstacles are grouped by primitive type. Currently only `cuboid` and `cylinder` primitives are supported.

### Cuboid obstacles
Cuboids are specified under `obstacles.cuboid`:

```json
"cuboid": {
  "cube_robot_stand": {
    "dims": [0.30, 0.25, 0.80],
    "pose": [-0.05, 0.00, -0.40, 1, 0, 0, 0]
  }
}
```

Each cuboid requires:
* `dims`: `[x, y, z]` sides lengths in meters
* `pose`: `[x, y, z, qw, qx, qy, qz]` in the problem's `world_frame`

### Cylinder obstacles
Cylinders are specificed under `obstacles.cylinder`:

```json
"cylinder": {
  "goal_post": {
    "radius": 0.035,
    "height": 0.24,
    "pose": [0.35, 0.15, 0.12, 1, 0, 0, 0]
  }
}
```

Each cylinder requires:
* `radius`: cylinder radius in meters
* `height`: cylinder height in meters
* `pose`: `[x, y, z, qw, qx, qy, qz]` in the problem's `world_frame`

### Additional Notes
* HJCD-IK currently only supports robots using revolute, prismatic, and fixed joints without any closed kinematic loops.
* Collision-Free support is currently limited to the Franka Panda and Fetch Arms with support for additional and custom robots coming soon!

## Cite
Please cite HCJD-IK if you found this work useful:
```bibtex
@article{yasutake2025hjcd,
  title={HJCD-IK: GPU-Accelerated Inverse Kinematics through Batched Hybrid Jacobian Coordinate Descent},
  author={Yasutake, Cael and Kingston, Zachary and Plancher, Brian},
  journal={arXiv preprint arXiv:2510.07514},
  year={2025}
}
```