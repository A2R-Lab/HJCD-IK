# HJCD-IK: Hybrid Jacobian Coordinate Descent Inverse Kinematics

[![arXiv:2510.07514](https://img.shields.io/badge/arXiv-2510.07514-b31b1b.svg)](https://arxiv.org/abs/2510.07514)

This repository contains the implementation from
[“HJCD-IK: GPU-Accelerated Inverse Kinematics through Batched Hybrid Jacobian Coordinate Descent”](https://arxiv.org/abs/2510.07514).

HJCD-IK is a GPU-accelerated, sampling-based hybrid inverse kinematics solver for generating one or
more robot configurations for a target end-effector pose.

## Requirements

- Linux
- NVIDIA GPU
- CUDA Toolkit 12.x or 13.x
- Python 3.9 or newer
- CMake 3.23 or newer
- GCC or Clang
- Eigen3
- nlohmann-json

## Installation

Clone the repository:

```bash
git clone --recurse-submodules https://github.com/A2R-Lab/HJCD-IK.git
cd HJCD-IK
```

Run the development setup:

```bash
chmod +x scripts/setup/setup_dev.sh
./scripts/setup/setup_dev.sh
source .venv/bin/activate
```

The script initializes the required submodules, creates a virtual environment,
installs dependencies, generates the robot model, and builds `hjcdik`.

If needed, convert the shell scripts to Unix line endings:

```bash
dos2unix scripts/setup/*.sh scripts/bench/*.sh
```

Verify the installation:

```bash
python - <<'PY'
import hjcdik

print("hjcdik:", hjcdik.__file__)
print("robot DoF:", hjcdik.num_joints())
PY
```

### Manual build

Initialize the required submodules:

```bash
./scripts/setup/bootstrap.sh
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade \
  pip \
  setuptools \
  wheel \
  cmake \
  ninja \
  scikit-build-core

python -m pip install \
  numpy \
  scipy \
  sympy \
  beautifulsoup4 \
  lxml \
  pytest
```

Generate the Panda model:

```bash
python scripts/codegen/generate_grid.py \
  csrc/urdf/panda.urdf \
  -t panda_grasptarget_hand
```

Build the package:

```bash
python -m pip install -e . --no-build-isolation
```

## Quick Start

```python
import hjcdik

target = hjcdik.sample_targets(num_targets=1, seed=0)[0]

result = hjcdik.generate_solutions(
    target,
    batch_size=2000,
    num_solutions=1,
)

print("solutions:", result["count"])
print("joint configurations:", result["joint_config"])
print("position errors:", result["pos_errors"])
print("orientation errors:", result["ori_errors"])
```

Target poses use:

```text
[x, y, z, qw, qx, qy, qz]
```

Position is in meters and quaternions use `wxyz` order.

## Collision-Enabled Build

Generate the Panda collision model:

```bash
python scripts/codegen/generate_grid.py \
  csrc/urdf/panda.urdf \
  -t panda_grasptarget_hand \
  --collision \
  --spherized-urdf \
  external/foam/assets/panda/smaller_panda_spherized.urdf
```

Rebuild:

```bash
python -m pip install -e . --no-build-isolation
```

After any code-generation change, rebuild with:

```bash
bash scripts/setup/rebuild.sh
```

Note: the tests and collision-free example require a collision-enabled build.

## Examples

Run the included examples:

```bash
python examples/01_open_world_solve.py
python examples/02_collision_free_solve.py
python examples/03_batch_sweep.py
```

## Tests

For the full test suite, use the collision-enabled Panda build above.

Run:

```bash
python -m pytest tests/ -v
```

Run one test file:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest tests/test_fk_equivalence.py -v
```

## HJCD-IK Benchmark

Run the default HJCD benchmark:

```bash
python benchmark/hjcd_ik_bench.py \
  --skip-grid-codegen
```

This runs 100 targets with batch sizes:

```text
1, 10, 100, 1000, 2000
```

and writes results to:

```text
results.yml
```

### Common options

```text
--num-targets <int>
--batches "<list>"
--num-solutions <int>
--yaml-out <path>
--urdf <path>
--grid-target <name>
--skip-grid-codegen
--seed <int>
```

Example:

```bash
python benchmark/hjcd_ik_bench.py \
  --batches "1,32,256,2048" \
  --num-targets 250 \
  --num-solutions 4 \
  --yaml-out results.yml \
  --skip-grid-codegen
```

## Collision-Free Benchmark

Run the Panda MotionBenchMaker benchmark:

```bash
python benchmark/hjcd_ik_bench.py \
  --skip-grid-codegen \
  --collision-free \
  --problems-json tests/mb_problems.json \
  --problem-set box_panda \
  --batches "1,10,100,1000,2000"
```

Collision modes are selected with `HJCD_CC_MODE`:

```bash
HJCD_CC_MODE=soft python benchmark/hjcd_ik_bench.py ...
HJCD_CC_MODE=hard python benchmark/hjcd_ik_bench.py ...
HJCD_CC_MODE=both python benchmark/hjcd_ik_bench.py ...
```

- `soft`: biases solutions away from environment collisions
- `hard`: filters self- and environment-colliding solutions
- `both`: combines both modes

## Optional Baselines

The paper benchmark can also run:

- PyRoki
- cuRobo v2
- IKFlow
- TRAC-IK

Install available baselines:

```bash
./scripts/setup/install_baselines.sh
```

Skip individual solvers when needed:

```bash
SKIP_CUROBO=1 ./scripts/setup/install_baselines.sh
SKIP_PYROKI=1 ./scripts/setup/install_baselines.sh
SKIP_IKFLOW=1 ./scripts/setup/install_baselines.sh
SKIP_TRACIK=1 ./scripts/setup/install_baselines.sh
```

Notes:

- cuRobo requires a compatible `cuda-core` backend.
- IKFlow requires model weights under `benchmark/assets/ikflow/weights/`.
- TRAC-IK requires additional native dependencies.

See
[`docs/source/user_guide/benchmarks/results.rst`](docs/source/user_guide/benchmarks/results.rst)
for detailed baseline instructions.

## Reproducing the Paper Benchmarks

### HJCD-only benchmark (open-world and collision-free)

```bash
HJCD_REGEN=1 \
SKIP_PYROKI=1 \
SKIP_CUROBO=1 \
SKIP_IKFLOW=1 \
./scripts/bench/run_paper_experiments.sh
```

### All installed solvers (open-world and collision-free)

```bash
HJCD_REGEN=1 \
./scripts/bench/run_paper_experiments.sh
```

### Tables I–IV, including Fetch, DoF scaling, and MMD

```bash
HJCD_REGEN=1 \
RUN_FETCH=1 \
RUN_DOF=1 \
RUN_MMD=1 \
./scripts/bench/run_paper_experiments.sh
```

Results are written to:

```text
benchmark/results/
```

Use `HJCD_REGEN=1` when running the paper benchmarks to ensure that HJCD-IK is rebuilt for the correct robot and end-effector frame.

After running the paper harness, restore the collision-enabled Panda build if
you plan to run collision examples or tests:

```bash
python scripts/codegen/generate_grid.py \
  csrc/urdf/panda.urdf \
  -t panda_grasptarget_hand \
  --collision \
  --spherized-urdf \
  external/foam/assets/panda/smaller_panda_spherized.urdf

python -m pip install -e . --no-build-isolation
```

Benchmark timings depend on the GPU and system load. Run timing experiments on
an otherwise idle GPU.

## Using a Different Robot

Generate a robot-specific model:

```bash
python scripts/codegen/generate_grid.py \
  <PATH_TO_URDF> \
  -t <FIXED_TARGET_NAME>
```

Example:

```bash
python scripts/codegen/generate_grid.py \
  csrc/urdf/fetch.urdf \
  -t ee_fixed
```

Then rebuild:

```bash
python -m pip install -e . --no-build-isolation
```

HJCD-IK supports revolute, prismatic, and fixed joints.

### Collision checking for a custom robot

Generate collision spheres from the URDF:

```bash
python scripts/codegen/generate_grid.py \
  path/to/robot.urdf \
  -t end_effector_fixed_joint \
  --collision \
  --collision-res 0.02
```

Or use a pre-spherized foam URDF:

```bash
python scripts/codegen/generate_grid.py \
  path/to/robot.urdf \
  -t end_effector_fixed_joint \
  --collision \
  --spherized-urdf path/to/robot_spherized.urdf
```

## Collision Environments

Collision environments use a MotionBenchMaker-style JSON format.

Each problem may contain:

```text
goal_pose
start
world_frame
obstacles
```

Examples are available in:

```text
tests/mb_problems.json
```

Supported obstacle types are:

- `sphere`
- `cuboid`
- `cylinder`

### Cuboid

```json
"cuboid": {
  "box": {
    "dims": [0.30, 0.25, 0.80],
    "pose": [-0.05, 0.00, -0.40, 1, 0, 0, 0]
  }
}
```

### Cylinder

```json
"cylinder": {
  "post": {
    "radius": 0.035,
    "height": 0.24,
    "pose": [0.35, 0.15, 0.12, 1, 0, 0, 0]
  }
}
```

### Sphere

```json
"sphere": {
  "ball": {
    "radius": 0.05,
    "pose": [0.40, 0.10, 0.30, 1, 0, 0, 0]
  }
}
```

All poses use:

```text
[x, y, z, qw, qx, qy, qz]
```

## Citation

```bibtex
@inproceedings{yasutake2026hjcdik,
  title     = {{HJCD-IK}: {GPU}-Accelerated Inverse Kinematics through Batched Hybrid Jacobian Coordinate Descent},
  author    = {Yasutake, Cael and Liu, Andrew H. and Kingston, Zachary and Plancher, Brian},
  booktitle = {2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026},
  note      = {arXiv:2510.07514}
}
```

## License

HJCD-IK is released under the [MIT License](LICENSE).