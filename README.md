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
- CMake 3.24 or newer
- GCC or Clang
- nlohmann-json

## Installation

Clone the repository:

```bash
git clone https://github.com/A2R-Lab/HJCD-IK.git
cd HJCD-IK
```

Run the development setup:

```bash
chmod +x scripts/setup/setup_dev.sh
./scripts/setup/setup_dev.sh
source .venv/bin/activate
```

The script initializes the required submodules, creates a virtual environment, installs dependencies,
regenerates the collision-enabled Panda model, and builds `hjcdik`.
Development setup and signed GPU-proof tooling require Python 3.11 or newer; the base package
supports Python 3.9 or newer.

If needed, convert the shell scripts to Unix line endings:

```bash
dos2unix scripts/setup/*.sh scripts/bench/*.sh
```

Verify the installation:

```bash
python - <<'PY'
import hjcdik

print("hjcdik:", hjcdik.__file__)
print("build/model:", hjcdik.build_info())
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

Install the build tools, then build the package and development dependencies against the committed,
collision-enabled `grid.cuh`:

```bash
python -m pip install --upgrade \
  pip setuptools wheel cmake ninja scikit-build-core pybind11
python -m pip install -e ".[dev]" --no-build-isolation
```

GRiD code generation is optional for the default Panda build. Install `.[codegen]` and use the
collision-enabled command below only when changing the URDF or end-effector target.

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

Each call solves **one target**. `batch_size` is the number of candidate configurations,
not the number of target poses. A wheel contains one compiled robot; use `build_info()`
to confirm its identity. See [upgrading and verified scope](docs/source/user_guide/upgrading.md)
for changes to collision defaults, native ownership, and error handling.

Target poses use:

```text
[x, y, z, qw, qx, qy, qz]
```

Target and returned pose positions are in meters; quaternions use `wxyz` order and are normalized on input.
Returned `pos_errors` are in **millimeters** and `ori_errors` are in **radians**. Check these errors against
your tolerances; the solver can return approximate candidates for unreachable targets. Collision filtering
can return fewer solutions, including zero. See [the Python API](docs/source/api_reference/python.rst).

In hard/both mode, collision-aware refinement may retain a free fallback within 5 mm / 0.05 rad
when this run finds no exactly converged free candidate. A nonempty result is not an accuracy
certificate: check both errors for the same candidate against your application's tolerances.
`build_info()["ee_target"]` identifies the compiled tool frame. MotionBenchMaker goals are
hand-frame poses, 105 mm from the default Panda TCP; the collision example converts them explicitly.
The benchmark's `--target-mode goal` requires a `panda_hand_joint` build, whereas `--target-mode cylinder`
uses the default `panda_grasptarget_hand` build. Do not pass hand poses directly to a TCP build.

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

The default Panda model keeps fixed finger-joint origins at +/-40 mm in the hand frame.
Foam supplies sphere shapes, but their placement uses this kinematic URDF. The frozen paper
reference uses +/-65 mm finger origins and is deliberately retained for historical comparisons.
The benchmark's `--collision-validation-model paper` (default) selects that legacy reference;
`--collision-validation-model hjcd` selects an independent URDF-derived check of the current
geometry. This flag changes only post-hoc validation, never the solver's compiled robot.
Both checks are environment-only; the solver's hard/both modes additionally check self-collision.
CSV/YAML collision results have a `.metadata.json` sidecar identifying the selected model,
source hashes, finger origins, and compiled-header identity.

Benchmark summaries count attempted queries, including zero-output failures, rather than returned
solutions. Configuration dumps retain the target/frame and empty records for independent FK scoring.
See [the current timing and evidence protocol](docs/development/timing_gate.md).

## Examples

Run the included examples:

```bash
python examples/01_open_world_solve.py
python examples/02_collision_free_solve.py
python examples/03_batch_sweep.py
```

## Tests

For the full test suite, use the collision-enabled Panda build above and install `.[dev,codegen]`.
GPU-proof receipt generation and the one-shot development setup require Python 3.11 or newer.

Run:

```bash
python -m pytest tests/ -v
```

When adding or renaming tests, regenerate and commit the proof manifest before recording a receipt:

```bash
python scripts/setup/update_gpu_proof_manifest.py
```

The GPU-proof policy binds the full test list, solver sources, executable docs/examples, build/codegen scripts, and dependency
gitlinks. A scoped `pytest -k ...` run is useful for diagnosis but cannot certify the full suite.

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

Select the policy explicitly with `collision_mode="hard"`, `"soft"`, or `"both"`:

```python
result = hjcdik.generate_solutions(..., collision_free=True, collision_mode="hard")
```

The benchmark also accepts `--collision-mode`; `HJCD_CC_MODE` remains the default of that flag for
benchmark scripts only (the Python API and native solver never read it).

- `hard` (default): filters self- and environment-colliding solutions; the result may contain fewer
  than `num_solutions`, including zero
- `soft`: ranks solutions using an environment penetration cost but does not guarantee collision freedom
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

Use an isolated checkout/virtual environment: this workflow regenerates robot headers and rebuilds
the extension for several frames. Set `OUT_DIR` to a **new, nonexistent directory** for each run;
the harness refuses to overwrite earlier evidence. The commands below reproduce the protocol,
not a guarantee of the published numerical results on different hardware or code revisions.
The completed [October 5 same-machine A/B](docs/development/evidence/audit_timing_2026-10-05/README.md)
is separate from the historical paper/competitor tables; a fresh full comparison remains outstanding.

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

HJCD-IK supports fixed-base serial chains with 1–32 independent revolute or continuous
joints rotating around local +Z; fixed joints may connect links and the tool frame.
Prismatic, mimic, branched, floating-base, and other-axis models are rejected.
GRiD supports more robot classes than this solver. See the
[custom-robot guide](docs/source/user_guide/tutorials/custom_robot.md).

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

## Funding Acknowledgement
This material is based upon work supported by the National Science Foundation (under Award [2411369](https://www.nsf.gov/awardsearch/show-award/?AWD_ID=2411369)). Any opinions, findings, conclusions, or recommendations expressed in this material are those of the authors and do not necessarily reflect those of the funding organizations.
