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
Runtime robot builds and prepared-artifact loading require Python 3.10+ through `grid-rbd`.
The current validation environment is Linux/Python 3.12/CUDA 13.2; this is not a tested version matrix.

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

solver = hjcdik.Solver()                      # the compiled robot; solver.info says which
targets = solver.sample_targets(4, seed=0)    # (4, 7) reachable poses

result = solver.solve(targets, batch_size=2000, num_solutions=1, collision=None)

print("solutions per target:", result.count)
best = result[0]                               # un-padded view of the first target
print("joint configuration:", best.q)
print("position error (mm):", best.pos_err_mm)
print("orientation error (rad):", best.ori_err_rad)
```

`solve` takes one pose or a batch of poses. `batch_size` is the number of candidate configurations
explored per target, not the number of target poses. On a collision build the default policy is
`collision="hard"` (self and environment filtering against the installed world); `collision=None` is
an open-world solve. The wheel ships one compiled robot (Panda); other robots are built from a URDF
through GRiD's artifact store with the `hjcdik[build]` extra (`hjcdik build arm.urdf --ee-link
<tool joint>`, then `hjcdik.Solver("<name>")`; see
[the Python API](docs/source/api_reference/python.rst)). `solver.info` identifies the robot in use.
For a mixed-unit workflow, see the [illustrative rail-mounted UR5e](docs/source/user_guide/tutorials/rail_ur5e.md):
build/load, world replacement, checked warm starts and zero-result handling without changing the default model.
See [upgrading and verified scope](docs/source/user_guide/upgrading.md) for the move from the
earlier function API.

Target poses use:

```text
[x, y, z, qw, qx, qy, qz]
```

Target and returned pose positions are in meters; quaternions use `wxyz` order and are normalized on input.
Returned `pos_err_mm` are in **millimeters** and `ori_err_rad` are in **radians**. Check these errors against
your tolerances; the solver can return approximate candidates for unreachable targets. Collision filtering
can leave fewer valid rows than requested, including zero (`result.count`, `result.valid`). See
[the Python API](docs/source/api_reference/python.rst).

In hard/both mode, collision-aware refinement may retain a free fallback within 5 mm / 0.05 rad
when this run finds no exactly converged free candidate. A nonempty result is not an accuracy
certificate: check both errors for the same candidate against your application's tolerances.
`solver.info.ee_frame` identifies the compiled tool frame. MotionBenchMaker goals are
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

## Benchmarks and paper reproduction

The benchmark harness (`benchmark/hjcd_ik_bench.py`), the optional competitor baselines and the paper
reproduction protocol are documented in
[`docs/source/user_guide/benchmarks/reproduction.md`](docs/source/user_guide/benchmarks/reproduction.md);
measured results and the fairness study are in
[`docs/source/user_guide/benchmarks/results.rst`](docs/source/user_guide/benchmarks/results.rst).
Timing numbers are only meaningful on an otherwise idle GPU.

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

HJCD-IK supports fixed-base serial chains with 1–32 independent revolute, continuous or
prismatic joints, with arbitrary joint axes. Fixed joints may connect links and the tool frame.
Mimic joints, actuated branches and floating bases are rejected. GRiD 0.6.0 also requires at
least one revolute or continuous joint; all-prismatic chains are rejected before generation.
The fixed tool target must
attach to the last actuated joint. `solver.info.joint_names`, `joint_types`, `joint_units` and
`joint_limits` describe the order and units of `q` and `q_init` (radians or metres).
Continuous joints use a bounded sampling interval of [-pi, pi]. Mixed-joint LM step norms
and warm-start ranking use raw joint coordinates, not a unit-invariant distance metric.
GRiD supports more robot classes than this solver. See the
[custom-robot guide](docs/source/user_guide/tutorials/custom_robot.md).

`Solver("ur5e")` builds the shipped nominal-calibration UR5e preset on first use with
`hjcdik[build]`. Its sphere approximation and tool frame are documented in the guide.
For coordinate metadata, world setup, warm starts and empty-result handling with mixed
joints, run [the analytic example](examples/mixed_joint_warm_start.py).

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
