# Installation & Quickstart

HJCD-IK is a GPU-accelerated, *batched* inverse kinematics solver: it explores many candidate joint
configurations in parallel (one candidate per coarse-search block, then one candidate per LM warp) and refines the
promising ones, with optional collision avoidance. Kinematics come from [GRiD](https://github.com/A2R-Lab/GRiD)
(a per-URDF generated `grid.cuh`); the warp-scoped linear algebra comes from
[GLASS](https://github.com/A2R-Lab/GLASS).

## Requirements

- Linux, a CUDA 12.x or 13.x toolkit (`nvcc`), and an NVIDIA GPU
- CMake ≥ 3.24, a C++17 host compiler
- Python ≥ 3.9

No system libraries beyond the CUDA toolkit are needed (the former `nlohmann-json` dependency went
with the native JSON parser in October 2026; obstacle worlds are plain arrays now).

## Build

```bash
git clone https://github.com/A2R-Lab/HJCD-IK
cd HJCD-IK
bash scripts/setup/bootstrap.sh      # only the required pinned submodules
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

This builds the `_hjcdik` extension. The CUDA architecture is auto-detected for the GPU present at
configure time (`CMAKE_CUDA_ARCHITECTURES=native`). For a fresh build targeting other GPUs,
set `CUDAARCHS` (for example `CUDAARCHS="86;89" python -m pip install -e .`), or explicitly override
a cached setting with `python -m pip install -e . -Ccmake.define.CMAKE_CUDA_ARCHITECTURES="86;89"`.
The checked-in collision-enabled Panda `grid.cuh` is used by default; codegen is not needed
for that build. See the custom-robot tutorial to generate a different model.

```{tip}
**One-shot dev setup** — system deps + pinned submodules + a `.venv` + the docs toolchain +
codegen + build:  `./scripts/setup/setup_dev.sh`  (`SKIP_APT=1` / `SKIP_BUILD=1` / `SKIP_SUBMODULES=1` to
skip steps).
```

Source archives retain the implementation, codegen sources, licenses, tests, and pre-spherized
robot URDFs, but omit upstream mesh collections, upstream tests/docs, and showcase media.
Use a recursive repository clone if you need those full asset collections.

The top-level submodules are `external/GRiD` (kinematics codegen → `grid.cuh`),
`external/GLASS` (warp CUDA linear algebra), and `external/foam` (the default Panda's
pre-spherized collision model). Development setup and GPU-proof tooling require Python 3.11+;
the base package supports Python 3.9+.

## Quickstart

```python
import hjcdik

solver = hjcdik.Solver()
print("robot:", solver.info.robot, "DOF:", solver.info.dof)

# Sample a reachable target: [x, y, z, qw, qx, qy, qz]
target = solver.sample_targets(1, seed=0)[0]

# Explore a batch of candidates for this target; collision=None is an open-world solve
out = solver.solve(
    target,
    batch_size=2000,     # candidates explored in parallel
    num_solutions=4,     # result capacity per target
    collision=None,
)
r = out[0]
print("returned:", r.count)
if r.count:
    print("best position error (mm):", r.pos_err_mm.min())
    print("best orientation error (rad):", r.ori_err_rad.min())
print("joint configs shape:", r.q.shape)
```

`batch_size` counts candidate configurations for one target, not independent targets; `solve`
accepts a `(T, 7)` batch of targets and returns `(T, num_solutions, ...)` arrays, NaN-padded beyond
`count`. Targets/returned poses use meters and `wxyz` quaternions; returned errors use millimeters
and radians. Check both errors against your application's tolerances, even when `count > 0`.

For collision-free solving, install the obstacles with `solver.set_world(...)` and solve with the
default `collision="hard"`; `examples/02_collision_free_solve.py` loads a MotionBenchMaker scene.
To target a different robot or end-effector frame, see {doc}`../tutorials/custom_robot`.

## Native executable and checks

The native executable is an open-world example/export tool; use the Python API for collision controls.
It is built by CMake but is not installed as a command by the Python wheel. A separate build directory
keeps native configuration independent of the editable Python build:

```bash
cmake -S . -B build-native -DBUILD_PYTHON=OFF -DHJCDIK_BUILD_NATIVE_TESTS=ON
cmake --build build-native -j2
ctest --test-dir build-native --output-on-failure
./build-native/hjcdik --help
./build-native/hjcdik --mode=single --batch_size=2000 --num_solutions=4 --yaml_out=results.yml
```

`single` samples one reachable target; `sweep` samples `--num_targets` reachable targets.
YAML preserves the legacy keys `Batch-Size`, `IK-time(ms)`, `Pos-Error`, and `Ori-Error`.
Position errors are **millimeters**, orientation errors are **radians**. `IK-time(ms)` divides
the total solve wall time by the returned count; it is not an independently measured per-solution
latency. For performance comparisons use the benchmark harness on an idle GPU.

`from_csv` consumes the unquoted numeric CSV interchange format from TRAC-IK exports:

```text
target_id,target_px,target_py,target_pz,target_qx,target_qy,target_qz,target_qw
7,0.3,0.0,0.5,0,0,0,1
```

```bash
./build-native/hjcdik --mode=from_csv --csv_in=targets.csv --csv_out=solutions.csv
```

Column order may vary and additional columns are allowed; required names must be present.
Positions are meters and quaternions are finite and nonzero (normalized by the solver).
Whitespace and CRLF line endings are accepted. Every nonblank row is validated; the first valid
row for each integer `target_id` supplies that target. This legacy MMD mode requests **50 solutions
per unique target**, overriding `--num_solutions`, and writes `target_id,sample_id,q1,...,qN`;
it can return fewer than 50. Returned configurations are approximate IK candidates, not
collision-checked paths. Unknown options, malformed numbers/rows, and output-write failures
return nonzero status. CSV diagnostics include the input line number.
