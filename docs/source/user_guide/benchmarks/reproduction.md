# Benchmarks and paper reproduction

How to run the HJCD-IK benchmark harness, install the optional competitor baselines, and reproduce the
paper workloads. Current-code reruns and historical reproduction are different activities (see below).
Measured numbers live on [the results page](results.rst). Latency is only meaningful on an
otherwise idle GPU; success-rate collection may share the GPU.

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
```
The harness uses the historical sampling seed internally; it has no `--seed` option.
For explicit target control, generate a set with `benchmark/gen_targets.py --seed ...`
and pass `--filtered-targets`, ensuring its tool frame matches the compiled model.

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
  --target-mode cylinder \
  --problems-json tests/mb_problems.json \
  --problem-set box_panda \
  --batches "1,10,100,1000,2000"
```
This is the paper-continuity cylinder-snapped target protocol for the default
`panda_grasptarget_hand` build. Dataset `goal_pose` targets instead require
`--target-mode goal` and a `panda_hand_joint` build. Do not mix the two frames.

Select the policy explicitly with `collision="hard"`, `"soft"`, or `"both"`:

```python
result = solver.solve(..., collision="hard")
```

The benchmark accepts the same policy through `--collision-mode`. Nothing reads an environment variable.

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
[the results page](results.rst)
for detailed baseline instructions.

## Historical paper-code anchor

The local annotated tag `iros-2026` is labeled “IROS 2026 version” and resolves to
`637f4476715ea37c5f89464e2605d4754ea99a21`. It retains the earlier function API,
`benchmark/ik_benchmark.py`, `include/test_urdf/` and the old collision implementation.
The tag is the available historical code anchor, not proof that every published
table was collected with that exact tree and dependency environment.

Inspect/rebuild it only in a separate checkout and virtual environment. Do not
replace the current checkout or use current bootstrap scripts on that revision:

```bash
git clone https://github.com/A2R-Lab/HJCD-IK.git hjcd-paper
cd hjcd-paper
git checkout --detach 637f4476715ea37c5f89464e2605d4754ea99a21
git submodule update --init --recursive
git submodule status --recursive
```

Its GRiD gitlink is `0a6c18ee5e1b69e21ccbde4ca2b012176b88a88f`, whose nested pins are
GRiDCodeGenerator `272f66c6512a712a6487ece45c171ece927569ff`, RBDReference
`c442c17a7b108a7a2ccab3a3e4f317b1fa376be7`, and URDFParser
`2c042b4c5e1cafcdf69d2643fc48482d2d9efb95`. However, the tag's `scripts/bootstrap.sh`
explicitly overrides GRiDCodeGenerator to `78312a66b50e623515d654c1ba983ff471237adf`.
Record which recipe is used; the gitlink-only and bootstrap configurations are not
identical. Inspect that script before running it: it also removes nested checkout
directories. No such historical bootstrap was run in this closeout.

After resolving that generator choice and installing the tag's declared build
dependencies in its own environment, its historical entry points are:

```bash
python -m pip install -e .
python benchmark/ik_benchmark.py --skip-grid-codegen --num-targets 100 --batches 1,10,100,1000,2000
```

The historical README also mentions a `--seed` flag that its parser does not expose;
the benchmark calls `sample_targets(..., seed=0)`. Retain the versioned robot geometry,
tool frames and obstacle data when investigating the old protocol.

Outstanding provenance: no complete locked historical Python/CUDA/competitor
environment or per-table collection manifest was established by this local audit.
The generator override needs confirmation against the actual paper experiment setup.
The historical build and full paper campaign were **not rerun** here, so this is an
auditable starting procedure, not a verified end-to-end reproduction claim. Hardware
experiments require the physical setup; timing also depends on hardware and software.

## Current-code reruns of paper workloads

Use an isolated checkout/virtual environment: this workflow regenerates robot headers and rebuilds
the extension for several frames. Set `OUT_DIR` to a **new, nonexistent directory** for each run;
the harness refuses to overwrite earlier evidence. The commands below rerun the documented workloads,
not a guarantee of the published numerical results on different hardware or code revisions.
The completed [October 5 same-machine A/B](https://github.com/A2R-Lab/HJCD-IK/tree/main/docs/development/evidence/audit_timing_2026-10-05/README.md)
and the [October 6 post-pin confirmation](https://github.com/A2R-Lab/HJCD-IK/tree/main/docs/development/evidence/postpin_timing_2026-10-06/README.md)
(performance-neutral) are separate from the historical paper/competitor tables. The
[October 7 full campaign](https://github.com/A2R-Lab/HJCD-IK/tree/main/docs/development/evidence/paper_rerun_2026-10-07/README.md) re-measured every table
with all baselines on that recorded revision, not every later revision.

Current hard/both collision filtering, self-collision checks, fallback behavior, the
object API and compiled geometry differ from the historical implementation. In
particular the frozen paper post-hoc model uses +/-65 mm finger origins, while the
compiled Panda uses +/-40 mm. `--collision-validation-model paper` selects only the
post-hoc judge; it does not restore the old solver. Current optional cuRobo is v2,
not necessarily the paper's competitor version. Keep these differences in reports.
The closeout audit checked CLI/parser paths and script syntax, not timing or a full rerun.

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
