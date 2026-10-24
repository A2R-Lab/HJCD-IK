# Benchmarks and paper reproduction

How to run the HJCD-IK benchmark harness, install the optional competitor baselines, and reproduce the
paper protocol. Measured numbers live on [the results page](results.rst). Latency is only meaningful on an
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

## Reproducing the Paper Benchmarks

Use an isolated checkout/virtual environment: this workflow regenerates robot headers and rebuilds
the extension for several frames. Set `OUT_DIR` to a **new, nonexistent directory** for each run;
the harness refuses to overwrite earlier evidence. The commands below reproduce the protocol,
not a guarantee of the published numerical results on different hardware or code revisions.
The completed [October 5 same-machine A/B](https://github.com/A2R-Lab/HJCD-IK/tree/main/docs/development/evidence/audit_timing_2026-10-05/README.md)
and the [October 6 post-pin confirmation](https://github.com/A2R-Lab/HJCD-IK/tree/main/docs/development/evidence/postpin_timing_2026-10-06/README.md)
(performance-neutral) are separate from the historical paper/competitor tables. The
[October 7 full campaign](https://github.com/A2R-Lab/HJCD-IK/tree/main/docs/development/evidence/paper_rerun_2026-10-07/README.md) re-measured every table
with all baselines on the shipped code.

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

