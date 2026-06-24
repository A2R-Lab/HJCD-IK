# Competitor baselines & reproducing the paper experiments

HJCD-IK is benchmarked against other GPU-parallel IK solvers (paper Tables I–IV). The baselines are
**optional and heavy** — HJCD-IK itself needs none of them. This page covers installing them, running
the experiments, and the comparison methodology.

:::{note}
This is a **methodology / how-to** page. The published numbers and figures live on the {doc}`results`
page and come **verbatim from the camera-ready paper** — the single source of truth. Any timings
mentioned below are setup/sanity checks, not benchmark results, and a run on your hardware will differ.
:::

## Solvers

| Solver | Where | Status in this repo |
|---|---|---|
| **HJCD-IK** | `benchmark/hjcd_ik_bench.py` | core (built extension) |
| **PyRoki** | `benchmark/baseline_bench.py --mode pyroki` | optional extra (JAX) |
| **cuRobo** (v2 API) | `benchmark/baseline_bench.py --mode curobo` | optional extra (torch + curobo@main + cuda-core) |
| **IKFlow** | `benchmark/baseline_ikflow.py` | optional extra (torch + ikflow); weights load **offline** from `benchmark/assets/ikflow/` |
| **TRAC-IK** (MMD ground truth) | `benchmark/gen_groundtruth_tracik.py` | optional extra (`tracikpy`) |
| MMD / MMD² metric | `benchmark/{compute_mmd,run_mmd}.py` | core (numpy/pandas); canonical IMQ MMD, reads config dumps |

> cuRobo + PyRoki imports are **mode-lazy** (`_HAS_CUROBO`): `--mode pyroki` runs with no cuRobo installed,
> and `--mode curobo` runs with no JAX/PyRoki concerns. The only remaining cuRobo coupling is the optional
> collision-validation of PyRoki's Table II solutions (see the collision note below).

## Install

Base install is unchanged and baseline-free:
```bash
python -m pip install -e .
```

Add the baselines with the helper (each stage is skippable):
```bash
./scripts/setup/install_baselines.sh                 # PyRoki + cuRobo
SKIP_CUROBO=1 ./scripts/setup/install_baselines.sh   # PyRoki only (no torch/cuRobo build)
SKIP_PYROKI=1 ./scripts/setup/install_baselines.sh   # cuRobo only
```
It installs the `baselines`+`plots` package extras then the solver stacks pip can't resolve directly,
each independently skippable (`SKIP_PYROKI` / `SKIP_CUROBO` / `SKIP_IKFLOW` / `SKIP_TRACIK`):
- **PyRoki**: `jax[cuda12]` + `jaxls` (git, brentyi) + `pyroki` (git, chungmin99).
- **cuRobo (v2)**: `torch` + `NVlabs/curobo@main` from source (`--no-build-isolation`, cloned to a persistent
  `$CUROBO_SRC` = `~/.cache/curobo_src`) + **`cuda-core[cu13]`** (the runtime kernel backend — v2 JIT-compiles
  CUDA kernels, no C++ build). On CUDA 12 set `CUDA_CORE_EXTRA=cu12`.
- **IKFlow**: `torch` + `ikflow`. Weights are loaded **offline** by `baseline_ikflow.py` from
  `benchmark/assets/ikflow/` (registry yaml committed; `.pkl` weights gitignored under `weights/`) — no
  download (the public GCS bucket 403s here).
- **TRAC-IK**: `mjd3/tracikpy` built **ROS-free** — apt `swig liborocos-kdl-dev libnlopt-dev liburdfdom-dev`,
  then the script vendors two shims (`benchmark/vendor/tracik/`) and patches `setup.py` (see issue below).
  Cloned to a persistent `$TRACIK_SRC` (`~/.cache/tracikpy_src`).

Git refs / CUDA extra are overridable (`PYROKI_REF`, `JAXLS_REF`, `CUROBO_REF`, `CUROBO_SRC`, `JAX_CUDA`).
Pins are best-effort; for an exact paper-matching environment reconcile against the co-author's `pip freeze`.

### Known install issues (verified 2026-06-23 on an RTX 5090 / CUDA 13.2 box)
- **PyRoki, IKFlow, jax(GPU), torch(GPU) install cleanly.** PyRoki passes the EE-frame check at 0.000 mm.
- **cuRobo = v2 API (`curobo@main`), builds + runs on Blackwell.** The harness was ported from the classic
  `curobo.wrap.reacher.ik_solver` API to v2 (`curobo.inverse_kinematics.InverseKinematics`,
  `curobo.scene.Scene`, `curobo.robot_builder.RobotBuilder`). The classic `v0.7.6` is abandoned here: it does
  **not** build on CUDA 13 / gcc 13 (`lerp` vs C++23 `std::lerp`) and predates sm_120. v2/main installs as a
  pure-Python package and JIT-compiles kernels at runtime via **cuda-core** — so `pip install 'cuda-core[cu13]'`
  is required (without it you get `ModuleNotFoundError: No module named 'cuda.core'` on first solve). Verified:
  open-world Panda (franka.yml) ~2 ms/solve sub-mm; URDF/DoF path (RobotBuilder) ~2 ms sub-mm; MMD dump and
  collision-free Table II both run.
- **cuRobo robots:** v2 bundles `franka.yml` (= Panda, tool `panda_hand`) — used directly for Panda Tables
  I/II/IV. Fetch and the DoF variants go through `--robot-urdf`: `RobotBuilder` builds a robot config from the
  URDF, rooted at `--base-link` (the harness trims the URDF to that subtree so the DoF count is fair, e.g.
  Fetch arm at `arm_mount_link`, not the mobile base).
- **`baseline_bench.py` runs without cuRobo.** Imports are lazy (`_HAS_CUROBO`), so `--mode pyroki` works with
  no cuRobo. `--mode curobo` exits early with a clear message if cuRobo is absent.
- **Table II PyRoki collision column — v2 follow-up.** Validating PyRoki's returned joints as collision-free
  used the classic `RobotWorld`; the v2 equivalent (export robot spheres at a config → `scene_collision_checker.
  get_sphere_distance`) is **not yet wired**, so that check returns "unknown" and the `collision_free` column is
  left blank for PyRoki (honest, not a crash). cuRobo's **own** collision-free IK (Table II `--mode curobo`) is
  fully supported via the in-optimizer collision cost. HJCD-IK uses its own collision filter (unaffected).
- **`tracikpy` is NOT on PyPI and assumes ROS.** Its C++ `#include`s `<kdl_parser/kdl_parser.hpp>` and
  `<urdf/model.h>` — both ROS packages, absent on a bare Linux box (you'll see
  `fatal error: kdl_parser/kdl_parser.hpp: No such file or directory`, then a missing `urdf::Model`). We build
  it **ROS-free**: `benchmark/vendor/tracik/` ships two tiny shims over urdfdom+KDL — a minimal `kdl_parser`
  (just `treeFromUrdfModel`, the only function tracikpy calls) and a `urdf::Model` wrapper whose
  `initString`/`initFile` delegate to `urdf::parseURDF`. `install_baselines.sh` drops them into the checkout
  and patches `setup.py` to link `urdfdom_model` (not the ROS `urdf`/`kdl_parser` libs) and add
  `-I/usr/include/urdfdom`. The one apt package most non-ROS boxes lack is **`liburdfdom-dev`** (provides
  `urdf_parser.h` + `liburdfdom_model.so`); `liborocos-kdl-dev`/`libnlopt-dev`/`swig` are the others.
  Verified 2026-06-22: builds + FK→IK round-trips to 0.0 mm on `panda.urdf` (base `panda_link0`, tip `panda_hand`).
- **IKFlow models load offline** from the co-author's registry `benchmark/assets/ikflow/model_descriptions.yaml`
  (merged into the installed ikflow package at runtime). Panda default = **`panda__full__lp191_5.25m`** (12
  nodes, latent dim 7 — the stock ikflow `panda_full_tpm` is a *different* architecture/weights, so do not use
  it for these `.pkl`s); Fetch = **`fetch_full_temp_nsc_tpm`**. Drop the `.pkl`s in
  `benchmark/assets/ikflow/weights/` (gitignored) and `baseline_ikflow.py` stages them into ikflow's cache by
  URL basename — no network. Override the search dir with `IKFLOW_WEIGHTS_DIR` or `--weights-dir`.

## Fair targets (shared across solvers)

Open-world numbers are only comparable if every solver solves the **same** poses. `benchmark/gen_targets.py`
samples a neutral Halton set in joint space and runs a pure-numpy URDF FK to the common EE frame —
default **`panda_hand`** (`--target panda_hand_joint`), matching the baseline scripts' open-world EE
(PyRoki `ik_beam_hand`). The numpy FK is HJCD-IK's own `tests/test_fk_equivalence.py` reference (<0.1 mm).
Core deps only — no GPU or baseline stack needed to regenerate:
```bash
python benchmark/gen_targets.py --num-targets 100 --out benchmark/targets/panda_open
#   -> panda_open.json  (HJCD-IK:  --filtered-targets)
#   -> panda_open.yml   (baselines: --goal_file)
```
Reusable for Fetch / 12·18·24-DoF via `--urdf <u> --target <ee_frame>` (see the EE map in
`docs/reproduce_paper_sweeps.md`).

> ⚠️ **EE-frame alignment — do this before trusting open-world numbers.** Decision: use the frames the
> baseline scripts actually used → open-world common frame = **`panda_hand`**. PyRoki open-world already
> uses it (`ik_beam_hand`). The two things to align:
> - **HJCD-IK** ships targeting `panda_grasptarget_hand` (TCP, ~10 cm out). For the open-world comparison,
>   rebuild it to `panda_hand` so it matches the shared targets:
>   `python scripts/codegen/generate_grid.py include/test_urdf/panda.urdf -t panda_hand_joint && bash scripts/setup/rebuild.sh`
>   (restore the default afterward, or keep a separate build). `run_paper_experiments.sh` does this when
>   `HJCD_REGEN=1`.
> - **cuRobo**: confirm `franka.yml`'s ee_link resolves to `panda_hand` (its FK-sampled goals already do).
>
> Validate before trusting numbers with the smoke test — it FKs the same configs through each installed
> solver's own model and compares to the reference (a ~0.1 m offset = wrong tool frame):
> ```bash
> python benchmark/check_ee_frames.py --num 8     # PASS/FAIL/SKIP per backend; run_paper_experiments.sh runs it
> ```
> Collision-free is separate: targets come from the shared problem set; PyRoki uses `panda_hand_tcp`
> (`ik_beam`) there.

## Run

One shot (collects per-solver CSVs to `benchmark/results/`):
```bash
./scripts/bench/run_paper_experiments.sh                            # Tables I+II, all solvers
RUN_DOF=1 RUN_MMD=1 ./scripts/bench/run_paper_experiments.sh        # + Table III (DoF) + Table IV (MMD)
HJCD_REGEN=1 RUN_DOF=1 RUN_MMD=1 ./scripts/bench/run_paper_experiments.sh  # also rebuild HJCD per frame/DoF (GPU)
SKIP_CUROBO=1 SKIP_PYROKI=1 ./scripts/bench/run_paper_experiments.sh  # HJCD-IK only
```
`RUN_DOF=1` sweeps 7/12/18/24-DoF (chained Panda variants) at B=1000: HJCD-IK rebuilt per DoF (when
`HJCD_REGEN=1`) and PyRoki/cuRobo via `--robot-urdf`. `RUN_FETCH=1` adds the Fetch open-world batch sweep
(EE = `ee_link`). `HJCD_REGEN=1` restores the default panda build on exit.
Or individually:
```bash
# open-world (Table I)
python benchmark/hjcd_ik_bench.py --skip-grid-codegen --filtered-targets benchmark/targets/panda_open.json \
    --batches 1,10,100,1000,2000 --solver hjcdik --csv-out benchmark/results/open_hjcdik.csv
python benchmark/baseline_bench.py --mode pyroki --goal_file benchmark/targets/panda_open.yml --seed_list 1,10,100,1000,2000
python benchmark/baseline_ikflow.py --goal_file benchmark/targets/panda_open.yml --seed_list 1,10,100,1000,2000 --csv-out benchmark/results/open_ikflow.csv

# collision-free (Table II), MotionBenchMaker bookshelf_thin_panda
python benchmark/hjcd_ik_bench.py --skip-grid-codegen --collision-free \
    --problems-json tests/mb_problems.json --problem-set bookshelf_thin_panda
MB_JSON_PATH=$PWD/tests/mb_problems.json python benchmark/baseline_bench.py --mode curobo \
    --collision_free --problem_set bookshelf_thin_panda
```

The paper's **"Batch"** axis maps to: HJCD-IK `--batches` (batch_size) == cuRobo/PyRoki `--seed_list`
(cuRobo `num_seeds` / PyRoki `num_seeds_init`).

## Tables & plots

`run_paper_experiments.sh` calls these at the end; run them standalone over any mix of per-solver CSVs
(both schemas auto-detected, baseline per-problem rows averaged):
```bash
# paper-style markdown table (Tables I–III)
python benchmark/make_tables.py benchmark/results/open_*.csv --title "Panda open-world (Table I)"

# accuracy-latency Pareto (Figs 4/5) — needs matplotlib (pip install -e ".[plots]")
python benchmark/plot_pareto.py benchmark/results/open_*.csv --out benchmark/results/pareto_open.png --annotate-batch
```
`make_tables.py` is stdlib-only; `plot_pareto.py` needs `matplotlib`. Left panel = position vs orientation
error; right = combined error vs solve time (HJCD-IK should sit lower-left / further left).

## Table IV — solution diversity (MMD)

MMD/MMD² measure how well a solver's returned batch matches a ground-truth distribution of IK solutions
(paper: TRAC-IK seeded samples); lower = better manifold coverage. Each solver dumps its K=50 best configs
per target (of a batch/seed count of 2000), and TRAC-IK provides the ground truth. Enable in the
orchestrator with `RUN_MMD=1`, or run by hand:
```bash
# per-solver config dumps over the shared targets (joint-space; open-world, panda_hand frame)
python benchmark/hjcd_ik_bench.py --skip-grid-codegen --filtered-targets benchmark/targets/panda_open.json \
    --solver hjcdik --mmd-dump dumps/hjcdik.json --mmd-batch 2000 --solutions-count 50
python benchmark/baseline_bench.py --mode pyroki --goal_file benchmark/targets/panda_open.yml \
    --mmd_dump dumps/pyroki.json --solutions_seed 2000 --solutions_k 50
# ground truth (needs tracikpy)
python benchmark/gen_groundtruth_tracik.py --targets benchmark/targets/panda_open.json --tip panda_hand \
    --num-samples 50 --out dumps/groundtruth.json
# IKFlow dump (offline weights)
python benchmark/baseline_ikflow.py --goal_file benchmark/targets/panda_open.yml \
    --mmd-dump dumps/ikflow.json --mmd-batch 2000 --solutions-count 50
# Table IV
python benchmark/run_mmd.py --groundtruth dumps/groundtruth.json \
    --solver-dump dumps/hjcdik.json dumps/pyroki.json dumps/curobo.json dumps/ikflow.json --out results/table4_mmd.md
```
The **canonical metric is `benchmark/compute_mmd.py`** (co-author's): joint-space MMD with an inverse
multi-quadric (IMQ) kernel, multi-bandwidth (median × {0.2,0.5,1,2,5}), **unbiased** MMD², averaged per pose.
`run_mmd.py` drives it from the per-solver config dumps and also writes a flat `<dump>.csv`
(`solver,pose_id,q1..qN`) next to each dump, so you can reproduce one column directly:
`python benchmark/compute_mmd.py --ref dumps/groundtruth.csv --cmp dumps/hjcdik.csv --group_col pose_id`.
(`mmd.py` is now just the dump-I/O + CSV exporter; the old Gaussian estimator there was non-canonical and
was removed.) Every solver — HJCD, PyRoki, cuRobo, IKFlow — writes the same dump schema.

## Metrics / units

- Position error in **mm**, orientation error in **rad**, time in **ms** — unified across harnesses
  (`hjcd_ik_bench.py` CSV cols `solver,Batch-Size,time_ms,pos_err_mm,ori_err_rad`; `baseline_bench.py`
  emits `Pos-Error(mm)` / `Ori-Error` / `IK-time(ms)`).
- Baseline success thresholds: position < 5 mm, orientation < 0.05 rad (cuRobo convention) — internal
  only; reported errors are the raw values.

## Coverage / gaps (TODO)

| Paper item | State |
|---|---|
| Table I — open-world, **Panda** | HJCD ✓; PyRoki/cuRobo/IKFlow wired (shared targets) |
| Table I — open-world, **Fetch** | ✓ HJCD + PyRoki/cuRobo via `--robot-urdf` + IKFlow (`fetch_full_temp_nsc_tpm`, offline); `RUN_FETCH=1` |
| Table II — collision-free, Panda | HJCD ✓; PyRoki/cuRobo wired |
| Table III — **DoF 7/12/18/24** | ✓ all solvers — HJCD per-robot codegen + baselines via `--robot-urdf`; run with `RUN_DOF=1` |
| Table IV — **MMD / MMD²** | compute ✓; TRAC-IK ground truth ✓ (`gen_groundtruth_tracik.py`); HJCD/PyRoki/cuRobo/IKFlow dumps ✓ — all wired, unrun |
| EE-frame equivalence | ✓ `benchmark/check_ee_frames.py` (gated smoke test) |
| Figs 4/5 — Pareto plots | ✓ `benchmark/plot_pareto.py` + `make_tables.py` (consume the run CSVs) |
| Fig 6 — hardware | out of scope (physical Franka, `realworld` branch) |

Bigger picture / running notes: `docs/open-tasks/baseline_repro_plan_2026-06-21.md` (local).
