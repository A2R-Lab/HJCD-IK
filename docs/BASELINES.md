# Competitor baselines & reproducing the paper experiments

HJCD-IK is benchmarked against other GPU-parallel IK solvers (paper Tables I–IV). The baselines are
**optional and heavy** — HJCD-IK itself needs none of them. This doc covers installing them, running
the experiments, and what is/isn't wired up yet.

## Solvers

| Solver | Where | Status in this repo |
|---|---|---|
| **HJCD-IK** | `benchmark/hjcd_ik_bench.py` | core (built extension) |
| **PyRoki** | `benchmark/baseline_bench.py --mode pyroki` | optional extra (JAX) |
| **cuRobo** | `benchmark/baseline_bench.py --mode curobo` | optional extra (torch, source build) |
| **IKFlow** | `benchmark/baseline_ikflow.py` | optional extra (torch + ikflow); **confirm model name/API** |
| **TRAC-IK** (MMD ground truth) | `benchmark/gen_groundtruth_tracik.py` | optional extra (`tracikpy`) |
| MMD / MMD² metric | `benchmark/{mmd,run_mmd}.py` | core (numpy); reads `--mmd-dump` config dumps |

> The baseline harness imports both the PyRoki and cuRobo stacks at module load, and uses cuRobo's
> `RobotWorld` for collision-checking even in `--mode pyroki`. So running *either* mode currently needs
> *both* stacks installed. (Making the imports mode-lazy so PyRoki can run without cuRobo is a tracked
> follow-up.)

## Install

Base install is unchanged and baseline-free:
```bash
python -m pip install -e .
```

Add the baselines with the helper (each stage is skippable):
```bash
./scripts/install_baselines.sh                 # PyRoki + cuRobo
SKIP_CUROBO=1 ./scripts/install_baselines.sh   # PyRoki only (no torch/cuRobo build)
SKIP_PYROKI=1 ./scripts/install_baselines.sh   # cuRobo only
```
It installs the `baselines`+`plots` package extras then the solver stacks pip can't resolve directly,
each independently skippable (`SKIP_PYROKI` / `SKIP_CUROBO` / `SKIP_IKFLOW` / `SKIP_TRACIK`):
- **PyRoki**: `jax[cuda12]` + `jaxls` (git, brentyi) + `pyroki` (git, chungmin99).
- **cuRobo**: `torch` + `NVlabs/curobo@v0.7.6` from source (`--no-build-isolation`), cloned to a persistent
  `$CUROBO_SRC` (`~/.cache/curobo_src`).
- **IKFlow**: `torch` + `ikflow` (downloads pretrained weights on first use).
- **TRAC-IK**: `mjd3/tracikpy` built **ROS-free** — apt `swig liborocos-kdl-dev libnlopt-dev liburdfdom-dev`,
  then the script vendors two shims (`benchmark/vendor/tracik/`) and patches `setup.py` (see issue below).
  Cloned to a persistent `$TRACIK_SRC` (`~/.cache/tracikpy_src`).

Git refs / CUDA extra are overridable (`PYROKI_REF`, `JAXLS_REF`, `CUROBO_REF`, `CUROBO_SRC`, `JAX_CUDA`).
Pins are best-effort; for an exact paper-matching environment reconcile against the co-author's `pip freeze`.

### Known install issues (verified 2026-06-22 on an RTX 5090 / CUDA 13.2 box)
- **PyRoki, IKFlow, jax(GPU), torch(GPU) install cleanly.** PyRoki passes the EE-frame check at 0.000 mm.
- **cuRobo API split:** `v0.8.0`+ and `main` are the **rewrite** (no `curobo.wrap.reacher.ik_solver`); the
  harness uses the **classic API = v0.7.x and earlier**, hence the `v0.7.6` pin. (`v0.8.0` builds fine but
  imports the wrong API.) **Blackwell caveat:** classic v0.7.x predates RTX 50xx (sm_120)/CUDA 13 and may
  not build/run there; the paper used an RTX 4060 / CUDA 12.5. If v0.7.6 fails on this box, match the
  co-author's cuRobo+torch+CUDA, or collect the cuRobo column on a CUDA-12 GPU. Changing `CUROBO_REF`
  re-clones `$CUROBO_SRC` automatically.
- **`baseline_bench.py` runs without cuRobo.** cuRobo imports are lazy (`_HAS_CUROBO`), so `--mode pyroki`
  open-world (Table I) and the DoF variants (Table III) work with PyRoki alone — no torch/cuRobo needed.
  Two cuRobo-only couplings remain by design: (1) `--mode curobo` exits early with a clear message if cuRobo
  is absent; (2) the **collision-free check for Table II uses cuRobo's `RobotWorld`** — with cuRobo absent,
  PyRoki collision-free still reports IK timing/accuracy but the `collision_free` column is left blank (you'll
  see a one-line WARNING). HJCD-IK uses its own collision filter, so the HJCD Table II column is unaffected.
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
- **IKFlow model names** are the `model_descriptions.yaml` keys: `panda_full_tpm` (default), `panda_lite_tpm`,
  and Fetch IS supported (`fetch_full_temp_tpm`). Weights download from a GCS bucket on first use; a
  network-restricted box may get HTTP 403 (have the co-author share the `.pkl` weights if so).

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
>   `python scripts/generate_grid.py include/test_urdf/panda.urdf -t panda_hand_joint && bash scripts/rebuild.sh`
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
./scripts/run_paper_experiments.sh                            # Tables I+II, all solvers
RUN_DOF=1 RUN_MMD=1 ./scripts/run_paper_experiments.sh        # + Table III (DoF) + Table IV (MMD)
HJCD_REGEN=1 RUN_DOF=1 RUN_MMD=1 ./scripts/run_paper_experiments.sh  # also rebuild HJCD per frame/DoF (GPU)
SKIP_CUROBO=1 SKIP_PYROKI=1 ./scripts/run_paper_experiments.sh  # HJCD-IK only
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
# Table IV
python benchmark/run_mmd.py --groundtruth dumps/groundtruth.json \
    --solver-dump dumps/hjcdik.json dumps/pyroki.json dumps/curobo.json --out results/table4_mmd.md
```
`mmd.py` uses a Gaussian kernel with a shared median-heuristic bandwidth (the *ranking* is robust; confirm
the exact kernel/bandwidth with the authors only if you need to match the absolute MMD values). The config
dump is a small JSON schema (see `run_mmd.py`), so **IKFlow drops in by writing the same dump** — the
remaining missing piece is the IKFlow solver itself (a `--mode ikflow` adapter + its pretrained weights).

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
| Table I — open-world, **Fetch** | ✓ HJCD + PyRoki/cuRobo via `--robot-urdf` + IKFlow (`fetch_full_temp_tpm`); `RUN_FETCH=1` |
| Table II — collision-free, Panda | HJCD ✓; PyRoki/cuRobo wired |
| Table III — **DoF 7/12/18/24** | ✓ all solvers — HJCD per-robot codegen + baselines via `--robot-urdf`; run with `RUN_DOF=1` |
| Table IV — **MMD / MMD²** | compute ✓; TRAC-IK ground truth ✓ (`gen_groundtruth_tracik.py`); HJCD/PyRoki/cuRobo/IKFlow dumps ✓ — all wired, unrun |
| EE-frame equivalence | ✓ `benchmark/check_ee_frames.py` (gated smoke test) |
| Figs 4/5 — Pareto plots | ✓ `benchmark/plot_pareto.py` + `make_tables.py` (consume the run CSVs) |
| Fig 6 — hardware | out of scope (physical Franka, `realworld` branch) |

Bigger picture / running notes: `docs/open-tasks/baseline_repro_plan_2026-06-21.md` (local).
