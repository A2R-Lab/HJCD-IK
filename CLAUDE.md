# CLAUDE.md — orientation for agents (and humans) on HJCD-IK

**HJCD-IK** = **Hybrid Jacobian Coordinate Descent Inverse Kinematics**: a GPU-accelerated, *batched*
IK solver (paper: [arXiv:2510.07514](https://arxiv.org/abs/2510.07514)). It generates many candidate IK
solutions in parallel for a 6-DOF end-effector target, with optional collision avoidance, built on top of
[GRiD](https://github.com/A2R-Lab/GRiD) (robot kinematics codegen) and
[GLASS](https://github.com/A2R-Lab/GLASS) (single-block CUDA linear algebra).

> **Before changing the kernel or codegen, read [`docs/development/agent_debugging_guide.md`](docs/development/agent_debugging_guide.md).**
> It is the runbook for HJCD-IK's recurring traps: stale `grid.cuh`, `FLANGE_IDX`/target mismatch,
> warp-vs-block sync in the solver loop, and submodule init.
> **The user-facing API contract is [`docs/source/user_guide/upgrading.md`](docs/source/user_guide/upgrading.md)**
> (hard collision filtering by default, `count` may be zero, errors raise, native `Result<T>` is move-only).

## Mental model

Each solve handles one target and a batch of candidate configurations. **Coarse search assigns one
candidate per block**, with warps evaluating joint-pair perturbations. **LM refinement assigns one
candidate per warp**, optionally packing several independent candidates into a block.
LM math and per-warp coarse scratch stay warp-scoped; coarse state shared across warps needs block
barriers. Two phases (`csrc/kernel/hjcd_kernel.cu`):
1. **Coarse search** (`coarse_search`): random restarts + greedy pairwise coordinate descent. The candidate
   sweep over the second joint runs **lane-parallel across the warp** (`for j = lane; j < N; j += WARP_SIZE`
   + warp min-reduce); each candidate recomputes only the **FK suffix** from its perturbed joint
   (`ee_fk_suffix_thread`, built on `grid::update_XmatHom_joint`) rather than a full chain — this is what
   makes high-DoF scale (was O(N³) serial-on-lane-0; see `docs/development/agent_debugging_guide.md` §5).
2. **LM refine** (`solve_lm_batched` / `lm_tuner`): single-warp Levenberg–Marquardt — build the 6×N geometric
   Jacobian (cross-products), form & solve the normal equations `(JᵀJ + λ·diag)Δq = Jᵀr` via a hand-rolled
   `__shfl` warp-Cholesky, with dogleg/line-search backtracking.

Forward kinematics produces the **world-frame joint transforms** `s_jointXforms[16·jid]` (4×4 each); the EE
pose error is computed as a **quaternion** error (`mat_to_quat` / `quat_err_rotvec`). For Panda: `N = 7`
joints. Target indices and transform counts are generated constants; the default
`panda_grasptarget_hand` build currently has `grid::EE_FIXED_FRAME_IDX = 10`.

## Key files

| Path | What it is |
|---|---|
| `csrc/kernel/hjcd_kernel.cu` | The solver: block-cooperative coarse search + warp-scoped LM refine + host orchestration (`generate_ik_solutions`). **The file you'll edit most.** |
| `csrc/kernel/hjcd_settings.h` | `HJCDSettings<T>` (coarse/LM tolerances, `lambda_init`), `mat4_mul`, FK helpers (`ee_fk_warp`/`ee_fk_thread`/`ee_fk_suffix_thread`), `#include "grid.cuh"`, `N`/`FLANGE_JID`/`GRASP_FIXED_IDX`. |
| `csrc/kernel/hjcd_kernel.h` | Native host API: move-only `Result<T>`, `generate_ik_solutions<T,RT>(target, batch, ...)`, `sample_random_target_poses`. |
| `csrc/kernel/main.cpp` | Native CLI (`single`/`sweep`/`from_csv`), open-world only; CTest-covered by `tests/native/`. |
| `csrc/generated/grid.cuh` | **Generated** GRiD kinematics + collision header. Do **not** hand-edit. Generated with `vendor_glass=False`, so it `#include "glass.cuh"`s the top-level GLASS instead of vendoring a copy (~6k lines, was ~15k). |
| `external/GRiD/` | Submodule: GRiD codegen (emits `grid.cuh` from a URDF). Its nested GLASS pin must equal `external/GLASS`. |
| `external/GLASS/` | Submodule: GLASS single-block / warp / thread linear algebra (`glass::warp::`, `glass::thread::`, `glass::block::`). |
| `external/foam/` | Submodule: pre-spherized Panda collision URDF (`assets/panda/smaller_panda_spherized.urdf`). |
| `csrc/kernel/grid_env.cuh` | Parses a MotionBenchMaker problem JSON → `grid_collision::Environment` (device obstacle set) for the collision kernels; `problem_document.h` caches the parsed document. |
| `csrc/bindings/pybind_module.cpp` | Python bindings → `generate_solutions`, `sample_targets`, `num_joints`, `collision_enabled`, `build_info`. `hjcdik/__init__.pyi` is the typed stub. |
| `tests/` | pytest suite (API contracts, FK equivalence, collision policy, codegen, stats, regression vs `baseline_metrics.json`); `tests/native/` = CTest native API + CLI contracts. `gpu-proof-tests.txt` is the signed-receipt manifest. |
| `gpu-proof.json` | Signed pytest-gpu-proof receipt of the last full GPU run; verified by CPU-only CI. |
| `benchmark/hjcd_ik_bench.py` | HJCD-IK benchmark harness: solved-rate, position/orientation error, timing. |
| `scripts/perf/timing_driver.py` | Neutral A/B latency driver: run once per endpoint venv, alternating rounds; see `docs/development/evidence/timing_gate_2026-10-02/`. |
| `benchmark/baseline_bench.py` | Competitor baselines (PyRoki/cuRobo, `--mode`); optional, see `docs/source/user_guide/benchmarks/results.rst`. |
| `benchmark/baseline_ikflow.py` | IKFlow baseline (standalone, torch); same CSV/MMD-dump schema. |
| `benchmark/check_ee_frames.py` | Gated smoke test: do all solvers agree on the EE (panda_hand) frame? |
| `benchmark/collision_oracles.py`, `score_collision_oracles.py` | Independent collision judges (sphere models incl. cuRobo's; `hull` and `visual` FCL meshes) and the scorer for stored configurations. |
| `benchmark/mbm_export.py`, `make_clearance_ladder.py`, `make_curobo_sphere_urdf.py` | MotionBenchMaker export (mesh scenes → primitives), clearance-ladder sets, HJCD on cuRobo's sphere model. |
| `benchmark/gen_targets.py` | Neutral Halton + numpy-FK shared open-world targets (fair cross-solver compare). |
| `benchmark/{make_tables,plot_pareto}.py` | Merge per-solver CSVs → paper tables / accuracy-latency Pareto (Figs 4/5). |
| `benchmark/{mmd,run_mmd,gen_groundtruth_tracik}.py` | MMD/MMD² (Table IV): config dumps + TRAC-IK ground truth. |
| `tests/{mb,wall}_problems.json` | MotionBenchMaker / wall collision problem sets. |

## Build & test

CMake 3.24+ / CUDA 12.x or 13.x / pybind11 (scikit-build-core). GRiD codegen runs at configure time when enabled.

```bash
sudo apt install -y nlohmann-json3-dev   # collision environment JSON header dependency
git submodule update --init --recursive          # GRiD + GLASS
python -m pip install -e .                        # builds the _hjcdik extension (CUDA arch auto-detected)
python benchmark/hjcd_ik_bench.py --skip-grid-codegen   # run the solver
```

`./scripts/setup/setup_dev.sh` does all of the above (system deps + pinned submodules + venv + codegen + build).
`scripts/setup/bootstrap.sh` alone pins the three submodules (GRiD, GLASS, foam) at the committed revisions.

**Testing and the GPU-proof gate.** `python -m pytest tests` needs a CUDA GPU and the collision-enabled Panda
build. There is no GPU CI runner: `.github/workflows/verify-gpu-proof.yml` only verifies the committed, signed
`gpu-proof.json` receipt, whose fingerprint covers `csrc/`, `hjcdik/`, `tests/`, `benchmark/`, `scripts/`,
`examples/`, `docs/source/`, the build files and the submodule gitlinks. **Any change to those paths makes the
receipt stale**: run the full suite and `scripts/setup/run_gpu_proof.sh` on a GPU (Python ≥ 3.11) and commit the
new receipt, after `scripts/setup/update_gpu_proof_manifest.py` if test IDs changed. A compile-only check without a
GPU is possible with the `nvidia-cuda-nvcc` pip wheel (`nvcc -c csrc/kernel/hjcd_kernel.cu -arch=sm_80 ...`).
Native checks: `cmake -S . -B build-native -DBUILD_PYTHON=OFF -DHJCDIK_BUILD_NATIVE_TESTS=ON && cmake --build
build-native && ctest --test-dir build-native`. `hjcdik.build_info()` reports the compiled header's SHA-256 so a
stale install or wrong-robot wheel can be detected without initializing CUDA.

Python API:
```python
from hjcdik import generate_solutions, sample_targets, num_joints
targets = sample_targets(num_targets=10, seed=0)         # list of [x,y,z, qw,qx,qy,qz]
out = generate_solutions(targets[0], batch_size=2000, num_solutions=4)
# out = {joint_config, pose, pos_errors, ori_errors, count}; count may be < num_solutions (even 0)
# pos_errors are mm, ori_errors are rad; refine_fp64=-1 (default) = fp64 for one solution, fp32 for several
```

## Conventions / discipline

- **Never hand-edit `grid.cuh`.** It is GRiD codegen output. To change the robot or EE target, run
  `python scripts/codegen/generate_grid.py <urdf> -t <target_frame>` and rebuild. Robot constants
  (`NUM_JOINTS`, topology counts) are baked per-URDF — read them from the generated symbols, never hardcode.
- **Collision is URDF-driven (grid_collision), not hand-coded.** Passing `--collision` to
  `generate_grid.py` bakes GRiD's `grid_collision` namespace (per-robot spheres + self-collision ranges)
  into `grid.cuh`; the kernel scores it via `grid_collision::collision_distance` (see
  `csrc/kernel/hjcd_kernel.cu` `score_environment_costs`). Sphere source: `--collision-res R` spherizes
  the URDF's own collision geometry, OR `--spherized-urdf <foam.urdf>` reads a pre-spherized (foam-format)
  URDF directly — use the latter when the URDF's collision meshes don't resolve on disk. **Panda uses the
  checked-in foam sphere geometry** (`external/foam/assets/panda/smaller_panda_spherized.urdf`,
  58 non-base spheres), bound to the kinematic URDF's fixed frames. Its +/-40 mm finger origins
  differ from the legacy paper reference's +/-65 mm origins. The build/codegen wires this automatically
  (see `CMakeLists.txt`). This is
  the **bring-your-own-URDF** path: `generate_grid.py <robot.urdf> --collision [...]` provides FK and
  collision for supported fixed-base serial arms with no hand-written per-robot header.
- **Collision-aware refinement (hard/both modes, 2026-10).** The cross-block early stop is raised only by a
  collision-free accurate candidate (`grid_collision::warp::config_free`: warp-scoped, generated sphere
  tables and per-warp joint transforms — no block barrier,
  no extra FK); LM seeds are ranked with a penalty on colliding coarse candidates; an accurate-but-colliding
  LM candidate gets `HJCD_REPAIR_ATTEMPTS` (4) kicked re-projections; the best collision-free configuration
  inside `HJCDSettings::cc_fallback_*` (5 mm / 0.05 rad) can be returned when this run finds no exactly
  converged free candidate. This does not establish infeasibility. `HJCD_CC_STOP` (bit 0 coarse, bit 1 LM)
  is the A/B knob. GRiD owns the tables and warp checker; HJCD has no duplicate collision sidecar.
- **Collision policy.** Python exposes `collision_mode="hard"|"soft"|"both"`; `hard` is the
  default and strictly excludes colliding candidates (self **+** environment). `soft` is a penetration-cost
  ranking mode and does not guarantee collision freedom; `both` ranks and filters. Neither the API nor the
  native solver reads `HJCD_CC_MODE`; only the benchmark CLI keeps it as the default of `--collision-mode`.
  Final filtering/ranking is post-solve; hard/both additionally check collision during refinement.
  Obstacle JSON uses `pose` (`[x,y,z,qw,qx,qy,qz]`); the legacy
  Euler `box` schema is gone.
- **`FLANGE_IDX` discipline.** The fixed EE target (`panda_grasptarget_hand`) and its index must agree across
  codegen, the kernel, and any benchmark problem. A mismatch silently solves to the wrong frame.
- **Warp-locality is the performance contract.** New math must stay warp-scoped (`__shfl_*_sync`, `__syncwarp`).
  Do not drop the solver onto block-scoped primitives.
- **Prefer GLASS/GRiD primitives over hand-rolled math.** Quaternion ops, norms, 4x4 products, reductions and the
  normal-equation solve all come from `glass::thread::` / `glass::warp::` / `glass::block::`; the hand-rolled
  helpers that remain (`solve_pos`/`solve_ori`, `ee_residual6`, `robust_row_weights`, `rank_score`) are HJCD-specific.
  Generic kernels that outgrow HJCD belong upstream (kinematics → GRiD, linear algebra → GLASS).
- **Tidy-ups must be numerics-preserving unless validated on a GPU.** The LM cost path has deliberate asymmetries
  (see the debugging guide §6); refactor by naming shared pieces, not by "fixing" them.
- **Non-determinism is accepted.** Candidate identity is not run-to-run stable (cross-block early-stop race);
  compare builds and write regression tests statistically (`scripts/bench/capture_baseline.py` suites, slack
  bands), never by pinning a returned configuration. Ruled 2026-10-02; no deterministic mode is planned.
- **Short, single-line commit messages; no `Co-Authored-By` footer.**

## History — where the code came from

**2026-07 — re-based on GRiD/GLASS (PR #2).** The bespoke Panda-only FK (`X_warp` / `X_single_thread`) was
replaced by GRiD's stock warp FK (`grid::ee_pose_inner_warp`), and the hand-rolled math (4x4 products, warp
reduce, warp Cholesky, quaternion error) moved onto GLASS's `glass::warp::` / `glass::thread::` tiers. The
end-effector frame is **per-robot** (codegen resolves `grid::EE_FIXED_FRAME_IDX` from the named target and
injects it; `hjcd_settings.h` consumes it) — see
[`docs/source/user_guide/benchmarks/results.rst`](docs/source/user_guide/benchmarks/results.rst)
for the per-robot EE map + how to regenerate the paper sweeps.

**2026-09/10 — audit hardening (PR #3).** Strict `hard` collision filtering became the default; the native API
got a move-only `Result<T>`, checked CUDA/GRiD initialization that raises instead of aborting, a solver lock, and
a per-device parsed-document + environment cache (scene switches ~14x faster on the host, kernel unchanged).
The build compiles the CUDA core once (`hjcdik_core`) for both front ends; native CTest + CLI contract tests,
`build_info()`, the typed stub and the signed GPU-proof manifest were added. Full record:
[`docs/development/audit_hardening_validation.md`](docs/development/audit_hardening_validation.md).

**2026-10 — de-vendored GLASS + tidy-up.** `grid.cuh` is generated with `vendor_glass=False` (one GLASS per
translation unit, header ~6k lines instead of ~15k). Dead helpers, duplicate host logic, the ignored native
`d_robotModel` argument, the `HJCD_CC_MODE` / `"auto"` shim and the legacy Euler obstacle schema were removed;
the device and host candidate ranking now share one `rank_score` (the host previously used a different
orientation weight).

**Collision migrated to `grid_collision`.** The former bespoke pRRTC stack (`csrc/collision/` +
`csrc/robots/{panda,fetch}.cuh`) is gone; collision is now GRiD's URDF-driven `grid_collision` baked into
`grid.cuh` (`--collision`), scored post-solve by `mark_collisions` for strict filtering and optionally by `score_environment_costs`
for soft ranking (hard/both now also check within the warp solver). Strict filtering can return fewer
solutions than the historical soft-ranking path. The paper reference model lives frozen under
`benchmark/reference/panda_collision_model.cuh`; it is NOT identical to the compiled geometry
because the fixed finger openings differ. `benchmark/panda_model.py` provides explicit `paper`
and URDF-derived `hjcd` models. Paper comparisons retain `paper`; implementation tests use `hjcd`.
Both Python oracles check environment collisions only, not the kernel's self-collision policy.

The collision code path is compiled in only when `grid.cuh` was generated with `--collision` — codegen emits
a `#define HJCD_HAS_COLLISION 1` sentinel and the kernel + `grid_env.cuh` guard all `grid_collision::` use on
it. A no-collision header (e.g. the DoF-scaling regens, or any BYO-URDF built without `--collision`) still
compiles and runs open-world; the Python API rejects a collision-free request in that build.

**Performance status (updated October 5).** The completed pre-upstream A/B gate has identical matched
quality counts, 19/20 cells consistently faster and one approximately unchanged; see
[`audit_timing_2026-10-05`](docs/development/evidence/audit_timing_2026-10-05/README.md).
The subsequent GRiD/GLASS pin update is validated separately, not silently included in those measurements.
Protocol details:
[`docs/development/timing_gate.md`](docs/development/timing_gate.md). The October 2 A/B driver forced
fp32 even for S=1, and its results must not be called auto-precision/default-S1 coverage. October 3
conservative-model dumps omitted some empty queries (2775/2771 of 2776); those groups require recollection.
The original measurements below are historical, not measurements of this revision.
Published numbers are the camera-ready paper's (RTX 4060) and live in
`docs/source/user_guide/benchmarks/results.rst`; the competitor columns have **not** been re-run on the current
code. Baselines are installed only in the local staging environment. Tracked evidence on the RTX 5090: the audit timing gate
(`docs/development/evidence/targeted_timing_2026-09-27/`, open-world B=2000 ≈ 1.3 ms fp64, kernel within 1%
before/after the audit) and the de-vendoring gate + HJCD-only paper rerun
(`docs/development/evidence/timing_gate_2026-10-02/`: landed code 0.1–1% faster than the previous main in all 18
A/B cells up to B=32000; Panda open-world 1.67–1.74 ms at B=100–2000, box_panda collision-free 1.9–2.3 ms, DoF
7/12/18/24 at B=1000 = 1.85/1.95/2.42/3.01 ms). `scripts/bench/run_paper_experiments.sh` (with `HJCD_REGEN=1`)
regenerates the paper protocol (~4.5 min HJCD-only; ~15 min with all baselines and `RUN_HARD=1`, see
`docs/development/evidence/paper_rerun_2026-10-03/`); `scripts/perf/timing_driver.py` is the
neutral two-endpoint A/B driver (alternate rounds, compare paired per-round medians);
`scripts/perf/run_all_timing_sweeps.sh` is the HJCD-only timing capture.

> **Build/test gotcha:** `ninja -C build` does NOT update the imported `.so` (it's the editable copy in
> site-packages). Always rebuild with **`scripts/setup/rebuild.sh`** (or `pip install -e . --no-build-isolation`).

## What next

**Current priority (2026-10-05):** completed timing reviewed; GRiD 8dccbfa and matching GLASS 9e57178
pulled. Revalidate and sign proof, preserve measured endpoints, assess whether compiled code changes need
targeted timing. No push or new timing without permission. G1–G4 are consumed. G5's generic collision-mesh
spherizer/report and the G6 foam preset mechanism now exist upstream; they are not automatically equivalent
to HJCD's historical visual-mesh experiments. GLASS point-transform/vote helpers are available but are not
called by the unchanged generated warp checker. Keep numerical-policy changes separate. The roadmap below
contains historical context. Visual meshes are an environment-only approximation with obstacle-shrink tolerances,
not physical ground truth; model substitution does not isolate every solver-policy difference.

The running roadmap is `docs/open-tasks/TODO.md` (local, gitignored agent scratch — recreate it from this list
if missing). Tracked project docs: this file, `docs/development/agent_debugging_guide.md`,
`docs/development/STARTUP_PROMPT.md`, `docs/source/user_guide/upgrading.md`,
`docs/source/user_guide/benchmarks/results.rst`, and the sphinx docs. In priority order:

1. **Re-record the GPU proof** after any `csrc/`/`tests/`/docs change (see *Testing and the GPU-proof gate*).
2. **Paper protocol rerun with baselines: DONE 2026-10-03** (`docs/development/evidence/paper_rerun_2026-10-03/`,
   summarised in `docs/source/user_guide/benchmarks/results.rst` "Rerun on the current code"). cuRobo v2 is a
   much stronger baseline than the paper's v0.7 and, once its collision spheres are active (the URDF-built robot
   had none — a harness defect fixed in `486b547`), it matches HJCD on the hard collision sets at similar
   latency. HJCD leads on open-world Panda latency (1.7 vs 6.4 ms), Fetch is cuRobo's, MMD is HJCD's.
   `benchmark/score_collision_oracles.py` re-scores stored configurations under sphere and FCL-mesh oracles;
   use it (not a single oracle) for any collision claim. **Fairness pass (same day, evening):** PyRoki is now
   collision-aware (foam spheres through its own API), the judges are `hull` (franka's convex collision
   meshes = MoveIt's geometry) / `visual` (true shape, the headline) / `hjcd`, `paper`, `curobo` spheres;
   `kitchen` + `table_bars` exported from the public MBM dataset (`benchmark/mbm_export.py`,
   `benchmark/problems/mb_extra_panda.json`); clearance ladder (`benchmark/make_clearance_ladder.py`). Under
   the true geometry every solver is 94–100 % on every set; on tight clearance the foam sphere model — not the
   search — is HJCD's limit (HJCD compiled on cuRobo's spheres, `benchmark/make_curobo_sphere_urdf.py`,
   matches cuRobo in the cage), and a residual ~10-point gap on the table sets is the post-solve filtering
   vs. collision-in-the-loop difference. Evidence: `docs/development/evidence/fairness_hardsets_2026-10-03/`.
   **Kernel follow-up (same night):** collision-aware early stop + seed ranking + repair round + success-band
   fallback (see *Conventions*) — the three dataset sets with misses went to 100 % and the ladder's table
   levels are within 0–4 points of cuRobo on foam's spheres; on cuRobo's spheres HJCD = cuRobo everywhere.
   Sphere-model fidelity tool `benchmark/make_bounded_bulge_spheres.py` (bulge/coverage vs the true meshes):
   conservative full-cover models (200–377 spheres) reduced observed success; their shared-GPU latency
   observations are not a valid performance claim. Foam
   stays the default; a broad→fine cascade (GRiD supports it) is the route if a conservative model is ever
   wanted. Open: the quiet-window A/B (open-world old vs new binary; `HJCD_CC_STOP=0` vs 3 in collision
   mode) + latency columns; HJCD accuracy not monotone in B (likely the same early-stop mechanism — re-measure).
3. **Two collision-scene protocols (ruled 2026-10-03).** MotionBenchMaker goals are `panda_hand` poses (the
   dataset's `goal_ik` puts `panda_hand` on them; the TCP is 105 mm further out). The paper's Table II used the
   *cylinder-snapped* target with the TCP frame, which is physically consistent for cylinder-grasp scenes
   (box, bookshelf, table) but not for `cage_panda` (no cylinders) or `box_panda_flipped` (goal 38 cm from
   any cylinder). `run_paper_experiments.sh` therefore runs Table II twice: the **paper protocol** on
   `box_panda` for continuity (`--target-mode cylinder`, TCP builds) and the **dataset protocol** on every set
   (`--target-mode goal`, `panda_hand` builds, `RUN_HARD=1`), all solvers validated by the same `hjcd` sphere
   oracle. The dataset protocol is the harder, honest benchmark: with the default TCP build and raw goals
   the accurate-rate over 64 problems was 47/64 (cage 0/8); at `panda_hand` it is 60/64 and the remaining
   misses are found at B=16000. Robustness work (informed second round, collision-aware refinement) is future
   work alongside floating-base support.
4. **Consume upstream collision primitives/models** — spec in
   `docs/development/upstream_asks_collision_2026-10.md` (G1 expose sphere tables, G2 warp-scoped `config_free`, G3
   broad→fine cascade, G4 parallel block path, G5 bounded-bulge spherizer + fidelity report, G6 named Panda presets);
   G1–G4 are integrated and HJCD's duplicate sidecar/checker are removed. G5/G6 remain upstream follow-ups.
5. **Upstream candidates (older):** grasptarget-offset FK (`ee_fk_warp`/`ee_fk_thread`/`ee_fk_suffix_thread`) and the
   batched pose-7 FK kernel → GRiD; the warp dogleg step and a `gn_step` variant that exposes diag(A)/g → GLASS.
6. **Branched-chain support** in `ee_fk_suffix_thread` (needs the parent table; the GRiD primitive is general).
7. **Self-hosted GPU runner** so `.github/workflows/test.yml` can leave manual-only mode.
