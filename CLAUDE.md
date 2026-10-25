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
   Jacobian (revolute cross-products or prismatic translation columns), form & solve the normal equations
   `(JᵀJ + λ·diag)Δq = Jᵀr` via GLASS warp linear algebra, with dogleg/line-search backtracking.

Forward kinematics produces the **world-frame joint transforms** `s_jointXforms[16·jid]` (4×4 each); the EE
pose error is computed as a **quaternion** error (`mat_to_quat` / `quat_err_rotvec`). For Panda: `N = 7`
joints. Target indices and transform counts are generated constants; the default
`panda_grasptarget_hand` build currently has `grid::EE_FIXED_FRAME_IDX = 10`.

## Key files

| Path | What it is |
|---|---|
| `csrc/kernel/hjcd_kernel.cu` | The solver: block-cooperative coarse search + warp-scoped LM refine + host orchestration (`generate_ik_solutions`). **The file you'll edit most.** |
| `csrc/kernel/hjcd_settings.h` | `HJCDSettings<T>` (coarse/LM tolerances, `lambda_init`), `mat4_mul`, FK helpers (`ee_fk_warp`/`ee_fk_thread`/`ee_fk_suffix_thread`), `#include "grid.cuh"`, `N`/`FLANGE_JID`/`GRASP_FIXED_IDX`. |
| `csrc/kernel/hjcd_kernel.h` | Native host API: move-only `Result<T>` (with `SolveStats`), `CollisionMode`, `WorldSpec` → `make_collision_world` (owning device world = GRiD's `grid_collision::DeviceEnvironment`), `generate_ik_solutions<T,RT>(target, batch, num_solutions, mode, world, collect_stats)`, `sample_random_target_poses`. |
| `csrc/kernel/main.cpp` | Native CLI (`single`/`sweep`/`from_csv`), open-world only; CTest-covered by `tests/native/`. |
| `csrc/generated/grid.cuh` | **Generated** GRiD kinematics + collision header. Do **not** hand-edit. Generated with `vendor_glass=False`, so it `#include "glass.cuh"`s the top-level GLASS instead of vendoring a copy (~6k lines, was ~15k). |
| `external/GRiD/` | Submodule: GRiD codegen (emits `grid.cuh` from a URDF). Its nested GLASS pin must equal `external/GLASS`. |
| `external/GLASS/` | Submodule: GLASS single-block / warp / thread linear algebra (`glass::warp::`, `glass::thread::`, `glass::block::`). |
| `external/foam/` | Submodule: pre-spherized Panda collision URDF (`assets/panda/smaller_panda_spherized.urdf`). |
| `csrc/bindings/pybind_module.cpp` | Low-level bindings (`World`, `solve_one`, `sample_targets`, `current_device`, `build_info`); every call runs on the solver's device and restores the caller's. Compiled by CMake into the packaged `_hjcdik` AND by `grid_rbd.build_artifact` into per-robot artifacts (`HJCDIK_MODULE_NAME` define = unique init symbol; `HJCDIK_BINDING_ABI` checked on import). The user API is pure Python: `hjcdik/_solver.py` (`Solver`, `Result`, `TargetResult`, `Stats`, `Info`, `SolveError`), `hjcdik/worlds.py` (MotionBenchMaker JSON → `set_world` arrays), `hjcdik/_build.py` (`build`/`load`/`list_builds` over GRiD's consumer-artifact store; `-DHJCDIK_EE_FIXED_FRAME_IDX` resolved from a kinematics-only generation), `hjcdik/__main__.py` (`hjcdik build|list|info`). |
| `csrc/collision/panda_foam_rebound.urdf` | foam's Panda spheres re-attached to `csrc/urdf/panda.urdf` by `grid-spherize --rebind-to` (GRiD V1); validated strictly; generates a header byte-identical to the foam path; the `"panda_foam"` preset for runtime builds. The wheel ships `csrc/` (minus `generated/`) as `hjcdik/_csrc` for runtime builds. |
| `csrc/urdf/iiwa14.urdf`, `csrc/collision/iiwa14_spheres.urdf` | the `"iiwa14"` preset (`hjcdik.presets()`, `Solver("iiwa14")` builds on first use): Drake's iiwa14 URDF (BSD, attribution in the file) and GRiD's spherizer output with its fidelity report in the header comment. |
| `hjcdik/_robot_model.py`, `csrc/generated/hjcd_joint_model.h` | Shared model validation and generated ordered joint axes/types/limits. Arbitrary-axis revolute, continuous and prismatic joints, 1–32, fixed-base serial only. CMake verifies the side header's GRiD hash; runtime builds stage it under a stable content-derived input path. |
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

Python API (since 2026-10-23; `docs/source/api_reference/python.rst` is the contract):
```python
import hjcdik
solver = hjcdik.Solver()                                  # the compiled robot, bound to the current device
targets = solver.sample_targets(10, seed=0)              # (10, 7) [x,y,z, qw,qx,qy,qz]
solver.set_world(boxes=..., capsules=..., spheres=..., cylinders=...)   # robot base frame, metres, wxyz
out = solver.solve(targets, batch_size=2000, num_solutions=4)          # Result: (T, S, ...) NaN-padded
# out.count (T,), out.valid (T,S), out.ok(pos_mm, ori_rad), out[i] = un-padded TargetResult
# collision omitted = "hard" on a collision build (self + env filtering); None = open-world; "soft"
# ranks only (needs a world); "both" ranks + filters. precision="auto"|"fp32"|"fp64"; stats=True → out.stats
```

## Conventions / discipline

- **Never hand-edit `grid.cuh` or `hjcd_joint_model.h`.** They are generated as a matched pair. To change the robot or EE target, run
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
  ranking mode and does not guarantee collision freedom; `both` ranks and filters. Nothing reads
  `HJCD_CC_MODE` any more (API, native solver or benchmark CLI); pass `--collision-mode` to the benchmark.
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
[`docs/development/history/audit_hardening_validation.md`](docs/development/history/audit_hardening_validation.md).

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

**Performance status (October 2026).** Every claim below is a same-machine (RTX 5090, CUDA 13) quiet-window
measurement with the protocol in [`docs/development/timing_gate.md`](docs/development/timing_gate.md)
(`scripts/perf/timing_driver.py` two-endpoint A/B, `scripts/perf/run_audit_gate.py`). Published paper numbers
(RTX 4060) live in `docs/source/user_guide/benchmarks/results.rst` and are historical.
- Audit A/B ([`audit_timing_2026-10-05`](docs/development/evidence/audit_timing_2026-10-05/README.md)): identical
  matched quality counts, 19/20 cells faster, one unchanged. GRiD/GLASS re-pin confirmed performance-neutral
  ([`postpin_timing_2026-10-06`](docs/development/evidence/postpin_timing_2026-10-06/README.md), 12 paired ratios
  within 0.9946–1.0046). The 2026-10-12 re-pin to GRiD v0.5.1 changed only a block-checker race fix; the 2026-10-23 re-pin to v0.5.5 and the
  2026-10-24 re-pin to v0.6.0 changed no kernel code HJCD executes (linalg-argument removal; plant block dropped from the
  kinematics header; checker defaults `margin=0`, `check_self=true` are the old verdict). The 10-24 API gate
  ([`api_timing_2026-10-24`](docs/development/evidence/api_timing_2026-10-24/README.md)): native entry points within
  0.6 % of the pre-API build, `Solver.solve` adds ~10 µs host time per target, collision scenes 13–16 % faster (no
  per-call JSON compare), runtime-built Panda == packaged, 2 cm margin ≈ +2–3 % while solvable.
- Full campaign on the shipped code, all baselines ([`paper_rerun_2026-10-07`](docs/development/evidence/paper_rerun_2026-10-07/README.md),
  results.rst *Current code — full campaign*): Panda open-world 1.62–1.65 ms at B ≥ 100 (cuRobo v2 5.9–6.4,
  PyRoki 6–7.2, IKFlow 2.5–6.2); Fetch 0.81–1.10 vs cuRobo 1.8–2.1; DoF 7/12/18/24 1.76/1.87/2.31/2.85 vs
  2.4/2.2/2.5/2.7; MMD 0.098 (best); collision scenes 1.8–3.2 ms at 99–100 % on all ten sets (cuRobo 2.4–2.5 ms
  at 98–100 %; collision-aware PyRoki 27–38 ms). `scripts/bench/run_paper_experiments.sh` reproduces the protocol
  (`docs/source/user_guide/benchmarks/reproduction.md`).
- Caveats on record: the 2026-10-02 A/B forced fp32 at S=1 (not auto-precision coverage); the 2026-10-03
  conservative-model dumps omitted empties and were recollected 2026-10-07 (same rates); shared-GPU latency
  observations are never performance claims.

> **Build/test gotcha:** `ninja -C build` does NOT update the imported `.so` (it's the editable copy in
> site-packages). Always rebuild with **`scripts/setup/rebuild.sh`** (or `pip install -e . --no-build-isolation`).

## Status and what next (updated 2026-10-25)

P2b general axes and prismatic joints are implemented locally. The all-revolute-+Z path retains
Panda's arithmetic. Independent FK/Jacobian, collision, mixed-unit scale and module-isolation
checks are in `tests/test_general_joints.py`; quality evidence is under
`docs/development/evidence/general_joints_2026-10-25/`. Mixed-joint LM and proximity use raw
coordinates; the finite tested scale range is not a universal convergence guarantee. Timing
against pre-P2b `3f5ec5a` remains pending a user-announced quiet window. Local commits are not
authorization to push. Fetch upstream preset assets require a distribution-license decision.

**Where the code is.** `origin/main` carries the closed fairness/hardening arc (2026-10-03 → 10-07) and the
re-pin to GRiD **v0.6.0** (commit `712df1c`, the PyPI `grid-rbd` 0.6.0 release, 2026-10-24, which shipped every HJCD ask:
G7 margin, G10 self toggle, G8a `DeviceEnvironment`, G9 `build_artifact`/`load_artifact`, V1 rebind, V2, V3, E1;
nested URDFParser `f391741`; GLASS `9e57178`, top-level and nested; pytest-gpu-proof 0.4.1 is still the latest release).
All six upstream asks (G1–G6, `docs/development/upstream_asks_collision_2026-10.md`) are in GRiD; HJCD has no
collision sidecar. GRiD's G5 fidelity report measures against collision (convex-hull) meshes, so
`benchmark/make_bounded_bulge_spheres.py --report` remains the visual-mesh fidelity tool of record.

**Rulings that are not derivable from the code** (keep; they explain the benchmark design):
- Two collision-scene protocols: the paper's cylinder-snapped TCP target on `box_panda` for continuity, and the
  MotionBenchMaker dataset protocol (`panda_hand` goals, every set, `RUN_HARD=1`) as the honest benchmark.
- The true-geometry (`visual` FCL mesh) judge is the headline; every sphere model is optimistic somewhere, so
  solvers are compared on identical spheres and the sphere model's bulge/coverage is stated.
- Foam stays the default sphere model. Conservative full-cover models (b10/b15) lower success and were not
  adopted; a broad→fine cascade (GRiD G3) is the route if one is ever wanted.
- Nullspace repair (C2) was implemented, measured and NOT shipped: the solver never returns an accurate
  configuration its own spheres reject (`docs/development/evidence/repair_nullspace_2026-10-07/`). Success is
  monotone in B with the collision-aware stop (`c1_batch_monotonic_2026-10-07/`).
- Non-determinism is accepted; diagnostics come back as `Result.stats` and the benchmark writes the legacy
  `ik_stats.csv` columns from them (`benchmark/_results_io.py`); the solver never opens files.

**What next — make HJCD-IK a usable package** (user ruling 2026-10-07; the local ledgers under
`docs/open-tasks/` hold the detail and are gitignored):
1. **User-shaped API (P1).** Stage 0 (contract) and Stage 1 (`Solver` over the installed robot, array
   worlds, padded `Result`, `precision`, `Result.stats`; `generate_solutions` removed) landed 2026-10-23.
   Stage 2 (runtime robot builds over GRiD 0.6.0's `grid_rbd.build_artifact`/`load_artifact`: `hjcdik build`,
   `Solver("<name>")`, `Solver("<urdf>", ee_link=...)`, `hjcdik[build]` extra, packaged Panda unchanged)
   landed 2026-10-24. Stage 3 (`collision_margin_m` on environment pairs only, self pairs at zero; `self_collision=
   False`; `q_init` warm starts (accurate ones skip coarse descent, converged candidates ranked by proximity);
   `seed`) landed 2026-10-24 too. GRiD's single `config_free` margin also widens self pairs, which rejects every
   Panda configuration above ~1 cm, so HJCD calls it twice when a margin is set (ask G7b: separate self margin).
   Plan: local `docs/open-tasks/api_proposal_v2_2026-10-13.md` (v2.2).
2. **Robots/presets (P2):** Panda, Fetch from `fetch_description` (HJCD's `csrc/urdf/fetch.urdf` differs from it
   by up to 37 cm — do not reuse it for collision sets), UR/iiwa/xArm/Gen3; prismatic joints in the coarse search.
3. **In-kernel multi-target (P3)** — needs a timing window to claim anything.
4. **Distribution (P4):** wheels, self-hosted GPU runner for `test.yml`, versioned API docs, DLPack interop.
5. **Robustness (P5):** cascade for conservative models, exposed margins.
Housekeeping runs alongside (de-bloat, docs truth, one receipt per batch).

**Standing rules.** Timing only on an empty GPU in a user-announced window (`timing_gate.md`); correctness
collection may share the GPU. Receipt discipline: commit → `scripts/setup/run_gpu_proof.sh` on a byte-clean
checkout → verify → commit `gpu-proof.json` (`update_gpu_proof_manifest.py` when test IDs change). Push only
with the user's say-so, over SSH. Never hand-edit generated headers. Kill by PID only.
