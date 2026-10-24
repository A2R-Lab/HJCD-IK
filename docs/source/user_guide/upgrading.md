# Upgrading and verified scope

This page records the API contract of the current release and how to move code written against
the earlier function API. Published paper tables and the research landing page remain historical
results; they are not benchmark measurements of every subsequent code revision.

## The object API (October 2026)

`generate_solutions`, `sample_targets`, `num_joints`, `collision_enabled` and `write_stats` are gone.
The package exposes `Solver`, `Result`, `TargetResult`, `Stats`, `Info`, `SolveError`, `build_info`,
`world_arrays` and the `worlds` loaders; see {doc}`../api_reference/python` for every field.

| Area | Current contract / migration |
| --- | --- |
| Entry point | `solver = hjcdik.Solver()` is the packaged robot, bound to the current CUDA device. `Solver("<name>")` loads a prepared runtime build; `Solver("arm.urdf", ee_link=...)` builds one through GRiD's artifact store (`hjcdik[build]` extra, `hjcdik build`). |
| One call, many targets | `solver.solve(poses)` takes `(7,)` or `(T, 7)`; `batch_size` is still the number of candidates per target. The whole batch is validated before any GPU work. |
| Collision default | Omitted `collision` means `"hard"` on a collision build (self and environment filtering) and `None` on an open-world build. The old default was collision checking off: pass `collision=None` to keep it. `"soft"` ranks without filtering and raises on an empty world; `"both"` ranks and filters. |
| Worlds | Obstacles are arrays installed once with `solver.set_world(boxes=, capsules=, spheres=, cylinders=)`, not JSON passed on every call. `hjcdik.worlds.from_mbm_problem` converts a MotionBenchMaker instance. A failed `set_world` leaves the previous world intact; `clear_world()` installs the empty world. |
| Results | `Result` arrays are `(T, S, ...)`, NaN-padded to `S = num_solutions`, read-only, with `count`, `valid`, `ok(pos_mm, ori_rad)` and `result[i]` for the un-padded view. Filtering and duplicate removal may leave fewer valid rows than requested, including zero. |
| Precision | `precision="auto" \| "fp32" \| "fp64"` replaces `refine_fp64=-1 \| 0 \| 1`. Auto is fp64 for one solution and fp32 otherwise; I/O stays float64. |
| Diagnostics | `solve(..., stats=True)` returns `Result.stats`; the solver no longer writes `ik_stats.csv`. The benchmark writes the same CSV columns from `Result.stats` (`benchmark/_results_io.py`). Unmeasured fields are `-1`. |
| Errors | `ValueError` for shapes, values, geometry and unsupported collision requests; `SolveError` (a `RuntimeError`, with `target_index` and `completed`) for CUDA failures, after best-effort cleanup. |
| Devices and threads | Every operation runs on the solver's device and restores the caller's current device. Calls are serialised; `set_world` waits for a solve in flight, so one world applies to a whole batched call. |
| Pose/units | Targets and returned poses are metres and `wxyz` quaternions; position errors are **millimetres**, orientation errors **radians**. Finite nonzero target quaternions are normalised. |
| Success | Check both errors for the **same candidate**. A valid row is not a pose-accuracy certificate; collision checks certify neither a trajectory nor mesh-level safety. |
| Native API | `generate_ik_solutions(target, batch, num_solutions, CollisionMode, const CollisionWorld*, collect_stats)` replaces the JSON triple and `write_stats`; worlds come from `make_collision_world(WorldSpec)`, which wraps GRiD's `grid_collision::DeviceEnvironment`. `Result<T>` is move-only and carries `stats`. The JSON parser and the `nlohmann-json` dependency left the native layer. |
| Other robots | Regenerating the committed header and rebuilding the package is no longer the only route: `hjcdik build <urdf> --ee-link <joint> [--spheres <preset or spherized URDF>]` compiles the robot once into GRiD's store and `Solver("<name>")` loads it without a compiler. The committed Panda header and the submodules stay the fixture of the packaged build. |
| Determinism | Candidate identity is not run-to-run stable: identical calls may return different, equally valid candidates. Aggregate over targets or pin tolerances, never a specific configuration. |

## Migrating a call

```python
import hjcdik

solver = hjcdik.Solver()
target = solver.sample_targets(1, seed=0)[0]
# old: out = hjcdik.generate_solutions(target, batch_size=2000, num_solutions=4)
out = solver.solve(target, batch_size=2000, num_solutions=4, collision=None)
r = out[0]
acceptable = r.ok(pos_mm=1.0, ori_rad=0.001)
print("candidates within 1 mm and 0.001 rad:", r.q[acceptable])
```

A collision-free call used to pass the problem JSON, set name and index on every solve. Now the
scene is installed once:

```python
problem = hjcdik.worlds.select_problem(document, "box_panda", 0)
solver.set_world(**hjcdik.worlds.from_mbm_problem(problem))
out = solver.solve(target, batch_size=2000, num_solutions=4)      # "hard" by default here
```

The open-world example above does not check collisions. Sampling reachable targets does not
guarantee that they are collision-free either.

## Robot and collision identity

A wheel contains one compiled robot. `hjcdik.build_info()` reports its name, joint count,
collision capability, named EE target, generated-header SHA256 and CUDA compiler version without
initializing CUDA; `solver.info` repeats them. Regenerate and rebuild to change the model; use
separate build directories and environments when comparing robots.

The supported solver model is a fixed-base serial chain of 1–32 independent
revolute/continuous local-+Z joints, plus fixed links/tool frames. Other GRiD robot
classes are not implicitly supported. See {doc}`tutorials/custom_robot`.

The default Panda keeps +/-40 mm fixed finger origins. The frozen paper collision
reference uses +/-65 mm. `--collision-validation-model paper` retains historical
post-hoc checking; `hjcd` selects the independent URDF-derived current geometry.
Neither flag changes the compiled solver. Both Python checks are environment-only;
hard/both solver modes also check self-collision using generated exclusions.
Collision CSV/YAML sidecars identify the selected model and compiled header.

## Validation and limits

The audit exercised Linux/Python 3.12, CUDA 13.2 and RTX 5090, including full Python
contracts, independent FK/collision checks, CUDA failure injection, focused memory
and synchronization sanitizers, native API/CLI checks, and source/wheel builds.
Custom-model correctness included no-collision Panda, Fetch and 12/18/24-DoF arms.
These checks are not certification for physical deployment.

The package declares Python 3.9+ and CUDA 12.x/13.x support, but this audit did not
exercise every version/architecture combination. Development setup and signed
proof tooling require Python 3.11+. Windows-native, multi-GPU runtime, context reset,
and other robot classes are not part of the verified matrix.

GPU results are recorded in `gpu-proof.json`; its manifest covers the full reviewed
suite, source and dependency pins, plus executable examples/documentation. Native
CTest, sanitizer and build checks are separate evidence, not implied by that receipt.

## Performance and paper comparisons

The CUDA core is compiled once for Python and the CLI; the default generated profile
omits unneeded dynamics. No new headline performance or MMD claim accompanies the API
change: the kernel is unchanged, and the latency comparison between `Solver.solve` and the
removed `generate_solutions` runs with matched collision settings in an announced timing window
(`docs/development/timing_gate.md`).

For before/after timing, use matched release builds on the same idle machine. Record
targets, robot/EE frame, geometry, precision, tolerances, output counts, initialization,
and error/success metrics alongside latency. Do not equate a per-returned-solution collision
percentage with per-query success, or compare a median with the paper's mean.
See {doc}`benchmarks/results` for the unchanged published results.

## Common setup problems

- **Wrong/stale robot:** inspect `hjcdik._hjcdik.__file__` and `build_info()`. Rebuild
  with `python -m pip install -e .`; `ninja -C build` alone does not update the
  editable-installed extension.
- **Missing codegen modules or meshes:** run `bash scripts/setup/bootstrap.sh` and
  install `.[codegen]`. Default Panda collision uses the bundled pre-spherized foam
  URDF, not unresolved mesh paths. Custom mesh codegen needs resolvable mesh assets.
- **No visible GPU at configure time:** set `CUDAARCHS` to the deployment GPU's
  architecture, as described in {doc}`getting_started/installation`. This does not
  enable CPU execution; solving still requires a compatible CUDA GPU/runtime.
- **Zero collision-free results:** check the compiled model, scene/frame, and pose
  accuracy. Increasing the candidate batch may help; switching to `soft` is not an
  equivalent collision-free solve.
