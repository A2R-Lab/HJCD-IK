# Upgrading and verified scope

This guide describes the audit-hardening changes relative to the earlier paper-era
API. Published paper tables and the research landing page remain historical results;
they are not benchmark measurements of every subsequent code revision.

## Changes callers should account for

| Area | Current contract / migration |
| --- | --- |
| Collision policy | Python defaults to explicit `collision_mode="hard"`: filter self/environment collisions. `soft` ranks environment penetration but does not promise collision freedom; `both` ranks and filters. |
| Environment variable | Neither the Python API nor the native solver reads `HJCD_CC_MODE` any more (the former `collision_mode="auto"` is gone). The benchmark CLI alone keeps that variable as the default of its `--collision-mode` flag; pass the flag explicitly for reproducible runs. |
| Returned count | Read `count`, and use arrays of shape `(count, ...)`. Filtering and duplicate removal may return fewer candidates than requested, including zero. |
| Pose/units | Targets and returned pose positions are meters; quaternions are `wxyz`. Position errors are **millimeters**, orientation errors **radians**. Finite nonzero target quaternions are normalized. |
| Success | Check both errors for the **same candidate**. A nonempty result is not a pose-accuracy certificate; collision checks certify neither a trajectory nor mesh-level safety. |
| Invalid input | Invalid sizes, poses, unsupported scenes and unavailable collision builds raise exceptions. Requests never silently fall back to open-world solving. Use `"obstacles": {}` for an explicit empty environment. |
| Native ownership | `Result<T>` is move-only and owns its host buffers. Remove manual `delete[]` calls; keep the result alive while reading its pointers. Recompile native callers against the new header. |
| Native signature | `generate_ik_solutions(target, batch, ...)` no longer takes the ignored `d_robotModel` argument; drop that `nullptr` from native callers. `collision_mode` is `0` (soft), `1` (hard, default) or `2` (both). Obstacle JSON must use `pose`; the legacy Euler `box`/`orientation_euler_xyz` forms are rejected. |
| CUDA failures | Checked runtime/model initialization failures raise exceptions, with best-effort cleanup. A context-invalidating CUDA error may still require restarting the process. |
| Concurrency | Python releases the GIL; native sampling/solving serialize shared state. This is safe concurrent calling, not concurrent GPU solve execution. Keep the CUDA context alive; `cudaDeviceReset` invalidates caches. |
| Determinism | Candidate identity is not run-to-run stable: identical calls may return different, equally valid candidates (the coarse search and single-solution refinement stop on a cross-block flag). Aggregate over targets or pin tolerances, never a specific configuration. |
| Statistics | Unmeasured collision/feasibility fields use `-1`, not a fabricated zero. The summary tool excludes these values. Do not combine soft and hard/both rows as one collision-success metric. |

The Python result dictionary and float64 I/O remain unchanged. `refine_fp64=-1`
uses fp64 refinement for one requested solution and fp32 for multiple solutions;
`0`/`1` force the precision. This is a policy, not a speed guarantee on every GPU.

For example, accept only candidates meeting both application tolerances:

```python
import hjcdik

target = hjcdik.sample_targets(1, seed=0)[0]
out = hjcdik.generate_solutions(target, batch_size=2000, num_solutions=4)
acceptable = (out["pos_errors"] < 1.0) & (out["ori_errors"] < 0.001)
print("candidates within 1 mm and 0.001 rad:", out["joint_config"][acceptable])
```

This open-world example does not check collisions. Sampling reachable targets does
not guarantee that they are collision-free either.

## Robot and collision identity

A wheel contains one compiled robot. `hjcdik.build_info()` reports its joint count,
collision capability, generated-header SHA256 and CUDA compiler version without
initializing CUDA. Regenerate and rebuild to change the model; use separate build
directories and environments when comparing robots.

The supported solver model is a fixed-base serial chain of 1–32 independent
revolute/continuous local-+Z joints, plus fixed links/tool frames. Other GRiD robot
classes are not implicitly supported. See {doc}`tutorials/custom_robot`.

The default Panda keeps +/-40 mm fixed finger origins. The frozen paper collision
reference uses +/-65 mm. `--collision-validation-model paper` retains historical
post-hoc checking; `hjcd` selects the independent URDF-derived current geometry.
Neither flag changes the compiled solver. Both Python checks are environment-only;
hard/both solver modes also check self-collision using generated exclusions.
Collision CSV/YAML sidecars identify the selected model and compiled header.

Exact-content scene caching now reuses the parsed JSON when changing scenes, while
rejecting stale geometry after changed contents or failed selection. It retains one
document/environment per device and precision specialization, not every scene ever
visited. Runtime improvement needs measurement on a quiet machine.

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
omits unneeded dynamics. These reduce duplicated build work, but do not establish a
measured clean-build speedup. No new headline performance or MMD claim accompanies
the correctness changes.

For before/after timing, use matched release builds on the same idle machine. Record
targets, robot/EE frame, geometry, precision, tolerances, output counts, initialization,
and error/success metrics alongside latency. Test repeated scenes and changing scenes
with both full and compact JSON. Do not equate a per-returned-solution collision
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
