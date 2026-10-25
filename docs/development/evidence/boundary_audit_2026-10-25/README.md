# API-boundary and maintenance audit — 2026-10-25

Current status: the approved A1–A4, B1 and negative-device fixes are implemented and verified;
see the [closure packet](../boundary_fixes_2026-10-25/README.md). Final timing remains pending.
The findings and probe outputs below are the preserved pre-fix audit, not current failures.

Reviewed source: `b1ef1dd` on `review/general-joints-closeout`; comparison baseline
`ce2a1cb`. The user approved the packaged-spheres fix and requested a broader audit,
including code bloat and drift. Only that approved behavior fix was implemented.
No dependency, native solver, generated header, model, benchmark or timing-stage change.
Nothing pushed. **Do not merge yet: four further corrections are recommended below.**

## Completed fix and verification

`Solver(spheres=...)` and `Solver("panda", spheres=...)` now reject every non-None sphere
override with `ValueError` and URDF-build guidance before compilation or CUDA work.
Default constructors remain unchanged. Named-runtime override rejection is also checked.
Six new cases cover both packaged constructor forms and preset/missing/empty values.

- Focused build/failure suite: 19 passed.
- Full signed suite: **339 passed**, no skips; receipt verified, committed as `0ff0571`.
- Source archive -> wheel -> isolated installation: passed on clean `b1ef1dd`.
  The minimal Panda phase checks the new failure without runtime-build extras installed.
  Rail build/load phases retain 32/32 cold and warm success in all four cells.
- Fresh Doxygen plus strict Sphinx: passed.
- No native rebuild/sanitizer rerun was needed for this Python constructor guard. Native
  source is byte-identical to the preceding verified closeout; its native/sanitizer evidence
  remains in [that packet](../closeout_2026-10-25/README.md), including the initial unexplained
  racecheck exit and subsequent clean combined repeat.

The signed suite certifies those test cases, not absence of defects. The audit probes below
found missing coverage despite the green suite. `report.json` records their actual outcomes;
`probes.py` reproduces them without timing or modifying the default artifact store.

## Findings, prioritized

### A1 — High: no-build URDF loading can silently select a different model

Location: `hjcdik/_solver.py:280`, `_build.default_name` and `load`.
The URDF constructor with `allow_build=False` resolves a display name, then loads that
name without comparing the requested URDF/sphere contents with the prepared artifact.

Reproduction: build the two-joint analytic example in a new private cache with its default
name and spheres; change the slider upper limit from 0.5 to 1.5 in the requested file;
construct from that file with `allow_build=False`. Construction succeeds with the **old
0.5 limit**, despite different input hashes. The same path does not bind sphere contents
either; changing geometry under a same-named request can select an unintended model.
GRiD correctly loads the explicitly named artifact; HJCD discarded the requested identity.

Recommended fix: in the URDF/no-build path, compare URDF hash, sphere hash/absence and exact
tool frame with GRiD's recorded artifact metadata before import. Mismatch should raise
`BuildError` and explain rebuilding or explicit name-based loading. Do not invoke codegen
or a compiler to validate a prepared load, and do not introduce another cache.
Tests: unchanged and relocated-identical inputs load without build calls; changed URDF,
changed spheres and changed target fail before CUDA. Preserve explicit name-only loading.

### A2 — High: codegen silently ignores an explicit collision model

Location: `scripts/codegen/generate_grid.py:87`.
`--spherized-urdf` is only processed inside `if args.collision`. Passing a valid sphere
model without `--collision` succeeds but emits neither `HJCD_HAS_COLLISION` nor the
`grid_collision` namespace. The public solver defaults to open-world for that build.
The runtime builder instead treats a sphere model as a collision request.

Reproduction: generate UR5e with `-t flange-tool0 --spherized-urdf <shipped spheres>`,
omitting `--collision`. Exit 0, collision sentinel false, collision namespace false.
This is HJCD's CLI policy, not a GRiD geometry failure.

Recommended fix: reject the contradictory CLI combination early with a precise instruction
to add `--collision`; do not silently change explicit opt-in semantics. Test valid and
nonexistent sphere paths in that combination, preserve both existing headers on rejection,
and retain a positive generation test with both options. Review other ignored option
combinations in the same pass, without broadening accepted robot classes.

### A3 — High: warm ranking can discard accurate candidates it already found

Location: `csrc/kernel/hjcd_kernel.cu:2326` and `2342`.
Initialization clamps warm starts, but proximity ranking reads the original unclamped
values. It also encodes a categorical preference with a finite sentinel (`1e6 + score`).
For sufficiently distant seeds, inaccurate candidates therefore sort ahead of converged ones.

Reproduction: packaged Panda, four sampled targets (seed31), B64/S4, open-world,
`q_init = full((1,7), 1e7)`, stats enabled. The saved run has accurate refined-pool counts
of [178,123,283,192] but returned accurate counts **[0,0,0,0]**. Using the same seeds clipped
to the documented joint limits yields **[4,4,4,4]**. Counts can vary with accepted solver
nondeterminism; the selection failure is not inferred from timing or a single coordinate.

Recommended fix: use the effective clamped seeds consistently and rank lexicographically
by convergence class, then the existing proximity/accuracy criteria. Do not substitute a
larger sentinel. Retain existing norm, tolerances, collision policy and ordinary bounded
seed behavior; this is not a unit-preconditioning or solver-tuning experiment.
Tests: large positive/negative finite seeds versus explicit clamping; accurate candidates
must outrank inaccurate ones even for large valid prismatic coordinates; both precisions,
hard/open-world, no NaN/overflow sorting and nonfinite inputs still rejected.
Native source/build identity changes require full correctness and targeted sanitizer checks.
Prepare affected timing endpoints, but defer any new timing to a quiet CPU/GPU window.
Prior performance evidence is not automatically proof for that future native edit.

This does not invalidate the existing normal-seed experiment counts. It demonstrates an
accepted-input failure outside those sets, not universal convergence or collision failure.

### A4 — Medium: custom packaged builds can report the wrong tool frame

Location: `CMakeLists.txt:189`.
CMake's target-name regex only captures `[A-Za-z0-9_]+`, unlike the accepted/generated
URDF names and runtime-build metadata. Applying the actual CMake parser to the actual
generated UR5e header maps **`flange-tool0` to `flange`**. The numeric FK target can remain
correct while `build_info`/`Solver.info.ee_frame` misidentify it, undermining frame checks.

Recommended fix: parse the complete quoted target or consume canonical model metadata,
then verify consistency with the paired joint-model header. Avoid adding a second metadata
schema. Add a CPU configuration/parser test with hyphen/dot names and a custom-build
metadata/FK smoke test. The default Panda path must remain unchanged.

### B1 — Medium follow-up: world quaternion validation is not scale robust

Location: `hjcdik/_solver.py:182`.
A finite, nonzero cylinder quaternion `[1e-200,0,0,0]` passes validation but its norm
underflows, so `world_arrays` returns nonfinite capsule endpoints. The later native upload
rejects nonfinite rows; this is not evidence that a bad world was silently installed.
Normalize by the largest absolute component before the norm and validate converted output.
Test equivalent small/large quaternion scales, boxes as well as cylinders, and atomic
preservation of the prior world on errors. Do not weaken the zero-quaternion check.

### B2 — Medium/low follow-up: scalar input validation drifts across entry points

Location: `hjcdik/_solver.py:245` and `sample_targets`/`solve`.
`Solver(device=-1)` is accepted and records -1; native `DeviceScope` interprets negative
indices as “use whichever device is current”, contradicting the fixed-device contract.
`sample_targets(1.9)` silently returns one target; solve sizes also use truncating `int()`.
This machine did not exercise cross-device migration, so no multi-GPU failure is claimed.
Reject negative device indices before build/CUDA work. Explicitly choose the integer-input
policy for floats/bools/NumPy integers and apply it consistently, with focused CPU tests.
Do not incidentally change every coercion rule as part of the already-approved fix.

## Bloat, duplication and drift assessment

The architecture is mostly sensible: public Python policy/result types, one native binding,
one CUDA solver, GRiD owning generated kinematics/collision and the artifact store, GLASS
owning algebra. There is no need to restore a collision sidecar or replace these with a new
framework. The Python package is about 1,130 lines, bindings314, solver2491; these are
inventory measurements, not runtime or quality metrics. Most recent diff volume is model
assets, frozen witnesses and evidence, not speculative abstraction.

| Area | Assessment / bounded improvement |
|---|---|
| Runtime builds | `_prepare_robot` emits kinematics once to discover the target, then GRiD generates the artifact header. Real duplicated generation, not duplicate solver logic. Use upstream structured target metadata if made public; no new GRiD work or speculative cache now. |
| Metadata | CMake, source codegen and runtime paths describe the same model differently; A4 is an actual consequence. Keep `hjcdik/_robot_model.py` the shared model authority and add cross-path parity tests. |
| Collision wrappers | Warp and block adapters repeat environment-only-margin policy, but have different synchronization contracts. Do not combine them mechanically or reimplement GRiD. Prefer policy parity tests over risky “deduplication”. |
| Test helpers | Rail tests reuse independent FK/geometry helpers, which is good; importing helpers out of `test_general_joints.py` couples test modules. A small non-collected helper module is reasonable later, not a new testing framework. |
| Dead residue | AST inventory finds unused `Sequence`, `Union`, `BuildError` imports in `_solver.py`, and `os`, `Sequence` in `_build.py`. Native `TargetFailure.index` is stored but Python parses the message. Tiny cleanup/structured-error opportunity, not meaningful bloat. |
| Drift | `generate_grid.py:161` still names deleted `grid_env.cuh`; native collision-policy comment near1961 uses obsolete enum numbers; package metadata says “hierarchical” while README says “Hybrid Jacobian”. Update these as one small docs/comment batch, preserving generated headers and code. |
| User docs | Stronger after closeout: explicit counts versus accuracy, frame/units, model limitations and historical versus current reproduction. Still favor runtime builds as the main custom-robot journey; keep header replacement as the advanced/native route. |
| Plans/evidence | Local ledgers repeat large historical status sections. Keep one current status pointer and archive history; do not rewrite archived measurements to sound current. Tracked evidence should stay reproducible, not become runnable package dependencies. |
| Packaging/CI | Isolated wheel checks and signed manifests are useful. Python3.12/CUDA13.2 is verified, not the entire declared matrix. GPU CI is manual; remote CI has not run for these unpushed commits. Dependencies have lower bounds, not a proven future-version compatibility guarantee. |

## Scope of this review

Read the public API/build/model/CLI/world paths; native binding and host selection/ownership;
codegen/CMake contract and flags; relevant tests, installed harness, timing-gate entry points,
README/API/migration docs and CI workflows. Checked preserved solver/dependency boundaries,
cached module identity protection, world/result ownership, packaging isolation, generated
header pairing and existing quality/evidence protocol. No new ownership defect was established
in those inspected paths; that is not a formal proof of every thread/platform/error case.

Bounded dynamic probes cover the findings above and use an independent temporary cache.
No full historical-paper rerun, new broad robot campaign, multi-GPU run, dependency update,
kernel tuning or performance measurements. No claim that code bloat caused a measured slowdown.

## Proposed next wave and gates

1. Approve A1–A4, plus B1 and negative-device validation, as a bounded correctness batch.
   Set integer-coercion and reserved preset/artifact-name policy explicitly rather than
   silently changing it. Preset/custom-name collisions merit an additional resolver test.
2. Add the missing invariant tests first: requested model identity must match, supplied
   collision options cannot disappear, clamped seeds have consistent meaning, tool names
   round-trip, quaternion scaling preserves geometry, validation precedes GPU work.
3. Implement narrow HJCD fixes; no dependency re-pin, new API abstraction, collision model,
   numerical tolerance or robot-support expansion. Small proven dead-import/comment cleanup
   may accompany them; leave structural reorganization until after correctness closes.
4. Rebuild affected artifacts; run focused tests, full signed suite, strict docs, installed
   workflow, native tests and sanitizer coverage proportionate to native changes. Commit
   source -> proof -> verify -> receipt; keep findings marked open until their tests pass.
5. Stage the final native candidate for targeted comparison against preserved evidence.
   Timing only in a newly announced quiet window; publish through review branch/PR only
   after explicit user approval and remote CI. Do not merge merely because the old suite is green.

## Reproduction and evidence

From the repository root, with the existing development environment and submodules:

```bash
.venv/bin/python docs/development/evidence/boundary_audit_2026-10-25/probes.py --out /tmp/hjcd-audit-NEW
.venv/bin/python scripts/setup/check_installed.py --out /tmp/hjcd-installed-NEW
scripts/setup/run_gpu_proof.sh
.venv/bin/gpu-proof verify --receipt gpu-proof.json --repo . --policy tests/gpu-proof-policy.yaml
```

The probe runner intentionally reports existing failures; it is not a passing regression test.
It retains generated inputs and the private artifact cache at its chosen new output directory.
Current raw working directory: `/tmp/hjcd-boundary-audit-20261025`.
The committed reports record source revision/hashes and exact installed commands; `SHA256SUMS`
binds the preserved outputs. Probe outcomes should change when the corresponding fixes land.
