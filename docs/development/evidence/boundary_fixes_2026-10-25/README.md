# Boundary correctness closeout

Scope: the approved fixes following the [boundary audit](../boundary_audit_2026-10-25/README.md),
starting from `9f3e017` on `review/general-joints-closeout`. Implementation is `70bbb70`;
`251832e` adds the prepared-artifact upgrade note without changing implementation.
Correctness is closed; final performance approval is pending a quiet window. Nothing pushed.

## Findings closed

| Finding | Change and evidence |
|---|---|
| A1 stale no-build URDF load | Compare URDF hash, sphere hash/presence and exact tool against GRiD's existing artifact metadata before import. Mismatch or missing required identity raises actionable BuildError. Real collision/open-world builds and mocked changed/missing/relocated identities are covered; explicit prepared-name semantics stay unchanged. |
| A2 ignored sphere option | Source codegen rejects `--spherized-urdf` without `--collision` before writing either header. Real/missing sphere paths reject; the valid collision combination still emits its sentinel and namespace. |
| A3 warm selection | Initialization and proximity ranking share the effective clamped coarse-precision seeds. Convergence is a category rather than a finite score offset. Direct ordering tests cover large distances, nonfinite scores and ties; all 18 extreme-seed GPU regressions failed before the fix and pass afterwards, across precisions/collision modes and per-target seeds. |
| A4 truncated tool name | CMake captures the full quoted name. Actual configuration checks cover hyphen/dot/Panda names; an isolated custom UR5e extension reports `flange-tool0` through both APIs and agrees with independent FK. |
| B1 quaternion scale | Normalize world quaternions after max-component scaling, including boxes before upload; validate converted cylinder endpoints against fp32 range. Tiny/large scales, unchanged input arrays and preservation of the previous world after rejection are covered. |
| B2 negative device | Reject negative integer device indices before model resolution/build/CUDA. NumPy integer device0 remains accepted. Broader scalar coercion policy is deliberately unchanged. |

No dependency re-pin, generated-default-header edit, model substitution, solver tolerance,
precision policy, collision policy, metric or supported-robot expansion. Refined-pool preference,
coarse top-up and duplicate tolerance remain unchanged. Small unused-import/comment corrections
and the package-name expansion remove drift; no cache/framework/sidecar was added.

The new errors are intentional behavior changes. Old prepared artifacts retain their original
compiled code: rebuild from original inputs with the updated package, then restart processes
that imported the old artifact. No-build validation checks model identity, not automatic code
upgrade. See the [migration guide](../../../source/user_guide/upgrading.md).

## Validation

- Stage1: `523ea72`, 353 tests passed, verified receipt `4d86601`.
- Stage2: `552ab06`, 372 tests passed, verified receipt `3d1a6f7`.
- Final implementation `70bbb70`: 383 passed, verified receipt `39d8ed3`.
- Final documentation checkpoint `251832e`: 383 passed; verified receipt `f9f48f6`.
- Focused final integration: 55 passed, including the custom packaged UR5e check.
- Native API/CLI: 2/2 passed; generated header pair freshness passed.
- Fresh Doxygen and strict Sphinx passed; final docs were rebuilt after the migration note.
- Final-binary warm-start memcheck, synccheck and racecheck: two precision cases each,
  process exit0, zero errors; racecheck zero hazards/warnings. Earlier closeout sanitizer
  history remains in its original packet; no claim that its unexplained exit was diagnosed.
- Clean-checkout source archive -> wheel -> isolated installation report and exact commands
  are included. Panda works without runtime-build extras; rail build and fresh-process load
  check cold/warm solves, blocked/restored worlds, unchanged model identity and changed-URDF rejection.

One custom smoke initially lacked the C++ diagnostic flag while enabling it for CUDA; this
was a test configuration error, corrected before the passing focused and signed runs. The
first installed run passed but observed the receipt temporarily absent during concurrent
proof regeneration; it was repeated after receipt commit for clean packaging provenance.
Earlier temporary outputs were retained, not silently substituted for clean evidence.

The tests retain their original thresholds; expected parser warnings are preserved. Large
joint-space ranking distances are tested directly in the host helper; this is not validation
of a physical robot with astronomical travel. No full historical-paper campaign, multi-GPU
or broad Python/CUDA compatibility matrix is claimed.

## Timing ready but not run

New isolated reference9f3e017 and candidate70bbb70 endpoints are precompiled for packaged
Panda, runtime Panda and 24-DoF. The later migration-note commit has identical solver/build
implementation. Previous timing endpoints and evidence are untouched.

Provenance preflight passes for all five legs. A 144-query correctness-only rehearsal has
equal solved/empty/returned counts in every matched cell and empty latency fields. The warm
leg uses one bounded midpoint seed identical on both builds, never an endpoint-generated IK
solution. This is harness verification, not performance evidence or universal solve success.

Local one-command launcher:
`docs/open-tasks/boundary-gate-2026-10-25/run.sh`.
Full coordinator instructions: `docs/open-tasks/boundary_timing_handoff_2026-10-25.md`.
Reserve 30–45 minutes for 41,600 measured queries plus warmups and the intentional cold-build
leg. Only run `--quiet-window` after a new explicit user/coordinator quiet CPU/GPU allocation.
The handoff specifies prerequisites, output paths, contention checks and stop/resume behavior.

Any quality loss or repeatable new slowdown beyond paired-round spread requires review;
the earlier accepted P2b tradeoff is not permission for additional regression. Keep raw
rounds and report absolute costs and ratios. No new performance conclusion is drawn here.

## Remaining decisions

Quiet timing and performance review, then explicit push/PR/merge authority and remote CI.
Integer coercion and reserved preset/custom-name policy remain separate compatibility decisions;
the audit's naming ambiguity is not claimed fixed. Duplicate target-discovery generation,
test-helper reorganization and structured native errors remain deferred maintenance.
GRiD G13, complete Z1 collision modeling, Fetch licensing/scope, hardware validation and
historical-paper provenance gaps are unchanged. No upstream work was performed.

`SHA256SUMS` binds the preserved logs/reports/configuration snapshots. Absolute paths in
timing snapshots refer to local precompiled artifacts, not portable binaries in this packet.
