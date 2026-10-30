# Usability closeout and review packet

Superseded current review status: [boundary audit](../boundary_audit_2026-10-25/README.md).
The packaged-spheres decision is resolved by `b1ef1dd` with 339 verified tests. Four
additional correctness findings remain open; do not interpret this historical packet
as merge approval. Its native validation and timing caveats remain preserved below.

Closeout implementation: `e8cc1c3..06def18`. Full unpublished-wave comparison base:
locally recorded `origin/main`, `119cacc` (not freshly fetched). Publication is not
authorized. The packet is a local review checkpoint, not a remote CI or merge approval.

## What changed

- Illustrative X-rail + UR5e example, frozen CPU-derived feasible witnesses, independent
  FK/Jacobian/geometry checks and a complete cold/warm/world-replacement workflow.
- Repeatable source-archive -> wheel -> isolated installation check, including packaged
  Panda without runtime-build extras and fresh-process prepared loading without compilation.
- Six focused build/load failure tests. Extension initialization errors raised from
  `module_from_spec` now become the intended public `BuildError`, with recovery guidance.
- User and agent documentation reconciled with the actual API, mixed units, collision
  ownership, measured tradeoffs and historical-paper provenance.

No dependency re-pin, default-model substitution, solver mathematics, collision policy,
precision selection, tolerance, compile flags or benchmark-setting change in this closeout.
The whole unpublished wave is larger: it includes GRiD 0.6.0 integration, runtime builds,
warm starts/seeds/margins, native host-loop batching, module isolation, general joint axes,
mixed prismatic chains, joint metadata and iiwa14/UR5e presets. Do not describe the whole
wave as documentation-only. See [migration guidance](../../../source/user_guide/upgrading.md).

## Validation

The final source `06def18` passed all 333 signed cases (previously 317), with no skips
or weakened thresholds; verification passed against the repository policy. The 94 parser
warnings are retained, including nominal URDF fixed-helper inertials; these are kinematics
models, not validated dynamics assets. Shared-box test elapsed times are not benchmarks.

| Check | Result / evidence |
|---|---|
| Final full signed GPU suite | 333 passed, receipt verified; `gpu-proof-suite.txt`, `gpu-proof-verify.txt` |
| Native CTest | 2/2 passed; `native-tests.txt` |
| Generated header pair freshness | Passed; `header-freshness.txt` |
| Fresh Doxygen + strict Sphinx | Passed; `sphinx.txt`; regenerated XML, not stale cached XML |
| Final installed workflow | Passed on clean `06def18`; `installed-report.json`, `installed-commands.json` |
| Rail memcheck / synccheck | 2 cases each, exit 0, zero errors; corresponding logs |
| Rail combined racecheck | 2 cases, exit 0, zero errors/warnings/hazards; `racecheck-final.txt` |

Rail acceptance was declared before outcomes: 32 targets per scene, B=2000/S=4,
fp32 and fp64, at least 31/32 same-candidate accurate targets per cold/warm cell at
5 mm / 0.001 rad. Measured 32/32 cold and 32/32 neighboring warm in all four cells;
blocked worlds returned zero and restored worlds recovered. The independent repository
oracle checks every returned hard-mode configuration for bounds, finite values and
sphere-model validity, not only the most accurate row. `rail-correctness.xml` records
the focused outcomes. Installed build and fresh-load phases each repeated all four cells.

Geometry checks cover 200 fixed-seed configurations across travel, the exact rail Jacobian
column, finite differences, independent collision verdicts and the formerly fixed arm base.
The assembly retains every supplied UR5e sphere; carriage, stationary supports and physical
hardware are not modeled. No new contact exclusions or production preset were invented.
The frozen witnesses were selected by independent FK and 3 mm model clearance, not by IK
success. These are known-feasible regression sets, not an unfiltered workspace-success claim.

The source archive was built into a wheel and installed outside the checkout with `-I`
and no PYTHONPATH/PYTHONHOME leakage. Panda solved 4/4 targets with neither grid-rbd nor
pybind11 installed. Runtime stages used grid-rbd 0.6.0, pybind11 3.0.4 and NumPy 2.5.1;
loading reused the same content key while build entry points were instrumented to fail.
Python 3.12.3 / CUDA 13.2 were exercised; this is not a Python/CUDA compatibility matrix.
The report includes archive, wheel, fixture and license hashes. Larger artifacts remain
at `/tmp/hjcd-installed-final-20261025`, not committed or published.

### Sanitizer history, including the abnormal run

The first combined fp32/fp64 racecheck reported two passing tests and zero hazards but
exited 11 after testing. It is preserved as `racecheck-initial-abnormal.txt` and is NOT
counted as a clean pass. Minimal load/FK and each precision separately subsequently
exited 0 with zero hazards (`racecheck-load.txt`, `racecheck-fp32.txt`, `racecheck-fp64.txt`).
The cause of the initial abnormal exit is not established. No teardown suppression,
weakened assertions or solver changes were used to obtain the subsequent results.
The final combined repeat on the precompiled `06def18` artifact passed both precisions,
reported zero hazards and exited normally (0). The initial exit is not reproducible in
this repeat; it remains a disclosed unexplained observation, not a proven fixed defect.

## Review findings and approval decisions

One constructor inconsistency remains for explicit user disposition: `Solver("panda",
spheres="/definitely/missing/requested-model.urdf")` succeeds and silently selects the
packaged geometry (`info.robot == "panda"`, `collision_enabled == True`, `artifact is None`).
The same applies with robot omitted. Named runtime builds reject such overrides.
Recommend rejecting non-None `spheres` for packaged robots with actionable URDF-build
guidance and regression cases for both constructor forms. This is a safety-relevant
pre-merge decision: do not claim the requested geometry was honored or wave it away as
cosmetic. The behavior-preservation plan requires approval before turning an accepted
call into an error; no change to this behavior was made in this checkpoint.

Review covered public/native contract alignment, build identity/module isolation,
world/result ownership, test-oracle reuse, packaged assets/licenses, source preservation,
and the full unpublished diff. No extra cache, collision sidecar or generic assembly
framework was added. Full-wave `git diff --check` reports existing generated URDF
whitespace and raw CSV CRLF endings; source/docs checks excluding these assets pass.
The closeout source diff passes; saved raw Sphinx output also contains trailing spaces.
Evidence bytes were not reformatted merely to silence the whitespace check.

## Timing applicability

No timing was performed during this shared-box closeout. Existing
[P2b evidence](../general_joint_timing_2026-10-25/README.md) has 70,400 queries across two
complete quiet-window passes, with matched solved/empty/returned counts. The accepted
Panda B=1/S=1 cost is +0.412% API (+8.35 us), +0.609% runtime API (+12.39 us);
24-DoF pooled changes range -0.096% to +0.121%. Matched quality does not mean every
query succeeded: the B=1 cells include 95/100 or 96/100 accurate targets.

Compared with `e8cc1c3`, comment-stripped native source and line counts are identical:

- `hjcd_kernel.cu`: SHA256 `6050d32568e6e9b685bbae80a18ed084eb23ba2cf7a8b0c35822bf57f9b07cf5`
- `hjcd_kernel.h`: SHA256 `25189158a0e1039d3702103393f16bb8a802ecbf34a2bedcc4c692aa2f52d428`
- `_solver.py`: identical AST after excluding module documentation.

The only package execution change is cold-path build/import failure handling. Existing
measurements apply to unchanged solver implementation, not a claim that the newly built
binary was itself timed. No new timing is required for these closeout edits. Optional
final-binary confirmation or rail-performance characterization needs a newly announced
quiet CPU/GPU window and a separate stage; old endpoints/evidence remain untouched.

## Paper preservation and limits

Paper tables, figures, reference geometry and benchmark scripts are unchanged. One
documented collision command now explicitly selects the cylinder target mode so it
matches the default compiled tool frame. No historical build or paper campaign was run.

The `iros-2026` tag resolves to `637f4476715ea37c5f89464e2605d4754ea99a21`.
Its GRiD gitlink is `0a6c18e`; the nested CodeGenerator gitlink is `272f66c`, but its
bootstrap overrides that generator to `78312a6`. The historical environment and per-table
collection revision are not fully locked. Thus a verified exact reproduction claim is
blocked by missing provenance, not established by retaining scripts. The
[reproduction guide](../../../source/user_guide/benchmarks/reproduction.md) records the
full hashes, separate-checkout procedure, unresolved choice, current-code differences
and actual checks (CLI help/script syntax and source inspection, not a fresh paper rerun).

## Deferred items and publication sequence

- GRiD G13: all-prismatic codegen; keep early rejection until upstream support and HJCD tests.
- Z1: validate a complete useful collision model; do not omit overlapping wrist links.
- Fetch: distribution-license decision and explicit torso/whole-robot scope before assets.
- Physical rail hardware validation, broader platform matrix, new knobs, batching/DLPack and
  wheel publication infrastructure remain separate work.
- Optional performance attribution is not a blocker after the user's accepted tradeoff.

Resolve the constructor decision and any remaining abnormal validation outcome first.
Use a local review branch for the eventual PR; do not push the ahead local main directly.
Push (SSH), external PR creation, remote CI and merge each remain outside this checkpoint's
authorization. Refresh the remote comparison before publication rather than assuming
the locally recorded origin has not advanced.

## Commands

From the HJCD-IK checkout with its development venv and initialized submodules:

```bash
scripts/setup/rebuild.sh
.venv/bin/python scripts/setup/check_installed.py --out /tmp/hjcd-installed-NEW
scripts/setup/run_gpu_proof.sh
.venv/bin/gpu-proof verify --receipt gpu-proof.json --repo . --policy tests/gpu-proof-policy.yaml
cmake --build build-native -j2
ctest --test-dir build-native --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode=1 .venv/bin/python -m pytest tests/test_rail_example.py -k partial_warp -q
compute-sanitizer --tool synccheck --error-exitcode=1 .venv/bin/python -m pytest tests/test_rail_example.py -k partial_warp -q
PYTHONFAULTHANDLER=1 compute-sanitizer --tool racecheck --error-exitcode=1 .venv/bin/python -m pytest tests/test_rail_example.py -k partial_warp -q
```

Precompile the focused tests without sanitizer first. For docs, run Doxygen from `docs/`
with an explicit fresh output directory, then use `sphinx-build -W --keep-going` and
`-D breathe_projects.hjcdik=<fresh-doxygen>/xml` to avoid validating stale API comments.
Receipt order remains source commit on clean tree -> full run -> verify -> receipt commit.
Evidence-only commits under this directory do not change its source fingerprint.
