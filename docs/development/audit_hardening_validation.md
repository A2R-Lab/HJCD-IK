# Audit-hardening pre-merge validation — 2026-09-27 UTC

Implementation checkpoint: `516b988`; signed receipt commit: `c6e134b`.
The feature branch includes `main` through `9dd1ea2` (funding acknowledgement).
This is a review record, not a new paper-results table or performance claim.

## Scope and compatibility

The branch hardens collision filtering/cache identity, CUDA error handling and
allocation ownership, native/Python contracts, synchronization, statistics, setup,
packaging and robot-specific builds. It compiles the CUDA core once for both front
ends and emits the kinematics-focused GRiD profile. The latest follow-ups reuse
parsed collision documents and explicitly distinguish historical/current Panda
collision geometry without changing either model.

Read [the upgrade guide](../source/user_guide/upgrading.md) before reviewing callers:
hard collision filtering is the Python default; zero/fewer results are legitimate;
native results are move-only owners; errors raise rather than silently falling back.
Concurrency is serialized, not asynchronous solve execution. Unsupported robot
classes are rejected rather than inherited implicitly from GRiD's wider support.

The user-facing pass corrected units and execution-model prose, supported-robot
claims, build/codegen instructions, and success-rate interpretation. It added
executable quickstart/example coverage, including empty results. The benchmark now
delegates generation to the canonical validated codegen script rather than an old
duplicate implementation. The paper workflow requires `HJCD_REGEN=1` when HJCD runs
and builds collision-enabled Panda for collision workloads/default restoration.
Its command wiring was tested with recording stubs, not performance runs.

## Dependency identities

| Component | Verified revision/version |
| --- | --- |
| GRiD (`modernizing-tests`) | `65fd051198e4ccc6f609238bca4cfa780eb2b55a` |
| Top-level and nested GLASS (`main`) | `8ce68a29bceb30c7764c9391d517a182d061697d` |
| RBDReference | `5f224e2ce2e3eab22968fbe4f83ceedc20696d40` |
| URDFParser | `29d3d78487f7db258410398b832760b48c10cb4c` |
| foam | `116928f71aaa7c40356d79c84d3c9ff1f4497d90` |
| pytest-gpu-proof | `0.4.0`, latest published PyPI release checked during this pass |

Remote availability was checked. The latest GRiD regeneration changes provenance,
unused integrator declarations and comments; default Panda FK/collision geometry
and GLASS math sources did not change in this dependency update. The new header
SHA256 is `3f6c65fd9e8e91c7ae50ee072215f45d4717edc8fe11a399ebc84e07922a802d`.

## Completed checks

Environment: Linux, RTX 5090, CUDA compiler 13.2.86, GCC 13.3, Python 3.12.
The release wheel was compiled independently without diagnostic CUDA flags.
Correctness/compilation ran on a shared machine; no performance timings were taken.

- **140 tests passed** on the editable build, independent archive-built release
  wheel, and an extracted Git-free source archive. Three expected fixed-base URDF
  inertia warnings; no skipped tests in these full runs.
- **Signed proof verified**: signature, complete manifest, source fingerprint and
  commit ancestry. Policy now also binds examples, README and `docs/source`, since
  their snippets are executed by the suite. Verification used clean checkouts.
- **Native API/CLI: 2/2 passed** for default Panda (diagnostic and release), Fetch,
  and 24-DoF builds. C++ front ends compiled with warnings-as-errors.
- **Collision + initialization-failure Memcheck: 31 passed, zero errors.**
  Native 24-DoF Memcheck also reported zero errors.
- Independent no-collision Fetch and 24-DoF wheels: four sampled targets, both
  precisions, W=1/3/4/16, 96 returned rows per model. FK, finite values, limits,
  output-error units and header identity checked. Maximum FK position differences
  were 0.35 micrometers (Fetch) and 3.59 micrometers (24 DoF).
- Pinned bootstrap succeeded in a clean clone reusing local dependency objects.
  A clean-clone source archive built an independently installed wheel; required
  source/tests/type information were included, local notes/user files excluded.
  Fresh environment `pip check` passed.
- Header freshness passed from Git and from a Git-free extraction using the nested
  GLASS revision label. Extract archives outside unrelated Git worktrees: GRiD's
  Git discovery can otherwise pick up the enclosing repository's HEAD.
- Exact CI site assembly and strict Doxygen/Sphinx build passed. Checked **790
  local links/assets across 14 rendered pages**, with no broken references.
  All research landing assets and published paper figures are unchanged. Automated
  browser preview was unavailable in this environment; visual review is not claimed.
- Whole-branch whitespace check passed. No dependency checkout edits were retained;
  the user's unrelated untracked timing driver was not modified or committed.

Only Python-suite outcomes are certified by the receipt; native, sanitizer,
packaging and docs checks above are additional evidence. Earlier audit checkpoints
also exercised 12/18-DoF models and broader sanitizers; those are not presented as
fresh executions against these latest pins.

## Reproduce the principal checks

Use a clean checkout with a CUDA GPU and the documented dependencies:

```bash
bash scripts/setup/bootstrap.sh
python -m pip install -e '.[dev,codegen]'
python scripts/codegen/check_grid_cuh_fresh.py
python -m pytest tests -q
cmake -S . -B build-native -DBUILD_PYTHON=OFF \
  -DHJCDIK_BUILD_NATIVE_TESTS=ON -DHJCDIK_WARNINGS_AS_ERRORS=ON
cmake --build build-native --parallel 2
ctest --test-dir build-native --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode=1 \
  python -m pytest tests/test_collision.py tests/test_initialization.py -q
python -m pip install -r docs/requirements.txt
bash scripts/build_site.sh
```

Follow the isolated-build instructions in the custom-robot tutorial for Fetch and
24 DoF. Build a source archive only from a clean tree: untracked nonignored files
can enter an sdist. Verify a wheel from that archive in a separate environment,
outside the checkout (or using Python's safe-path mode), to avoid shadowing the
installed extension with the source package.

Record proof after committing the reviewed test manifest, then commit its receipt
and verify again. Preserve its recorded ancestor when merging, or regenerate after
rewriting history. PR docs checks build only; deployment is restricted to `main`.

## Remaining gates

1. **Authorized quiet-window targeted runtime A/B** of the latest release wheel
   against frozen pre-cache-fix evidence: full/compact JSON, repeated/changing
   scenes, open-world control, matching quality checks and runtime dependencies.
2. Review the mega PR and verify remote CI on its actual merge candidate. No push,
   PR publication or merge is implied by this local validation record.
3. A visual preview remains advisable before publishing the docs. The landing
   content is intentionally unchanged for IROS week.

Clean/incremental build-speed measurements, a paper-protocol comparison, competitor
and MMD reruns, other architectures/toolchains and physical robot testing remain
separate work. Do not claim those results based on this validation.
