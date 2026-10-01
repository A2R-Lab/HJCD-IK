# Audit-hardening pre-merge validation — 2026-09-27 UTC

Implementation checkpoint: `516b988`; signed receipt commit: `c6e134b`.
The feature branch includes `main` through `9dd1ea2` (funding acknowledgement).
This is a review record with bounded before/after measurements, not a new
paper-results table or a cross-hardware performance claim.

## GRiD main integration — 2026-10-01

The final integration follows GRiD's merged `main`, superseding the September 27
dependency identities below. `.gitmodules` now tracks `main` rather than
`modernizing-tests`; bootstrap still checks out the exact committed pin.

- GRiD: `0a14c0f1d6957687d606cc9d3dde6f6bce004233`.
- Nested URDFParser: `07dee119204be9578bc64f54475bca93f7e13988`.
- Nested RBDReference: `595e4a3cfafae74d5b248e08d13dc667d625c6b9`.
- Top/nested GLASS and foam pins are unchanged.

Panda collision-enabled, Fetch and 24-DoF no-collision headers regenerated under
these pins are **byte-identical** to the previously validated headers. GRiD's
collision helper sources and HJCD's compiled solver/GLASS inputs are unchanged.
The newer upstream dynamics fixes concern floating/mimic models outside HJCD's
supported fixed-base model. The parser now raises informative errors for malformed
input rather than returning `None`; valid bundled models retain their output.

The full suite passed again: **140 tests**, three expected fixed-base inertia
warnings; native API/CLI **2/2 passed**, using the existing validated release
binaries. No recompilation or additional performance measurement is necessary for
identical compiled inputs. A fresh signed 140-test receipt now binds the dependency
update at `175479f`; the September receipt is superseded for this new gitlink.

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
Correctness/compilation ran on a shared machine. Separate authorized quiet-window
runtime campaigns completed afterward; see the measured scope below.

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
- Focused Racecheck covered partial blocks, both precisions, early-stop and hard/both
  collision paths: **zero errors and zero warnings**. Synccheck on both partial-block
  precision cases: **2 passed, zero errors**.
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
  All research landing assets and published paper figures are unchanged. Isolated
  headless Chrome screenshots of the docs homepage (desktop) and upgrade guide
  (desktop/mobile) were subsequently inspected. This is a sampled visual review,
  not exhaustive browser or accessibility coverage.
- Whole-branch whitespace check passed with `core.whitespace=cr-at-eol` to respect
  the existing CRLF source files. No dependency checkout edits were retained;
  the user's unrelated untracked timing driver was not modified or committed.

Only Python-suite outcomes are certified by the receipt; native, sanitizer,
packaging and docs checks above are additional evidence. Earlier audit checkpoints
also exercised 12/18-DoF models and broader sanitizers; those are not presented as
fresh executions against these latest pins.

## Completed runtime gate

The first matched-release campaign compared pre-audit `4ccb09b`, audited `2b746a1`
and checked-GRiD `209be95` on the same RTX 5090. Its 22,464 measured calls all found
an independently checked solution. Across 16 open-world cases (Panda/24 DoF,
batches 128/2000, one/four outputs, both precisions), median paired-round
`209be95`/pre-audit changes ranged from -1.54% to +2.03%, essentially neutral.
It exposed a repeatable full-document collision-input penalty, motivating the
document-cache follow-up. These are historical endpoint results, not fresh
24-DoF timings of the final pins.

The follow-up compared frozen `209be95` against implementation `516b988`, unchanged
through `cc71978`, on 2026-09-27 04:30:21–04:33:08 UTC. All 60 workers completed:
**7,680/7,680 timed calls solved**, 28,416 returned configurations, no independent
environment-collision or joint-limit failures, and no observed foreign GPU compute
processes. The same Python 3.12.3 / NumPy 2.5.1 runtime served both release builds.

| Workload | Baseline → latest median ms | Paired-round median changes |
| --- | --- | --- |
| Open, one solution, fp64 | 1.2869 → 1.2857 | -0.28% … +0.89% |
| Open, four solutions, fp32 | 1.4649 → 1.4684 | -0.08% … +0.52% |
| Hard, full JSON, repeated scene | 2.8542 → 2.4419 | -14.93% … -13.71% |
| Hard, full JSON, changing scene | 34.3388 → 2.4531 | -92.92% … -92.82% |
| Hard, compact JSON, repeated scene | 2.1395 → 2.1106 | -1.51% … -1.09% |
| Hard, compact JSON, changing scene | 2.9849 → 2.1170 | -29.29% … -28.96% |

The `both` policy showed the same pattern; full results, p95 values, exact harness,
inputs/provenance and interpretation are in the
[frozen timing evidence](evidence/targeted_timing_2026-09-27/README.md).
Positive changes mean slower. Aggregated medians and paired-round ratios are
different statistics; do not derive one from the other.

This closes the bounded cache/latest-pin runtime gate: open controls remained
within 1% and collision-input handling improved. The roughly 14x full-document
scene-switch improvement is **host document caching, not a 14x faster GPU IK
kernel**. Only 32 frozen targets/scenes repeated at batch 2000 and W=1 were used;
collision cases requested four solutions with fp32 refinement. The follow-up
oracle checks environment collisions, not independent self-collision correctness.
Cold initialization was excluded from warmed timings and recorded separately.
Sampling contention guards cannot exclude interference between observations.

## Final host-only review

On 2026-09-27, a fresh fetch still placed `origin/main` at integrated `9dd1ea2`.
Review of the aggregate branch changes covered shared-state synchronization and
ownership, collision cache failure/invalidation paths, Python/native/CLI contracts,
supported codegen models, setup/build duplication, packaging and CI/deployment
scope. No new merge-blocking defect was identified in that review; it is not a
claim of exhaustive verification or an independent second reviewer's approval.

With the GPU hidden, **33 host regression tests passed** (codegen, benchmark setup,
geometry identity, proof policy, statistics summary and native document cache),
with three expected fixed-base inertia warnings. The frozen campaign's **14
synthetic harness tests passed**. Saved raw outputs independently reproduced all
ten paired comparisons, medians/p95, quality totals and the archived checksums.
No timing, CUDA compilation or GPU execution was performed for this closeout.

Only developer evidence changed after `cc71978`; implementation, tested user
docs/examples and dependency pins remain identical to the signed and measured
checkpoint. Preserve the user-owned untracked timing driver; use a clean checkout
for receipt verification and source packaging.

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

1. Review the mega PR and verify remote CI on its actual merge candidate. No push,
   PR publication or merge is implied by this local validation record.
2. Preserve the receipt's recorded ancestor (or regenerate after a history rewrite),
   then check main CI, installation and docs deployment after an authorized merge.
   The landing content is intentionally unchanged for IROS week.

Clean/incremental build-speed measurements, a paper-protocol comparison, competitor
and MMD reruns, other architectures/toolchains and physical robot testing remain
separate work. Do not claim those results based on this validation.
