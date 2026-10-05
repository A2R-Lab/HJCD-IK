# October 4 skeptical-audit follow-up

Scope: correctness, integration, usability/documentation and precompilation. Shared GPU; **no new performance
measurements**. No push authorized. Original baseline: local main `5d606ce` (nine unpushed commits).

## Implemented

- Moved coarse warp-local constant initialization ahead of divergent loops with a publishing block barrier.
  The original minimal Racecheck case reported 304 WAR errors; numerical solves alone did not detect it.
- Consumed GRiD `333bded` (includes `d2403f3` G1–G4), regenerated the default header, and replaced HJCD's
  duplicate collision sphere sidecar/checker with GRiD's generated warp API. The removed sidecar is recoverable
  from Git. Top-level and GRiD-nested GLASS remain consistently at `8ce68a2`; GRiD has not yet pinned `9e57178`.
- Added compiled named-tool metadata and explicit frame alignment in examples, regression collection and
  collision benchmarks. Unified obstacle conversion; sphere obstacles no longer disappear from the oracles.
- Made the mandatory cuRobo-model test self-contained with the attributed frozen asset (no optional skip).
- Changed benchmark accounting to attempted queries, retaining empty outputs and target/frame metadata;
  independent FK scoring rejects incomplete/duplicate groups. Paper workers now fail closed and require a
  fresh output directory. Timing calls use keyword precision, explicit model identities and quality checks.
- Added a one-command precompiled A/B gate with a clock-free correctness mode, foreign-PID checks,
  complete-cell validation and provenance. See [timing_gate.md](timing_gate.md).
- Corrected fallback, mesh-validation and historical timing claims. Original raw evidence and research
  landing page are unchanged. Current performance remains unverified.

## Validation completed

- Full Python suite: **188 passed, zero skipped**, on CUDA 13.2 / RTX 5090 / Python 3.12. Three expected
  URDFParser base-inertial warnings. GPU-proof manifest updated; final signed receipt is the authority.
- CTest: native API + CLI contracts, **2/2 passed**.
- Racecheck: 16 collision refinement combinations (fp32/fp64, 1/3/4 LM warps, stop bits and repair settings),
  **zero errors and zero warnings**. Diagnostic Release build with `-lineinfo`.
- Synccheck: same 16 cases, **zero errors**. Memcheck: expanded 20-case matrix including blocked-scene
  single-solution repair/fallback in both precisions, **zero errors**.
- Default generated-header freshness check passed; Sphinx `-W --keep-going` build passed.
- Isolated Fetch and 24-DoF no-collision Release builds: native API/CLI **2/2 passed each**.
- Actual optional FCL visual-mesh sphere-obstacle construction/check passed in the baseline venv.
- Release hand-frame endpoints are precompiled at `1d2cba2` (pre-refinement) and `2302d22` (final native
  code; subsequent edits are harness/docs/proof). Python 3.12.3, NumPy 2.5.3, nvcc 13.2.86, Release `-O3`,
  no additional CUDA flags. A clock-free three-endpoint S=1/4 smoke (B=8, 100 targets per leg) completed
  1,200 calls; accuracy counts were identical in every paired cell. This is not a statistical speed gate.

## Fresh dataset correctness (no clocks)

All ten standard/extra dataset sets were recollected at B=100/2000, S=1, default precision and hard
collision mode, with independent URDF FK and environment-sphere validation. **2,000/2,000 query records**
were retained, including the protocol for empty outputs (none occurred in this run). Per-group pose-success
counts exactly match the archived default-model `hjcdik_ccstop` run: 995/1000 at B=100 and **999/1000 at
B=2000**. At B=2000 all eight original sets and table_bars reached 100/100; kitchen remained 99/100.
Independent visual-mesh scoring at 1 mm also gives 999/1000 at B=2000; hull and other sphere judges can
disagree. These are model- and tolerance-specific results, not a universal collision or accuracy guarantee.

The exact historical clearance ladder was recollected with its original >=20-query set selection:
**2,776/2,776 records**, 2,599 pose-accurate versus 2,597 in the archived default-model dump. Two groups
gain one accurate query each; no group loses one. Candidate nondeterminism means this small gain is not
evidence of an algorithmic improvement. The old workload's pre-cache-fix lineage is preserved deliberately
for the matched comparison, not presented as a freshly generated ladder. Conservative b10/b15 historical
groups remain marked incomplete and must be recollected before reusing their rates in a new publication.

Full local records, oracle tables, build hashes and launch config are under
`docs/open-tasks/audit_gate_2026-10-04/` (ignored staging, not shipped in the package). The timing handoff
contains exact paths. The research landing page and paper headline tables have not been updated.

These are targeted sanitizer checks, not exhaustive formal race or collision safety proofs. A successful
suite does not establish speed, full collision-model fidelity, or unchanged statistical success on every
benchmark scene. Quiet-window A/B and the final publication campaign remain separate gates.
