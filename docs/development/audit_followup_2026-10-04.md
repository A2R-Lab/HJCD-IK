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

- Full Python suite: **187 passed, zero skipped**, on CUDA 13.2 / RTX 5090 / Python 3.12. Three expected
  URDFParser base-inertial warnings. GPU-proof manifest updated; final signed receipt is the authority.
- CTest: native API + CLI contracts, **2/2 passed**.
- Racecheck: 16 collision refinement combinations (fp32/fp64, 1/3/4 LM warps, stop bits and repair settings),
  **zero errors and zero warnings**. Diagnostic Release build with `-lineinfo`.
- Synccheck: same 16 cases, **zero errors**. Memcheck: expanded 20-case matrix including blocked-scene
  single-solution repair/fallback in both precisions, **zero errors**.
- Default generated-header freshness check passed; Sphinx `-W --keep-going` build passed.

These are targeted sanitizer checks, not exhaustive formal race or collision safety proofs. A successful
suite does not establish speed, full collision-model fidelity, or unchanged statistical success on every
benchmark scene. Quiet-window A/B and the final publication campaign remain separate gates.
