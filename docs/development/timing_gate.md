# Current correctness and timing protocol

The October 4 audit separates correctness (shared GPU allowed) from performance (announced quiet window).
No new latency or paper-speedup claim is justified until the final compiled endpoints pass the latter.

## Evidence corrections

- The October 2 neutral driver passed `False` as positional argument eight, which is `refine_fp64`, not
  `write_stats`. Its S=1 measurements are **forced fp32**, not the default fp64. Raw data are preserved.
- October 3 conservative sphere-model ladder dumps contain 2775 (b10) and 2771 (b15) records instead of
  2776. Empty outputs were omitted. Those groups require recollection; do not interpret their old
  returned-record percentages as per-query success. The default-model groups were complete.
- Visual meshes are an independent environment-only approximation. They exclude the base and self
  collision and shrink obstacles by the reported tolerance. They are not physical ground truth.
- Shared-GPU latency estimates (including the old conservative-model “3–4×” observation) are not valid
  performance evidence. The research landing page and original paper tables remain historical.

## Query contract

`benchmark/query_results.py` owns query accounting and independent Panda FK. Configuration JSONL records
contain the actual target, compiled frame, problem index, batch, returned count, selected candidate, and
an explicit empty record for a zero-output attempt. Both errors must pass for the same candidate.
`score_collision_oracles.py` rejects duplicate/incomplete groups and missing requested judges by default.
`--allow-legacy-reports` and `--allow-missing-oracles` are explicit exploratory opt-outs, not publication gates.
For a subset experiment supply a correspondingly indexed subset problem manifest; do not score a partial
file as though it were the entire dataset. Different solvers/batches must cover the same selected sets.

CSV summaries count attempts (including empty failures), not returned solutions. Empty errors are infinite;
`queries` and `pose_success(%)` make the denominator and joint accuracy explicit. Collision-only percentages
are not interchangeable with pose-and-collision success. Final publication scoring uses independent FK.

## Single-launch A/B gate

Launcher: `scripts/perf/run_audit_gate.py`. Run from the repository root using `.venv/bin/python`.
The local handoff specifies the prepared config's absolute path and exact precompiled endpoint identities.

```bash
.venv/bin/python scripts/perf/run_audit_gate.py --config /absolute/path/gate.json \
  --out /absolute/path/new-results --quiet-window
```

Prerequisites: `nvidia-smi`, a quiet GPU, two installed **Release** builds (no `-G`; use a separate diagnostic
build for sanitizers), identical Panda hand-frame targets and sphere geometry, initialized pinned dependencies.
`gate.json` stores endpoint interpreter/repository/commit/header SHA/binary SHA and optional explicit HJCD
environment settings; shared workload paths/hashes; target counts; rounds, solution counts and batch lists.
Endpoint `main` is the reference. Add `branch` and optionally `branch-nostop` (`HJCD_CC_STOP=0`). Both endpoint
headers must be generated for `panda_hand_joint`; an older build without named-frame metadata requires its
header to be inspected and hashed during staging. No builds occur in the timing launcher.

Use `--correctness-only` instead of `--quiet-window` to exercise the same solves without reading clocks or
writing fabricated timing values. A small config is appropriate for a shared-GPU smoke check. Performance
mode checks foreign GPU PIDs before and during workers, clears inherited HJCD settings, alternates endpoints,
and fails on worker errors. It records exact build/binary/input provenance in `manifest.json`; stdout/stderr
in individual logs; query counts and quality with timings in `results.csv`; paired ratios in `analysis.json`.
Incomplete runs cannot produce a complete analysis. A quality drop is flagged for review, not automatically
excused as nondeterminism. Median ratios alone do not establish a performance win; inspect per-round spread,
accuracy, output counts, and repeated runs. Paper comparisons require matching precision/frame/geometry and
the paper's mean metric, not these A/B medians.

Estimated full gate: roughly 20–35 minutes for six rounds, S=1/4 and the prepared batch sweep; hardware and
solver convergence affect this. No clean timing estimate is made from shared-GPU work. Stop with Ctrl-C;
the launcher terminates only its own worker PID. Partial output is preserved but invalid. Resume by launching
into a **new output directory**, not appending to the partial run. Never use `pkill` on shared machines.

After reviewing the A/B gate, run the paper/baseline campaign in an isolated staging checkout with fresh
output directories. Do not run codegen-mutating paper scripts in the working checkout during other work.
Refresh result tables only from complete, matched-protocol records; re-sign the GPU proof after fingerprinted
source/docs changes. Push remains a separate user-approved action.
