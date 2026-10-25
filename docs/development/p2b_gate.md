# General-joint timing gate

Correctness may run on the shared GPU. Measurements require a user-announced quiet GPU
and CPU window. Nothing in the staging commands below performs a performance measurement.
Do not replace the reference with an older API endpoint or interpret shared-box build
durations as compile-speed evidence.

## Prepare

Use a new stage directory. The staging helper never removes or overwrites a checkout.
It pins Python package versions to the development environment, clones local submodules,
generates a hand-frame Panda pair, builds its packaged/runtime and 24-DoF artifacts, and
checks solve quality before writing endpoint manifests. Each endpoint has its own cache.

```bash
.venv/bin/python scripts/perf/stage_joint_gate.py --commit 3f5ec5a --label main --directory /absolute/stage/main
.venv/bin/python scripts/perf/stage_joint_gate.py --commit <final-candidate> --label branch --directory /absolute/stage/branch
.venv/bin/python scripts/perf/assemble_joint_gate.py --stage /absolute/stage --targets /absolute/shared-panda-targets.json
.venv/bin/python scripts/perf/run_joint_gate.py --stage /absolute/stage --out /absolute/unused-output --check
```

The Panda targets must be the shared 100-target hand-frame workload used by the existing
audit gate. The assembler creates 100 independent-FK high-DoF targets with seed 20261025.
It freezes input hashes, endpoint commits, installed binary hashes and header hashes.
Prepared artifacts must remain unchanged. The runtime and native comparisons each use
matched reference/candidate entry points; API overhead is not mistaken for kernel overhead.

## Single launch

From the HJCD-IK repository root, in the announced window:

```bash
.venv/bin/python scripts/perf/run_joint_gate.py --stage /absolute/stage --out /absolute/new-results --quiet-window
```

Prerequisites are the precompiled endpoints, their venvs and local caches, `nvidia-smi`,
the same compiler for both endpoints, and no other compute work. Allow roughly 30–45 minutes
for four comparison legs plus two alternating rounds of cold-build/cache-hit measurements.
This is a scheduling estimate, not a shared-box performance result. All solve artifacts
are precompiled. Only the final compile-cost leg intentionally compiles fresh artifacts.

The four solve legs are API, native entry, prepared runtime Panda and 24-DoF. Each uses
four alternating rounds, S=1/4 with auto precision, 100 targets per cell. Panda open-world
batches are 1/2000/8000/32000; collision `box_panda` batches are 100/2000. The high-DoF leg
is open-world only. Every query, including empty results, is counted; errors are checked
against independent FK. Native results are unpacked after the clock. Results appear in
`api/`, `native/`, `runtime/`, `highdof/`, each with manifest, CSV, analysis and worker logs.

The final `compile/` leg measures cold runtime builds and repeated-build cache hits for
Panda and 24-DoF, with fresh caches per endpoint/round. It includes code generation, compile
and publication, not `pip install` or packaged-wheel installation. Build logs and artifacts
remain in those caches. The two rounds are descriptive; obtain more samples before making
a small compile-speed claim. Correctness-only mode skips this timed leg entirely.

## Review and interruptions

Incomplete groups, mixed provenance, foreign GPU work, invalid quality accounting or an
independent-FK disagreement invalidate the run. As in the existing protocol, a drop in
solved count is flagged for review, never automatically excused as nondeterminism. Inspect
returned/empty counts and per-round spread. Any repeatable latency increase beyond that
spread needs investigation before merge; a median near one is not alone a pass. Borderline
or noisy findings require another complete quiet run, without changing criteria after
seeing results. Neither the historical 0.6 percent range nor the new-robot correctness
results define an acceptance threshold. No automatic performance approval is emitted.

Stop with Ctrl-C in the launching terminal. Workers are terminated by their owned PIDs,
including compiler descendants; no name-based process killing is used. Partial output is
preserved and is not publishable. Resume into a new output directory, never append to the
partial run. A failed worker stops the remaining legs. `--check` does not create outputs;
`--correctness-only` runs the solve workloads without clocks and skips compile timing.

Push and merge remain separate user-approved actions. The research landing page and paper
tables are unchanged; this gate is a matched same-machine regression check, not a paper rerun.
