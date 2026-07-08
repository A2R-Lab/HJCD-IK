#!/usr/bin/env bash
# One-stop HJCD-IK TIMING sweep runner — RUN ON A QUIET GPU.
#
# Stages every HJCD-only timing sweep behind a single command so a full perf capture is easy to fire later
# (timing must be isolated: one bench at a time, quiet GPU — correctness testing is separate, see
# scripts/setup/run_gpu_proof.sh). Competitor baselines (PyRoki/cuRobo/IKFlow) are SEPARATE and heavy — for
# the full cross-solver paper tables run scripts/setup/install_baselines.sh + scripts/bench/run_paper_experiments.sh.
#
#   scripts/perf/run_all_timing_sweeps.sh                        # all sweeps -> benchmark/results/timing_<ts>/
#   SWEEPS=openworld,collfree scripts/perf/run_all_timing_sweeps.sh   # subset
#   BATCHES="1,32,256,2048" NUM_TARGETS=250 scripts/perf/run_all_timing_sweeps.sh
#
# Sweeps (comma-list in SWEEPS; default = all, in this order):
#   openworld  — Table I open-world Panda latency vs batch (also the post-GRiD-bump regression reference)
#   collfree   — Table II collision-free Panda (bookshelf_thin_panda)
#   multiwarp  — W=1,2,4,8 x fp64/fp32 sweep (scripts/perf/time_multiwarp_sweep.py)
#   dof        — Table III DoF 7/12/18/24 fp32-vs-fp64 A/B (regen+rebuild per DoF; restores Panda) — HEAVIEST, runs last
#
# Prereq: a built `hjcdik` on the default panda_grasptarget_hand build (scripts/setup/rebuild.sh).
# Outputs (gitignored benchmark/results/) are run-reference only — per repo policy, never commit/publish these;
# published numbers must match the camera-ready paper.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PYTHON:-.venv/bin/python}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT="benchmark/results/timing_${TS}"
mkdir -p "$OUT"
BATCHES="${BATCHES:-1,10,100,1000,2000}"
NUM_TARGETS="${NUM_TARGETS:-100}"
SWEEPS="${SWEEPS:-openworld,collfree,multiwarp,dof}"
MB_JSON="tests/mb_problems.json"
PROBLEM_SET="${PROBLEM_SET:-bookshelf_thin_panda}"
SETTLE_SECS="${SETTLE_SECS:-6}"

has() { echo ",${SWEEPS}," | grep -q ",$1,"; }
banner() { echo; echo "==================== $* ===================="; }
# Let the GPU drain between sweeps so the next one's startup quiet-check doesn't sample the
# previous sweep's teardown (utilization.gpu is time-averaged). No sleep before the FIRST sweep.
_first_stage=1
settle() {
  if [ "$_first_stage" = "1" ]; then _first_stage=0; return; fi
  echo "[timing] settling ${SETTLE_SECS}s for the GPU to go idle before the next sweep..."
  sleep "$SETTLE_SECS"
}

# --- preconditions -------------------------------------------------------------------------------------
if ! "$PY" -c "import hjcdik" 2>/dev/null; then
  echo "ERROR: hjcdik not importable — build first: scripts/setup/rebuild.sh" >&2; exit 1
fi
UTIL="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)"
if [ "${UTIL:-0}" -gt 5 ]; then
  echo "WARNING: GPU utilization ${UTIL}% — timing wants a QUIET GPU; numbers may be noisy. Continue anyway."
fi
echo "[timing] results -> $OUT   sweeps: $SWEEPS   batches: $BATCHES   targets: $NUM_TARGETS"

# --- openworld (Table I) : latency vs batch, default build (post-bump regression reference) ------------
if has openworld; then
  settle
  banner "Table I — open-world Panda (batch sweep)"
  "$PY" benchmark/hjcd_ik_bench.py --skip-grid-codegen --num-targets "$NUM_TARGETS" \
      --batches "$BATCHES" --num-solutions 1 --solver hjcdik \
      --csv-out "$OUT/openworld_hjcdik.csv" 2>&1 | tee "$OUT/openworld.log" \
    || echo "[timing] openworld FAILED (continuing)"
fi

# --- collfree (Table II) : collision-free on a MotionBenchMaker scene -----------------------------------
if has collfree; then
  settle
  banner "Table II — collision-free Panda ($PROBLEM_SET)"
  "$PY" benchmark/hjcd_ik_bench.py --skip-grid-codegen --collision-free \
      --problems-json "$MB_JSON" --problem-set "$PROBLEM_SET" \
      --batches "$BATCHES" --num-solutions 1 --solver hjcdik \
      --csv-out "$OUT/collfree_hjcdik.csv" 2>&1 | tee "$OUT/collfree.log" \
    || echo "[timing] collfree FAILED (continuing)"
fi

# --- multiwarp : W=1,2,4,8 x fp64/fp32 ------------------------------------------------------------------
if has multiwarp; then
  settle
  banner "Multiwarp W sweep (fp64/fp32)"
  "$PY" scripts/perf/time_multiwarp_sweep.py --warps 1,2,4,8 --batches 256,2000,16384 \
      --precisions fp64,fp32 2>&1 | tee "$OUT/multiwarp.log" \
    || echo "[timing] multiwarp FAILED (continuing)"
fi

# --- dof (Table III) : fp32-vs-fp64 A/B per DoF (regen+rebuild each; restores Panda). HEAVIEST -> last --
if has dof; then
  settle
  banner "Table III — DoF 7/12/18/24 fp32-vs-fp64 A/B (regen+rebuild per DoF)"
  echo "[timing] this regenerates + rebuilds the extension per DoF and restores the Panda build on exit."
  scripts/perf/dof_scaling_ab.sh 2>&1 | tee "$OUT/dof_ab.log" \
    || echo "[timing] dof A/B FAILED (continuing; check the default Panda build was restored)"
fi

banner "DONE"
echo "[timing] all requested sweeps complete -> $OUT"
ls -1 "$OUT"
