#!/usr/bin/env bash
# Full paper protocol (HJCD + PyRoki + cuRobo + IKFlow + TRAC-IK ground truth) in the staged clone,
# with a GPU/CPU activity monitor. Usage:
#   run_paper_full.sh            # full campaign (quiet window)
#   DRY=1 run_paper_full.sh      # tiny functional dry run (3 targets, batches 1,10) - not for timing
set -uo pipefail
STAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$STAGE/hjcd"
OUT="$STAGE/${OUT_NAME:-paper_results}"
mkdir -p "$OUT"
( while true; do
    echo "$(date +%T) | $(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader | tr '\n' ';') | $(top -bn1 | sed -n 3p | cut -c1-40)"
    sleep 3
  done > "$OUT/monitor.log" 2>&1 ) &
MON=$!
START=$(date +%s)
if [ "${DRY:-0}" = "1" ]; then export NUM_TARGETS=3 BATCHES="1,10" DOF_BATCH=10; fi
HJCD_REGEN=1 RUN_FETCH=1 RUN_DOF=1 RUN_MMD=1 RUN_HARD=1 OUT_DIR="$OUT" \
  bash scripts/bench/run_paper_experiments.sh > "$OUT/run.log" 2>&1
RC=$?
kill "$MON"
echo "paper rc=$RC elapsed=$(( $(date +%s) - START ))s" | tee -a "$OUT/run.log"
git status --short | tee -a "$OUT/run.log"    # should be empty: the script restores the default build
exit $RC
