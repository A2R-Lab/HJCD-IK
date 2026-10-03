#!/usr/bin/env bash
# HJCD-only paper protocol in the staged branch clone, with a GPU/CPU activity monitor.
set -uo pipefail
STAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$STAGE/branch"
mkdir -p "$STAGE/paper_results"
( while true; do
    echo "$(date +%T) | $(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader | tr '\n' ';') | $(top -bn1 | sed -n 3p | cut -c1-40)"
    sleep 3
  done > "$STAGE/paper_monitor.log" 2>&1 ) &
MON=$!
START=$(date +%s)
HJCD_REGEN=1 RUN_FETCH=1 RUN_DOF=1 RUN_MMD=1 SKIP_PYROKI=1 SKIP_CUROBO=1 SKIP_IKFLOW=1 \
  OUT_DIR="$STAGE/paper_results" bash scripts/bench/run_paper_experiments.sh > "$STAGE/paper_run.log" 2>&1
RC=$?
kill "$MON"
echo "paper rc=$RC elapsed=$(( $(date +%s) - START ))s" | tee -a "$STAGE/paper_run.log"
exit $RC
