#!/usr/bin/env bash
# Poll for a SUSTAINED quiet-GPU window, then run the multi-warp timing sweep.
# Safe to run unattended overnight: the sweep itself re-guards and aborts on mid-run
# contention (no bogus numbers). Writes results to docs/open-tasks/multiwarp_timing_<ts>.md
# and a marker so the morning review is one glance. Other agents share the GPU, so a
# quiet window may never come — in that case it reports that, having produced nothing false.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY=.venv/bin/python
LOG=docs/open-tasks/multiwarp_timing_result.md
DEADLINE=$(( $(date +%s) + 8*3600 ))   # give up after 8h
NEED_QUIET=3                            # consecutive quiet samples required
INTERVAL=90                            # seconds between samples
MAX_UTIL=3

quiet_sample () {
  local util apps
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
  apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -c . || true)
  [ "${util:-100}" -le "$MAX_UTIL" ] && [ "${apps:-9}" -le 0 ]
}

streak=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  if quiet_sample; then
    streak=$((streak+1))
  else
    streak=0
  fi
  if [ "$streak" -ge "$NEED_QUIET" ]; then
    echo "[poll] sustained quiet window detected; running sweep $(date)"
    {
      echo "# Multi-warp timing sweep result ($(date -u +%Y-%m-%dT%H:%MZ))"
      echo
      echo '## num_solutions=1 (early-stop)'
      $PY scripts/perf/time_multiwarp_sweep.py --num-solutions 1 2>&1
      echo
      echo '## num_solutions=4 (no early-stop, full Krep refine)'
      $PY scripts/perf/time_multiwarp_sweep.py --num-solutions 4 2>&1
    } > "$LOG.tmp" 2>&1
    if grep -q "best W per" "$LOG.tmp"; then
      mv "$LOG.tmp" "$LOG"
      echo "SWEEP_OK $(date)" >> "$LOG"
      echo "[poll] sweep complete -> $LOG"
      exit 0
    else
      echo "[poll] sweep aborted (contention reappeared mid-run); keep polling"
      cat "$LOG.tmp" | tail -3
      streak=0
    fi
  fi
  sleep "$INTERVAL"
done
echo "# Multi-warp timing: GPU never reached a sustained quiet window before deadline ($(date))" > "$LOG"
echo "SWEEP_TIMEOUT $(date)" >> "$LOG"
echo "[poll] gave up after deadline; no quiet window."
exit 1
