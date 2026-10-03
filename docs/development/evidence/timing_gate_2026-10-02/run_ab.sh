#!/usr/bin/env bash
# main-vs-branch A/B, alternating order per round (ABBA) to cancel clock drift.
set -euo pipefail
STAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$STAGE"
OUT="${OUT:-$STAGE/ab_results.csv}"; ROUNDS="${ROUNDS:-6}"
DRV="$STAGE/branch/scripts/perf/timing_driver.py"; MB="$STAGE/branch/tests/mb_problems.json"
OPEN_B="${OPEN_B:-1,10,100,1000,2000,8000,32000}"; CC_B="${CC_B:-1000,2000}"
run() { # label S
  local py="$STAGE/$1/.venv/bin/python"
  env -u PYTHONPATH PYTHONSAFEPATH=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$py" "$DRV" --label "$1-S$2" --leg table1 --round "$r" \
    --targets-json "$STAGE/ab_targets.json" --batches "$OPEN_B" --num-solutions "$2" --out "$OUT" | grep -v "^\[.*hjcdik =" || true
  env -u PYTHONPATH PYTHONSAFEPATH=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$py" "$DRV" --label "$1-S$2" --leg table2 --round "$r" \
    --problems-json "$MB" --batches "$CC_B" --num-solutions "$2" --out "$OUT" | grep -v "^\[.*hjcdik =" || true
}
for r in $(seq 1 "$ROUNDS"); do
  if (( r % 2 )); then order="main branch"; else order="branch main"; fi
  for S in 1 4; do for ep in $order; do run "$ep" "$S"; done; done
done
echo "done -> $OUT"
