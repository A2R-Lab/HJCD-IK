#!/usr/bin/env bash
# HJCD built on cuRobo's sphere model (benchmark/reference/panda_curobo_spherized.urdf, panda_hand frame):
# correctness collection on the 8 dataset sets, the clearance ladder and the kitchen/table_bars sets.
set -uo pipefail
STAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$STAGE/hjcd"
.venv/bin/python -c "import hjcdik,sys; i=hjcdik.build_info(); sys.exit(0 if i['grid_header_sha256'].startswith('af92cbf7') else 1)" || { echo "wrong build"; exit 1; }
run() { # $1 problems json, $2 out dir, $3 min kept
  local MB="$1" OUT="$2"; mkdir -p "$OUT"; : > "$OUT/hjcdik_cusph.jsonl"
  for set in $(.venv/bin/python -c "
import json,sys; d=json.load(open(sys.argv[1]))
print(' '.join(s for s in sorted(d['problems']) if len(d['problems'][s]) >= int(sys.argv[2])))" "$MB" "$3"); do
    echo "--- $set ($(date +%H:%M:%S))"
    .venv/bin/python benchmark/hjcd_ik_bench.py --skip-grid-codegen --collision-free --target-mode goal --solver hjcdik_cusph \
      --collision-validation-model curobo --problems-json "$MB" --problem-set "$set" --batches 100,2000 \
      --num-solutions 1 --yaml-out "$OUT/hjcd_cusph_$set.yml" --csv-out "$OUT/hjcd_cusph_$set.csv" \
      --configs-out "$OUT/hjcdik_cusph.jsonl" > "$OUT/log_hjcd_cusph_$set.txt" 2>&1 || echo "hjcd $set rc=$?"
  done
}
run "$PWD/tests/mb_problems.json" "$STAGE/oracle_study" 1
run "$STAGE/ladder/mb_tight.json" "$STAGE/ladder/results" 20
run "$STAGE/mbm/mb_extra.json" "$STAGE/mbm/results" 1
echo "CUSPH_DONE ($(date +%H:%M:%S))"
