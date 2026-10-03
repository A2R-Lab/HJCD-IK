#!/usr/bin/env bash
# (1) PyRoki collision-aware re-collection on the 8 original sets; (2) re-score every study with all judges.
set -uo pipefail
STAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$STAGE/hjcd"
OUT="$STAGE/oracle_study"; MB="$PWD/tests/mb_problems.json"
: > "$OUT/pyroki_coll.jsonl"
for set in $(.venv/bin/python -c "import json,sys;print(' '.join(sorted(json.load(open(sys.argv[1]))['problems'])))" "$MB"); do
  echo "--- pyroki(coll) $set ($(date +%H:%M:%S))"
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 MB_JSON_PATH="$MB" .venv/bin/python benchmark/baseline_bench.py --mode pyroki --collision_free --mb-target goal \
    --ee-link panda_hand --collision-validation-model hjcd --problem_set "$set" --num_instances 100 \
    --seed_list 100,2000 --save_path "$OUT" --file_name "pyrcoll_$set" \
    --configs_out "$OUT/pyroki_coll.jsonl" > "$OUT/log_pyrcoll_$set.txt" 2>&1 || echo "pyroki $set rc=$?"
done
echo "--- scoring ($(date +%H:%M:%S))"
.venv/bin/python benchmark/score_collision_oracles.py "$OUT/hjcdik.jsonl" "$OUT/curobo.jsonl" "$OUT/pyroki_coll.jsonl" --problems "$MB" \
  --out "$OUT/oracle_table_v2.md" --json-out "$OUT/scores_v2.json" 2>&1 | grep -v -i warn > /dev/null || echo "score1 rc=$?"
.venv/bin/python benchmark/score_collision_oracles.py "$STAGE"/ladder/results/*.jsonl --problems "$STAGE/ladder/mb_tight.json" \
  --out "$STAGE/ladder/oracle_table_v2.md" --json-out "$STAGE/ladder/scores_v2.json" 2>&1 | grep -v -i warn > /dev/null || echo "score2 rc=$?"
.venv/bin/python benchmark/score_collision_oracles.py "$STAGE"/mbm/results/*.jsonl --problems "$STAGE/mbm/mb_extra.json" \
  --out "$STAGE/mbm/oracle_table_v2.md" --json-out "$STAGE/mbm/scores_v2.json" 2>&1 | grep -v -i warn > /dev/null || echo "score3 rc=$?"
.venv/bin/python benchmark/plot_clearance_ladder.py "$STAGE/ladder/scores_v2.json" --ladder "$STAGE/ladder/mb_tight.json" \
  --out "$STAGE/ladder/ladder_v2.png" --table "$STAGE/ladder/ladder_v2.md" 2>&1 | grep -v -i warn > /dev/null || echo "plot rc=$?"
echo "RESCORE_DONE ($(date +%H:%M:%S))"
