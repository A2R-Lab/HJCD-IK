#!/usr/bin/env bash
# Clearance-ladder collection (correctness only; GPU may be shared): HJCD, cuRobo (spheres) and PyRoki
# (collision-aware) on every set in ladder/mb_tight.json, dataset protocol (panda_hand, goal_pose), B=100,2000.
# Configurations are dumped for benchmark/score_collision_oracles.py. Levels with < MIN_KEPT problems are skipped.
set -uo pipefail
STAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$STAGE/hjcd"
OUT="$STAGE/ladder/results"; mkdir -p "$OUT"
MB="$STAGE/ladder/mb_tight.json"
BATCHES="${BATCHES:-100,2000}"
MIN_KEPT="${MIN_KEPT:-20}"
SETS=$(.venv/bin/python -c "
import json,sys; d=json.load(open(sys.argv[1]))
print(' '.join(s for s in sorted(d['problems']) if len(d['problems'][s]) >= int(sys.argv[2])))" "$MB" "$MIN_KEPT")
echo "sets: $SETS"
START=$(date +%s)
for set in $SETS; do
  echo "--- $set ($(date +%H:%M:%S))"
  .venv/bin/python benchmark/hjcd_ik_bench.py --skip-grid-codegen --collision-free --target-mode goal \
    --collision-validation-model hjcd --problems-json "$MB" --problem-set "$set" --batches "$BATCHES" \
    --num-solutions 1 --yaml-out "$OUT/hjcd_$set.yml" --csv-out "$OUT/hjcd_$set.csv" \
    --configs-out "$OUT/hjcdik.jsonl" > "$OUT/log_hjcd_$set.txt" 2>&1 || echo "hjcd $set rc=$?"
  MB_JSON_PATH="$MB" .venv/bin/python benchmark/baseline_bench.py --mode curobo --collision_free --mb-target goal \
    --collision-validation-model hjcd --problem_set "$set" --num_instances 100 \
    --robot-urdf csrc/urdf/panda.urdf --base-link panda_link0 --ee-link panda_hand \
    --seed_list "$BATCHES" --save_path "$OUT" --file_name "cur_$set" \
    --configs_out "$OUT/curobo.jsonl" > "$OUT/log_curobo_$set.txt" 2>&1 || echo "curobo $set rc=$?"
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 MB_JSON_PATH="$MB" .venv/bin/python benchmark/baseline_bench.py --mode pyroki --collision_free --mb-target goal \
    --ee-link panda_hand --collision-validation-model hjcd --problem_set "$set" --num_instances 100 \
    --seed_list "$BATCHES" --save_path "$OUT" --file_name "pyr_$set" \
    --configs_out "$OUT/pyroki.jsonl" > "$OUT/log_pyroki_$set.txt" 2>&1 || echo "pyroki $set rc=$?"
done
echo "elapsed $(( $(date +%s) - START ))s"
wc -l "$OUT"/*.jsonl
.venv/bin/python benchmark/score_collision_oracles.py "$OUT"/*.jsonl --problems "$MB" --out "$OUT/oracle_table.md" 2>&1 | grep -v -i warn
echo LADDER_DONE
