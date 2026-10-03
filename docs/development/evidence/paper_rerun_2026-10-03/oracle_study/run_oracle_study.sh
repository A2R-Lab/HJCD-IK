#!/usr/bin/env bash
# Collect returned configurations (dataset protocol: panda_hand, goal_pose) from HJCD, cuRobo and PyRoki on
# every MotionBenchMaker set at the given batch sizes, for offline multi-oracle scoring.
set -uo pipefail
STAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$STAGE/hjcd"
OUT="$STAGE/oracle_study"; mkdir -p "$OUT"
BATCHES="${BATCHES:-100,2000}"
MB="$PWD/tests/mb_problems.json"
SETS=$(.venv/bin/python -c "import json,sys;print(' '.join(sorted(json.load(open(sys.argv[1]))['problems'])))" "$MB")
START=$(date +%s)
for set in $SETS; do
  echo "--- $set"
  .venv/bin/python benchmark/hjcd_ik_bench.py --skip-grid-codegen --collision-free --target-mode goal \
    --collision-validation-model hjcd --problems-json "$MB" --problem-set "$set" --batches "$BATCHES" \
    --num-solutions 1 --yaml-out "$OUT/hjcd_$set.yml" --csv-out "$OUT/hjcd_$set.csv" \
    --configs-out "$OUT/hjcdik.jsonl" > "$OUT/log_hjcd_$set.txt" 2>&1 || echo "hjcd $set rc=$?"
  MB_JSON_PATH="$MB" .venv/bin/python benchmark/baseline_bench.py --mode curobo --collision_free --mb-target goal \
    --collision-validation-model hjcd --problem_set "$set" --num_instances 100 \
    --robot-urdf csrc/urdf/panda.urdf --base-link panda_link0 --ee-link panda_hand \
    --seed_list "$BATCHES" --save_path "$OUT" --file_name "cur_$set" \
    --configs_out "$OUT/curobo.jsonl" > "$OUT/log_curobo_$set.txt" 2>&1 || echo "curobo $set rc=$?"
  MB_JSON_PATH="$MB" .venv/bin/python benchmark/baseline_bench.py --mode pyroki --collision_free --mb-target goal \
    --ee-link panda_hand --collision-validation-model hjcd --problem_set "$set" --num_instances 100 \
    --seed_list "$BATCHES" --save_path "$OUT" --file_name "pyr_$set" \
    --configs_out "$OUT/pyroki.jsonl" > "$OUT/log_pyroki_$set.txt" 2>&1 || echo "pyroki $set rc=$?"
done
echo "elapsed $(( $(date +%s) - START ))s"
wc -l "$OUT"/*.jsonl
