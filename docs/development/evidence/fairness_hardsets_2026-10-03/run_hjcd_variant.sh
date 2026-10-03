#!/usr/bin/env bash
# HJCD-only correctness collection for one kernel/model variant, all three problem files.
#   run_hjcd_variant.sh <label> <expected-header-sha-prefix> [extra hjcd_ik_bench args...]
set -uo pipefail
STAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$STAGE/hjcd"
LABEL="$1"; SHA="$2"; shift 2; EXTRA=("$@"); [ ${#EXTRA[@]} -eq 0 ] && EXTRA=()
.venv/bin/python -c "import hjcdik,sys; sys.exit(0 if hjcdik.build_info()['grid_header_sha256'].startswith('$SHA') else 1)" || { echo "wrong build"; exit 1; }
run() { local MB="$1" OUT="$2"; mkdir -p "$OUT"; : > "$OUT/${LABEL}.jsonl"
  for set in $(.venv/bin/python -c "
import json,sys; d=json.load(open(sys.argv[1]))
print(' '.join(s for s in sorted(d['problems']) if len(d['problems'][s]) >= int(sys.argv[2])))" "$MB" "$3"); do
    .venv/bin/python benchmark/hjcd_ik_bench.py --skip-grid-codegen --collision-free --target-mode goal --solver "$LABEL" \
      --collision-validation-model hjcd --problems-json "$MB" --problem-set "$set" --batches 100,2000 \
      --num-solutions 1 --yaml-out "$OUT/${LABEL}_$set.yml" --csv-out "$OUT/${LABEL}_$set.csv" \
      --configs-out "$OUT/${LABEL}.jsonl" "${EXTRA[@]}" > "$OUT/log_${LABEL}_$set.txt" 2>&1 || echo "$LABEL $set rc=$?"
  done; }
run "$PWD/tests/mb_problems.json" "$STAGE/oracle_study" 1
run "$STAGE/ladder/mb_tight.json" "$STAGE/ladder/results" 20
run "$STAGE/mbm/mb_extra.json" "$STAGE/mbm/results" 1
echo "VARIANT_DONE $LABEL ($(date +%H:%M:%S))"
