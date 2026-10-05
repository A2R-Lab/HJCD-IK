#!/usr/bin/env bash
# Run the HJCD-IK paper experiments across solvers and collect per-solver CSVs.
#
# All solvers see the SAME open-world targets (benchmark/gen_targets.py) for a fair head-to-head.
# Baselines are heavy and skippable; HJCD-IK always runs (it's the point). Output goes to OUT_DIR.
#
#   HJCD_REGEN=1 ./scripts/bench/run_paper_experiments.sh
#   HJCD_REGEN=1 SKIP_CUROBO=1 SKIP_PYROKI=1 SKIP_IKFLOW=1 ./scripts/bench/run_paper_experiments.sh
#
# Env overrides:
#   OUT_DIR     results dir                 (default: benchmark/results)
#   PYTHON      interpreter                 (default: .venv/bin/python, else python3)
#   NUM_TARGETS open-world target count     (default: 100)
#   BATCHES     batch / seed sweep          (default: 1,10,100,1000,2000)
#   PROBLEM_SET collision-free MB set       (default: box_panda)
#   SKIP_HJCD / SKIP_PYROKI / SKIP_CUROBO   set =1 to skip a solver
#
# Prereqs: a built `hjcdik` (GPU) for HJCD; `scripts/setup/install_baselines.sh` for the baselines.
# Coverage (all wired; opt-in flags): Table I open-world Panda (always) + Fetch (RUN_FETCH=1),
#   Table II collision-free Panda: paper protocol on PROBLEM_SET + dataset protocol (panda_hand, goal_pose) on
#   MB_SETS (default PROBLEM_SET; RUN_HARD=1 = every set of tests/mb_problems.json + benchmark/problems/mb_extra_panda.json), Table III DoF 7/12/18/24 (RUN_DOF=1), Table IV MMD (RUN_MMD=1).
# Extra env: HJCD_REGEN=1 re-codegens+rebuilds HJCD per EE frame / DoF and restores the default build on successful completion
#   (heavy: GPU compiles); RUN_FETCH / RUN_DOF / RUN_MMD / DOF_BATCH select the optional tables.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x .venv/bin/python ]; then PY=".venv/bin/python"; else PY="python3"; fi
fi
# This workflow switches between panda_hand, grasptarget and optional robot variants.
# Refuse to benchmark a stale or mismatched compiled model.
if [ "${SKIP_HJCD:-0}" != "1" ] && [ "${HJCD_REGEN:-0}" != "1" ]; then
  echo "ERROR: HJCD_REGEN=1 is required to rebuild the matching robot/frame for each paper workload." >&2
  exit 2
fi
OUT_DIR="${OUT_DIR:-benchmark/results}"
if [ -e "$OUT_DIR" ]; then
  echo "ERROR: OUT_DIR must be a new directory; refusing to mix or overwrite previous evidence." >&2
  exit 2
fi
NUM_TARGETS="${NUM_TARGETS:-100}"
BATCHES="${BATCHES:-1,10,100,1000,2000}"
PROBLEM_SET="${PROBLEM_SET:-box_panda}"
MB_JSON="$(pwd)/tests/mb_problems.json"
TGT="$(pwd)/benchmark/targets/panda_open"
mkdir -p "$OUT_DIR" "$(dirname "$TGT")"
# Every Table II solve appends its returned configuration here (one JSON line per problem x batch), so the
# collision claims can be re-scored offline under independent oracles (benchmark/score_collision_oracles.py).
CFG_PAPER="$OUT_DIR/configs_collfree_paper"; CFG_HAND="$OUT_DIR/configs_collfree_hand"
mkdir -p "$CFG_PAPER" "$CFG_HAND"

echo "=== [0] shared open-world targets (neutral Halton, panda_hand frame) ==="
# Match HJCD's internal sample_targets sequence for the Panda Table I targets.
"$PY" benchmark/gen_targets.py --scramble cranley-patterson --num-targets "$NUM_TARGETS" --out "$TGT"

echo "=== [0b] EE-frame equivalence check (installed backends) ==="
"$PY" benchmark/check_ee_frames.py --num 8

echo "=== [Table I] open-world, Panda ==="
if [ "${HJCD_REGEN:-0}" = "1" ] && [ "${SKIP_HJCD:-0}" != "1" ]; then
  # Align HJCD-IK's EE to the shared open-world frame (panda_hand). Heavy: codegen + rebuild (GPU).
  echo "--- regen HJCD-IK to panda_hand frame (matches shared open-world targets) ---"
  "$PY" scripts/codegen/generate_grid.py csrc/urdf/panda.urdf -t panda_hand_joint
  bash scripts/setup/rebuild.sh
fi
if [ "${SKIP_HJCD:-0}" != "1" ]; then
  echo "--- HJCD-IK ---"
  "$PY" benchmark/hjcd_ik_bench.py --skip-grid-codegen --filtered-targets "$TGT.json" \
    --batches "$BATCHES" --num-solutions 1 --solver hjcdik --csv-out "$OUT_DIR/open_hjcdik.csv"
fi
if [ "${SKIP_PYROKI:-0}" != "1" ]; then
  echo "--- PyRoki ---"
  "$PY" benchmark/baseline_bench.py --mode pyroki --goal_file "$TGT.yml" \
    --seed_list "$BATCHES" --save_path "$OUT_DIR" --file_name open \
    || { echo "ERROR: (PyRoki Table I failed)" >&2; exit 1; }
fi
if [ "${SKIP_CUROBO:-0}" != "1" ]; then
  echo "--- cuRobo ---"
  "$PY" benchmark/baseline_bench.py --mode curobo --goal_file "$TGT.yml" \
    --seed_list "$BATCHES" --save_path "$OUT_DIR" --file_name open \
    || { echo "ERROR: (cuRobo Table I failed)" >&2; exit 1; }
fi
if [ "${SKIP_IKFLOW:-0}" != "1" ]; then
  echo "--- IKFlow ---"
  "$PY" benchmark/baseline_ikflow.py --goal_file "$TGT.yml" --seed_list "$BATCHES" \
    --csv-out "$OUT_DIR/open_ikflow.csv" || { echo "ERROR: (IKFlow failed or not installed)" >&2; exit 1; }
fi

if [ "${RUN_FETCH:-0}" = "1" ]; then
  echo "=== [Table I] open-world, Fetch (EE = ee_link, zero offset) ==="
  FTGT="$(dirname "$TGT")/fetch_open"
  "$PY" benchmark/gen_targets.py --urdf csrc/urdf/fetch.urdf --target ee_fixed \
    --num-targets "$NUM_TARGETS" --out "$FTGT"
  if [ "${SKIP_HJCD:-0}" != "1" ]; then
    [ "${HJCD_REGEN:-0}" = "1" ] && { "$PY" scripts/codegen/generate_grid.py csrc/urdf/fetch.urdf -t ee_fixed && bash scripts/setup/rebuild.sh; }
    "$PY" benchmark/hjcd_ik_bench.py --skip-grid-codegen --filtered-targets "$FTGT.json" \
      --batches "$BATCHES" --num-solutions 1 --solver hjcdik --csv-out "$OUT_DIR/fetch_open_hjcdik.csv"
  fi
  [ "${SKIP_PYROKI:-0}" = "1" ] || "$PY" benchmark/baseline_bench.py --mode pyroki --goal_file "$FTGT.yml" \
    --robot-urdf csrc/urdf/fetch.urdf --ee-link ee_link --base-link arm_mount_link \
    --seed_list "$BATCHES" --save_path "$OUT_DIR" --file_name fetch_open \
    || { echo "ERROR: (PyRoki Fetch failed)" >&2; exit 1; }
  [ "${SKIP_CUROBO:-0}" = "1" ] || "$PY" benchmark/baseline_bench.py --mode curobo --goal_file "$FTGT.yml" \
    --robot-urdf csrc/urdf/fetch.urdf --ee-link ee_link --base-link arm_mount_link \
    --seed_list "$BATCHES" --save_path "$OUT_DIR" --file_name fetch_open \
    || { echo "ERROR: (cuRobo Fetch failed)" >&2; exit 1; }
  [ "${SKIP_IKFLOW:-0}" = "1" ] || "$PY" benchmark/baseline_ikflow.py --goal_file "$FTGT.yml" \
    --model fetch_full_temp_nsc_tpm --seed_list "$BATCHES" --csv-out "$OUT_DIR/fetch_open_ikflow.csv" || { echo "ERROR: (IKFlow fetch skipped)" >&2; exit 1; }
  "$PY" benchmark/make_tables.py $OUT_DIR/fetch_open_*.csv --title "Fetch open-world (Table I)" \
    --out "$OUT_DIR/table_fetch.md" || true
  "$PY" benchmark/plot_pareto.py $OUT_DIR/fetch_open_*.csv --out "$OUT_DIR/pareto_fetch.png" \
    --title "Fetch open-world" --annotate-batch || true
fi

echo "=== [Table II-paper] collision-free, Panda, $PROBLEM_SET — paper protocol ==="
# Paper protocol (camera-ready Table II): the target is the closest cylinder's xy with goal_pose's z and
# orientation, solved for the TCP (HJCD panda_grasptarget_hand; cuRobo panda_grasptarget; PyRoki
# panda_hand_tcp, 103.4 mm vs our 105 mm). Physically consistent for cylinder-grasp scenes such as
# box_panda, kept for continuity with the published numbers. Every solver's collision_free column is
# validated by the same URDF-derived ("hjcd") sphere oracle.
if [ "${HJCD_REGEN:-0}" = "1" ] && [ "${SKIP_HJCD:-0}" != "1" ]; then
  echo "--- regen HJCD-IK to panda_grasptarget_hand + collision (paper-protocol TCP frame) ---"
  "$PY" scripts/codegen/generate_grid.py csrc/urdf/panda.urdf -t panda_grasptarget_hand \
    --collision --spherized-urdf external/foam/assets/panda/smaller_panda_spherized.urdf
  bash scripts/setup/rebuild.sh
fi
if [ "${SKIP_HJCD:-0}" != "1" ]; then
  echo "--- HJCD-IK ---"
  "$PY" benchmark/hjcd_ik_bench.py --skip-grid-codegen --collision-free --target-mode cylinder \
    --collision-validation-model hjcd --problems-json "$MB_JSON" --problem-set "$PROBLEM_SET" \
    --batches "$BATCHES" --num-solutions 1 --solver hjcdik --csv-out "$OUT_DIR/collfree_paper_hjcdik.csv" \
    --configs-out "$CFG_PAPER/hjcdik.jsonl"
fi
if [ "${SKIP_PYROKI:-0}" != "1" ]; then
  echo "--- PyRoki ---"
  MB_JSON_PATH="$MB_JSON" "$PY" benchmark/baseline_bench.py --mode pyroki --collision_free \
    --mb-target cylinder --ee-link panda_hand_tcp --collision-validation-model hjcd \
    --problem_set "$PROBLEM_SET" --num_instances "$NUM_TARGETS" \
    --seed_list "$BATCHES" --save_path "$OUT_DIR" --file_name collfree_paper --configs_out "$CFG_PAPER/pyroki.jsonl" \
    || { echo "ERROR: (PyRoki Table II-paper failed)" >&2; exit 1; }
fi
if [ "${SKIP_CUROBO:-0}" != "1" ]; then
  echo "--- cuRobo ---"
  MB_JSON_PATH="$MB_JSON" "$PY" benchmark/baseline_bench.py --mode curobo --collision_free \
    --mb-target cylinder --collision-validation-model hjcd \
    --problem_set "$PROBLEM_SET" --num_instances "$NUM_TARGETS" \
    --robot-urdf csrc/urdf/panda.urdf --base-link panda_link0 --ee-link panda_grasptarget \
    --seed_list "$BATCHES" --save_path "$OUT_DIR" --file_name collfree_paper --configs_out "$CFG_PAPER/curobo.jsonl" \
    || { echo "ERROR: (cuRobo Table II-paper failed)" >&2; exit 1; }
fi

# Dataset protocol: the MotionBenchMaker goal_pose exactly as posed. Those goals are panda_hand poses
# (the dataset's goal_ik solutions put panda_hand on them, 105 mm from the TCP), so every solver is
# pointed at panda_hand and no target snapping is applied. This is the only protocol that is valid on
# every set: cage_panda has no cylinders and box_panda_flipped's goal is 38 cm from any cylinder.
# MB_SETS (default: $PROBLEM_SET) selects the sets; RUN_HARD=1 runs every set in the problem file.
# Each entry is <problems-json>:<set>; RUN_HARD=1 adds every set of tests/mb_problems.json AND of
# benchmark/problems/mb_extra_panda.json (kitchen, table_bars — exported from the public MotionBenchMaker data).
MB_EXTRA_JSON="$(pwd)/benchmark/problems/mb_extra_panda.json"
if [ -n "${MB_SETS:-}" ]; then
  MB_RUNS="$(for s in $MB_SETS; do printf '%s:%s ' "$MB_JSON" "$s"; done)"
elif [ "${RUN_HARD:-0}" = "1" ]; then
  MB_RUNS="$("$PY" -c "import json,sys
for path in sys.argv[1:]:
    print(' '.join(f'{path}:{s}' for s in sorted(json.load(open(path))['problems'])))" "$MB_JSON" "$MB_EXTRA_JSON" | tr '\n' ' ')"
else
  MB_RUNS="$MB_JSON:$PROBLEM_SET"
fi
MB_SETS="$(for r in $MB_RUNS; do printf '%s ' "${r##*:}"; done)"
echo "=== [Table II-dataset] collision-free, Panda, panda_hand frame, goal_pose as posed: $MB_SETS ==="
if [ "${HJCD_REGEN:-0}" = "1" ] && [ "${SKIP_HJCD:-0}" != "1" ]; then
  echo "--- regen HJCD-IK to panda_hand_joint + collision (dataset frame) ---"
  "$PY" scripts/codegen/generate_grid.py csrc/urdf/panda.urdf -t panda_hand_joint \
    --collision --spherized-urdf external/foam/assets/panda/smaller_panda_spherized.urdf
  bash scripts/setup/rebuild.sh
fi
for run in $MB_RUNS; do
  SET_JSON="${run%%:*}"; set="${run##*:}"
  echo "--- set $set ($(basename "$SET_JSON")) ---"
  if [ "${SKIP_HJCD:-0}" != "1" ]; then
    "$PY" benchmark/hjcd_ik_bench.py --skip-grid-codegen --collision-free --target-mode goal \
      --collision-validation-model hjcd --problems-json "$SET_JSON" --problem-set "$set" \
      --batches "$BATCHES" --num-solutions 1 --solver hjcdik --csv-out "$OUT_DIR/collfree_hand_${set}_hjcdik.csv" \
      --configs-out "$CFG_HAND/hjcdik.jsonl"
  fi
  [ "${SKIP_PYROKI:-0}" = "1" ] || MB_JSON_PATH="$SET_JSON" "$PY" benchmark/baseline_bench.py --mode pyroki --collision_free \
    --mb-target goal --ee-link panda_hand --collision-validation-model hjcd \
    --problem_set "$set" --num_instances "$NUM_TARGETS" \
    --seed_list "$BATCHES" --save_path "$OUT_DIR" --file_name "collfree_hand_${set}" --configs_out "$CFG_HAND/pyroki.jsonl" \
    || { echo "ERROR: (PyRoki Table II-dataset $set failed)" >&2; exit 1; }
  [ "${SKIP_CUROBO:-0}" = "1" ] || MB_JSON_PATH="$SET_JSON" "$PY" benchmark/baseline_bench.py --mode curobo --collision_free \
    --mb-target goal --collision-validation-model hjcd \
    --problem_set "$set" --num_instances "$NUM_TARGETS" \
    --robot-urdf csrc/urdf/panda.urdf --base-link panda_link0 --ee-link panda_hand \
    --seed_list "$BATCHES" --save_path "$OUT_DIR" --file_name "collfree_hand_${set}" --configs_out "$CFG_HAND/curobo.jsonl" \
    || { echo "ERROR: (cuRobo Table II-dataset $set failed)" >&2; exit 1; }
  # Explicit per-solver files: a `collfree_hand_${set}_*` glob would also match e.g. box_panda_flipped.
  "$PY" benchmark/make_tables.py $(ls $OUT_DIR/collfree_hand_${set}_{hjcdik,pyroki,curobo}.csv 2>/dev/null) \
    --title "Panda collision-free, $set (dataset protocol: panda_hand, goal_pose)" \
    --out "$OUT_DIR/table_collfree_hand_${set}.md" || true
done

if [ "${RUN_DOF:-0}" = "1" ]; then
  echo "=== [Table III] DoF scalability (open-world, B=${DOF_BATCH:-1000}, panda_hand frame) ==="
  DOF_B="${DOF_BATCH:-1000}"
  for spec in "7:panda.urdf" "12:panda_ext_12dof.urdf" "18:panda_ext_18dof.urdf" "24:panda_ext_24dof.urdf"; do
    d="${spec%%:*}"; urdf="csrc/urdf/${spec##*:}"; dtgt="$(dirname "$TGT")/panda_dof${d}"
    echo "--- DoF=$d ($urdf) ---"
    "$PY" benchmark/gen_targets.py --urdf "$urdf" --target panda_hand_joint --num-targets "$NUM_TARGETS" --out "$dtgt"
    if [ "${SKIP_HJCD:-0}" != "1" ]; then
      [ "${HJCD_REGEN:-0}" = "1" ] && { "$PY" scripts/codegen/generate_grid.py "$urdf" -t panda_hand_joint && bash scripts/setup/rebuild.sh; }
      "$PY" benchmark/hjcd_ik_bench.py --skip-grid-codegen --filtered-targets "$dtgt.json" \
        --batches "$DOF_B" --num-solutions 1 --solver hjcdik --csv-out "$OUT_DIR/dof${d}_hjcdik.csv"
    fi
    [ "${SKIP_PYROKI:-0}" = "1" ] || "$PY" benchmark/baseline_bench.py --mode pyroki --goal_file "$dtgt.yml" \
      --robot-urdf "$urdf" --ee-link panda_hand --base-link panda_link0 --seed_list "$DOF_B" --save_path "$OUT_DIR" --file_name "dof${d}" \
      || { echo "ERROR: (PyRoki DoF=$d failed)" >&2; exit 1; }
    [ "${SKIP_CUROBO:-0}" = "1" ] || "$PY" benchmark/baseline_bench.py --mode curobo --goal_file "$dtgt.yml" \
      --robot-urdf "$urdf" --ee-link panda_hand --base-link panda_link0 --seed_list "$DOF_B" --save_path "$OUT_DIR" --file_name "dof${d}" \
      || { echo "ERROR: (cuRobo DoF=$d failed)" >&2; exit 1; }
    "$PY" benchmark/make_tables.py $OUT_DIR/dof${d}_*.csv --title "DoF=$d (Table III)" --out "$OUT_DIR/table_dof${d}.md" || true
  done
fi

if [ "${RUN_MMD:-0}" = "1" ]; then
  echo "=== [Table IV] MMD solution-diversity (open-world, K=50 of batch 2000) ==="
  DUMPS="$OUT_DIR/dumps"; mkdir -p "$DUMPS"
  # MMD uses the Panda open-world targets (panda_hand frame); RUN_DOF (if it ran) leaves HJCD built for the
  # last DoF variant, so re-pin HJCD back to panda_hand before the HJCD dump.
  if [ "${HJCD_REGEN:-0}" = "1" ] && [ "${SKIP_HJCD:-0}" != "1" ]; then
    echo "--- regen HJCD-IK to panda_hand (for the MMD dump) ---"
    "$PY" scripts/codegen/generate_grid.py csrc/urdf/panda.urdf -t panda_hand_joint && bash scripts/setup/rebuild.sh
  fi
  [ "${SKIP_HJCD:-0}" = "1" ]   || "$PY" benchmark/hjcd_ik_bench.py --skip-grid-codegen \
      --filtered-targets "$TGT.json" --solver hjcdik --mmd-dump "$DUMPS/hjcdik.json" \
      --mmd-batch 2000 --solutions-count 50
  [ "${SKIP_PYROKI:-0}" = "1" ] || "$PY" benchmark/baseline_bench.py --mode pyroki \
      --goal_file "$TGT.yml" --mmd_dump "$DUMPS/pyroki.json" --solutions_seed 2000 --solutions_k 50 \
      || { echo "ERROR: (PyRoki MMD dump failed)" >&2; exit 1; }
  [ "${SKIP_CUROBO:-0}" = "1" ] || "$PY" benchmark/baseline_bench.py --mode curobo \
      --goal_file "$TGT.yml" --mmd_dump "$DUMPS/curobo.json" --solutions_seed 2000 --solutions_k 50 \
      || { echo "ERROR: (cuRobo MMD dump failed)" >&2; exit 1; }
  [ "${SKIP_IKFLOW:-0}" = "1" ] || "$PY" benchmark/baseline_ikflow.py --goal_file "$TGT.yml" \
      --mmd-dump "$DUMPS/ikflow.json" --mmd-batch 2000 --solutions-count 50 || { echo "ERROR: (ikflow mmd dump skipped)" >&2; exit 1; }
  "$PY" benchmark/gen_groundtruth_tracik.py --targets "$TGT.json" --tip panda_hand \
      --num-samples 50 --out "$DUMPS/groundtruth.json" || { echo "ERROR: (groundtruth skipped — TRAC-IK not installed)" >&2; exit 1; }
  DUMP_ARGS=""; for s in hjcdik pyroki curobo ikflow; do [ -f "$DUMPS/$s.json" ] && DUMP_ARGS="$DUMP_ARGS $DUMPS/$s.json"; done
  if [ -f "$DUMPS/groundtruth.json" ] && [ -n "$DUMP_ARGS" ]; then
    "$PY" benchmark/run_mmd.py --groundtruth "$DUMPS/groundtruth.json" --solver-dump $DUMP_ARGS \
        --out "$OUT_DIR/table4_mmd.md"
  else
    echo "(MMD table skipped — need groundtruth + >=1 solver dump)"
  fi
fi

echo "=== [tables + plots] merge per-solver CSVs ==="
# Unquoted globs so they expand; missing matplotlib only skips the plots (tables are stdlib-only).
"$PY" benchmark/make_tables.py $OUT_DIR/open_*.csv --title "Panda open-world (Table I)" \
  --out "$OUT_DIR/table_open.md" || true
"$PY" benchmark/make_tables.py $OUT_DIR/collfree_paper_*.csv --title "Panda collision-free, $PROBLEM_SET (Table II, paper protocol)" \
  --out "$OUT_DIR/table_collfree.md" || true
"$PY" benchmark/plot_pareto.py $OUT_DIR/open_*.csv --out "$OUT_DIR/pareto_open.png" \
  --title "Panda open-world" --annotate-batch || { echo "ERROR: (open plot skipped — pip install -e '.[plots]')" >&2; exit 1; }
"$PY" benchmark/plot_pareto.py $OUT_DIR/collfree_paper_*.csv --out "$OUT_DIR/pareto_collfree.png" \
  --title "Panda collision-free (paper protocol)" --annotate-batch || true
"$PY" benchmark/plot_pareto.py $(ls $OUT_DIR/collfree_hand_${PROBLEM_SET}_{hjcdik,pyroki,curobo}.csv 2>/dev/null) --out "$OUT_DIR/pareto_collfree_hand.png" \
  --title "Panda collision-free, $PROBLEM_SET (dataset protocol)" --annotate-batch || true
# Independent judges of the stored configurations: sphere oracles + FCL mesh (needs python-fcl; skipped otherwise).
"$PY" - "$MB_JSON" "$MB_EXTRA_JSON" "$OUT_DIR/problems_merged.json" <<'PYEOF'
import json, os, sys
merged = {}
for path in sys.argv[1:3]:
    if os.path.exists(path):
        merged.update(json.load(open(path))["problems"])
json.dump({"problems": merged}, open(sys.argv[3], "w"))
PYEOF
for proto in paper hand; do
  cfg="$OUT_DIR/configs_collfree_${proto}"
  required_sets="$MB_SETS"
  if [ "$proto" = paper ]; then required_sets="$PROBLEM_SET"; fi
  "$PY" benchmark/score_collision_oracles.py "$cfg"/*.jsonl --problems "$OUT_DIR/problems_merged.json" \
    --problem-sets $required_sets \
    --out "$OUT_DIR/table_oracles_collfree_${proto}.md" > /dev/null || { echo "ERROR: (oracle scoring for $proto skipped)" >&2; exit 1; }
done

if [ "${HJCD_REGEN:-0}" = "1" ] && [ "${SKIP_HJCD:-0}" != "1" ]; then
  echo "=== restoring HJCD-IK to the default panda_grasptarget_hand build ==="
  "$PY" scripts/codegen/generate_grid.py csrc/urdf/panda.urdf -t panda_grasptarget_hand \
    --collision --spherized-urdf external/foam/assets/panda/smaller_panda_spherized.urdf
  bash scripts/setup/rebuild.sh
fi

echo "=== done. Outputs in $OUT_DIR/ ==="
ls -1 "$OUT_DIR"
