#!/usr/bin/env bash
# Run the HJCD-IK paper experiments across solvers and collect per-solver CSVs.
#
# All solvers see the SAME open-world targets (benchmark/gen_targets.py) for a fair head-to-head.
# Baselines are heavy and skippable; HJCD-IK always runs (it's the point). Output goes to OUT_DIR.
#
#   ./scripts/bench/run_paper_experiments.sh
#   SKIP_CUROBO=1 SKIP_PYROKI=1 ./scripts/bench/run_paper_experiments.sh   # HJCD-IK only
#
# Env overrides:
#   OUT_DIR     results dir                 (default: benchmark/results)
#   PYTHON      interpreter                 (default: .venv/bin/python, else python3)
#   NUM_TARGETS open-world target count     (default: 100)
#   BATCHES     batch / seed sweep          (default: 1,10,100,1000,2000)
#   PROBLEM_SET collision-free MB set       (default: bookshelf_thin_panda)
#   SKIP_HJCD / SKIP_PYROKI / SKIP_CUROBO   set =1 to skip a solver
#
# Prereqs: a built `hjcdik` (GPU) for HJCD; `scripts/setup/install_baselines.sh` for the baselines.
# Coverage (all wired; opt-in flags): Table I open-world Panda (always) + Fetch (RUN_FETCH=1),
#   Table II collision-free Panda (always), Table III DoF 7/12/18/24 (RUN_DOF=1), Table IV MMD (RUN_MMD=1).
# Extra env: HJCD_REGEN=1 re-codegens+rebuilds HJCD per EE frame / DoF and restores the default build on exit
#   (heavy: GPU compiles); RUN_FETCH / RUN_DOF / RUN_MMD / DOF_BATCH select the optional tables.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x .venv/bin/python ]; then PY=".venv/bin/python"; else PY="python3"; fi
fi
OUT_DIR="${OUT_DIR:-benchmark/results}"
NUM_TARGETS="${NUM_TARGETS:-100}"
BATCHES="${BATCHES:-1,10,100,1000,2000}"
PROBLEM_SET="${PROBLEM_SET:-bookshelf_thin_panda}"
MB_JSON="$(pwd)/tests/mb_problems.json"
TGT="$(pwd)/benchmark/targets/panda_open"
mkdir -p "$OUT_DIR" "$(dirname "$TGT")"

echo "=== [0] shared open-world targets (neutral Halton, panda_hand frame) ==="
# Match HJCD's internal sample_targets sequence for the Panda Table I targets.
"$PY" benchmark/gen_targets.py --scramble cranley-patterson --num-targets "$NUM_TARGETS" --out "$TGT"

echo "=== [0b] EE-frame equivalence check (informational; installed backends only) ==="
"$PY" benchmark/check_ee_frames.py --num 8 || echo "(frame check flagged a mismatch — see docs/source/user_guide/benchmarks/results.rst)"

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
    --seed_list "$BATCHES" --save_path "$OUT_DIR" --file_name open
fi
if [ "${SKIP_CUROBO:-0}" != "1" ]; then
  echo "--- cuRobo ---"
  "$PY" benchmark/baseline_bench.py --mode curobo --goal_file "$TGT.yml" \
    --seed_list "$BATCHES" --save_path "$OUT_DIR" --file_name open
fi
if [ "${SKIP_IKFLOW:-0}" != "1" ]; then
  echo "--- IKFlow ---"
  "$PY" benchmark/baseline_ikflow.py --goal_file "$TGT.yml" --seed_list "$BATCHES" \
    --csv-out "$OUT_DIR/open_ikflow.csv" || echo "(IKFlow skipped — not installed)"
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
    --seed_list "$BATCHES" --save_path "$OUT_DIR" --file_name fetch_open
  [ "${SKIP_CUROBO:-0}" = "1" ] || "$PY" benchmark/baseline_bench.py --mode curobo --goal_file "$FTGT.yml" \
    --robot-urdf csrc/urdf/fetch.urdf --ee-link ee_link --base-link arm_mount_link \
    --seed_list "$BATCHES" --save_path "$OUT_DIR" --file_name fetch_open
  [ "${SKIP_IKFLOW:-0}" = "1" ] || "$PY" benchmark/baseline_ikflow.py --goal_file "$FTGT.yml" \
    --model fetch_full_temp_nsc_tpm --seed_list "$BATCHES" --csv-out "$OUT_DIR/fetch_open_ikflow.csv" || echo "(IKFlow fetch skipped)"
  "$PY" benchmark/make_tables.py $OUT_DIR/fetch_open_*.csv --title "Fetch open-world (Table I)" \
    --out "$OUT_DIR/table_fetch.md" || true
  "$PY" benchmark/plot_pareto.py $OUT_DIR/fetch_open_*.csv --out "$OUT_DIR/pareto_fetch.png" \
    --title "Fetch open-world" --annotate-batch || true
fi

echo "=== [Table II] collision-free, Panda, $PROBLEM_SET ==="
# The MotionBenchMaker problems are posed in the panda_grasptarget_hand EE frame, NOT the panda_hand
# open-world frame. Table I's HJCD_REGEN leaves HJCD on panda_hand, so re-pin to grasptarget here or
# every target is solved in the wrong frame (constant ~563 mm error).
if [ "${HJCD_REGEN:-0}" = "1" ] && [ "${SKIP_HJCD:-0}" != "1" ]; then
  echo "--- regen HJCD-IK to panda_grasptarget_hand (MB problems are in this frame) ---"
  "$PY" scripts/codegen/generate_grid.py csrc/urdf/panda.urdf -t panda_grasptarget_hand && bash scripts/setup/rebuild.sh
fi
if [ "${SKIP_HJCD:-0}" != "1" ]; then
  echo "--- HJCD-IK ---"
  "$PY" benchmark/hjcd_ik_bench.py --skip-grid-codegen --collision-free \
    --problems-json "$MB_JSON" --problem-set "$PROBLEM_SET" \
    --batches "$BATCHES" --num-solutions 1 --solver hjcdik --csv-out "$OUT_DIR/collfree_hjcdik.csv"
fi
if [ "${SKIP_PYROKI:-0}" != "1" ]; then
  echo "--- PyRoki ---"
  MB_JSON_PATH="$MB_JSON" "$PY" benchmark/baseline_bench.py --mode pyroki --collision_free \
    --problem_set "$PROBLEM_SET" --num_instances "$NUM_TARGETS" \
    --seed_list "$BATCHES" --save_path "$OUT_DIR" --file_name collfree
fi
if [ "${SKIP_CUROBO:-0}" != "1" ]; then
  echo "--- cuRobo ---"
  MB_JSON_PATH="$MB_JSON" "$PY" benchmark/baseline_bench.py --mode curobo --collision_free \
    --problem_set "$PROBLEM_SET" --num_instances "$NUM_TARGETS" \
    --robot-urdf csrc/urdf/panda.urdf --base-link panda_link0 --ee-link panda_grasptarget \
    --seed_list "$BATCHES" --save_path "$OUT_DIR" --file_name collfree \
    || echo "(cuRobo Table II skipped — solver error; column left blank, run continues)"
fi

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
      --robot-urdf "$urdf" --ee-link panda_hand --base-link panda_link0 --seed_list "$DOF_B" --save_path "$OUT_DIR" --file_name "dof${d}"
    [ "${SKIP_CUROBO:-0}" = "1" ] || "$PY" benchmark/baseline_bench.py --mode curobo --goal_file "$dtgt.yml" \
      --robot-urdf "$urdf" --ee-link panda_hand --base-link panda_link0 --seed_list "$DOF_B" --save_path "$OUT_DIR" --file_name "dof${d}" \
      || echo "(cuRobo DoF=$d skipped — solver error; run continues)"
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
      --goal_file "$TGT.yml" --mmd_dump "$DUMPS/pyroki.json" --solutions_seed 2000 --solutions_k 50
  [ "${SKIP_CUROBO:-0}" = "1" ] || "$PY" benchmark/baseline_bench.py --mode curobo \
      --goal_file "$TGT.yml" --mmd_dump "$DUMPS/curobo.json" --solutions_seed 2000 --solutions_k 50 \
      || echo "(cuRobo MMD dump skipped — solver error; run continues)"
  [ "${SKIP_IKFLOW:-0}" = "1" ] || "$PY" benchmark/baseline_ikflow.py --goal_file "$TGT.yml" \
      --mmd-dump "$DUMPS/ikflow.json" --mmd-batch 2000 --solutions-count 50 || echo "(ikflow mmd dump skipped)"
  "$PY" benchmark/gen_groundtruth_tracik.py --targets "$TGT.json" --tip panda_hand \
      --num-samples 50 --out "$DUMPS/groundtruth.json" || echo "(groundtruth skipped — TRAC-IK not installed)"
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
"$PY" benchmark/make_tables.py $OUT_DIR/collfree_*.csv --title "Panda collision-free (Table II)" \
  --out "$OUT_DIR/table_collfree.md" || true
"$PY" benchmark/plot_pareto.py $OUT_DIR/open_*.csv --out "$OUT_DIR/pareto_open.png" \
  --title "Panda open-world" --annotate-batch || echo "(open plot skipped — pip install -e '.[plots]')"
"$PY" benchmark/plot_pareto.py $OUT_DIR/collfree_*.csv --out "$OUT_DIR/pareto_collfree.png" \
  --title "Panda collision-free" --annotate-batch || true

if [ "${HJCD_REGEN:-0}" = "1" ] && [ "${SKIP_HJCD:-0}" != "1" ]; then
  echo "=== restoring HJCD-IK to the default panda_grasptarget_hand build ==="
  "$PY" scripts/codegen/generate_grid.py csrc/urdf/panda.urdf -t panda_grasptarget_hand && bash scripts/setup/rebuild.sh
fi

echo "=== done. Outputs in $OUT_DIR/ ==="
ls -1 "$OUT_DIR"
