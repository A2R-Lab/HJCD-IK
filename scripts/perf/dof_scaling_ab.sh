#!/usr/bin/env bash
# DoF scaling regression harness — RUN ONLY ON A QUIET GPU (shared with other agents).
#
# For each DoF in {7,12,18,24} it regenerates grid.cuh for that robot, rebuilds the
# extension (scripts/setup/rebuild.sh — ninja alone leaves the imported .so STALE), then runs
# the fp32-vs-fp64 LM-solve A/B (scripts/perf/dof_scaling_ab.py). Panda is restored on exit.
#
# Decisive test (hypothesis #1 in docs/open-tasks/dof_scaling_regression_2026-06-18.md):
# if the fp64/fp32 ratio grows with DoF and fp32 stays ~flat, the fp64 O(DoF^3)
# warp-Cholesky is the cause -> fp32/mixed solve becomes the high-DoF default.
#
# With --nsys it ALSO profiles the default (fp64) build per DoF and prints the
# coarse_search-vs-lm_tuner kernel split (hypothesis #2: is it the solve or the FK chain).
#
# Usage:
#   scripts/perf/dof_scaling_ab.sh                  # A/B all DoF, results -> stdout + local doc
#   scripts/perf/dof_scaling_ab.sh --dofs 7,24      # subset
#   scripts/perf/dof_scaling_ab.sh --nsys           # + per-DoF fp64 kernel split
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root
PY=.venv/bin/python
OUT="docs/open-tasks/paper_sweep_results_5090.md"   # local (gitignored) results sink

DOFS="7,12,18,24"
NSYS=0
BATCH=1000
TOLS="1e-8:1e-8"   # default: tight (paper-equivalent). Sweep with --tols 1e-8:1e-8,1e-4:1e-3,1e-3:1e-2
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dofs)  DOFS="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --tols)  TOLS="$2"; shift 2 ;;
    --nsys)  NSYS=1; shift ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

# DoF -> URDF + GRiD fixed-target name (authoritative map: docs/source/user_guide/benchmarks/results.rst)
urdf_for()   { case "$1" in 7) echo csrc/urdf/panda.urdf ;;
                            12) echo csrc/urdf/panda_ext_12dof.urdf ;;
                            18) echo csrc/urdf/panda_ext_18dof.urdf ;;
                            24) echo csrc/urdf/panda_ext_24dof.urdf ;;
                            *) echo "BAD DoF: $1" >&2; return 1 ;; esac; }
target_for() { case "$1" in 7|24) echo panda_grasptarget_hand ;;
                            12|18) echo panda_hand_joint ;;
                            *) echo "BAD DoF: $1" >&2; return 1 ;; esac; }

restore_panda() {
  echo ">> restoring Panda (7-DoF) build"
  $PY scripts/codegen/generate_grid.py csrc/urdf/panda.urdf -t panda_grasptarget_hand
  scripts/setup/rebuild.sh
}
trap restore_panda EXIT

STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{
  echo ""
  echo "### DoF-scaling fp32-vs-fp64 A/B  (B=$BATCH, RTX 5090, $STAMP)"
  echo "(harness: scripts/perf/dof_scaling_ab.sh; see docs/open-tasks/dof_scaling_regression_2026-06-18.md)"
} | tee -a "$OUT"

IFS=',' read -ra DLIST <<< "$DOFS"
for dof in "${DLIST[@]}"; do
  urdf="$(urdf_for "$dof")"; tgt="$(target_for "$dof")"
  echo ""
  echo "============================================================"
  echo ">> DoF=$dof  urdf=$urdf  -t $tgt"
  echo "============================================================"
  $PY scripts/codegen/generate_grid.py "$urdf" -t "$tgt"
  scripts/setup/rebuild.sh
  $PY scripts/perf/dof_scaling_ab.py --batch "$BATCH" --tols "$TOLS" | tee -a "$OUT"

  if [[ "$NSYS" == "1" ]]; then
    # Best-effort: nsys can fail on restricted hosts (CPU-profiling perms). Never let it
    # abort the sweep — the wall-clock A/B above is the primary signal.
    set +e
    rep="/tmp/dof_${dof}_fp64"
    echo ">> nsys profiling DoF=$dof fp64 (default, tight tol) -> kernel split"
    nsys profile -o "$rep" --force-overwrite true \
      $PY scripts/perf/dof_scaling_ab.py --profile-prec fp64 --batch "$BATCH" --warmup 3 --iters 5 >/dev/null 2>&1
    {
      echo ""
      echo "_nsys kernel split DoF=$dof fp64 (coarse_search vs lm_tuner):_"
      split=$(nsys stats --report cuda_gpu_kern_sum "${rep}.nsys-rep" 2>/dev/null \
              | grep -iE 'coarse_search|lm_tuner')
      if [[ -n "$split" ]]; then echo "$split"; else echo "(nsys split unavailable on this host)"; fi
    } | tee -a "$OUT"
    set -e
  fi
done

echo ""
echo ">> A/B sweep done. Results appended to $OUT"
# restore_panda runs via the EXIT trap
