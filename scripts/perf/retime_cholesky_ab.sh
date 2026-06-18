#!/usr/bin/env bash
# A/B retime for the Phase-3 glass cholesky swap. RUN ONLY ON A QUIET GPU
# (nvidia-smi util ~0 — timing under contention is meaningless; see the 10ms bogus run).
#
# Compares the committed glass-cholesky HEAD against the prior hand-rolled commit,
# same merged-GRiD pin, same driver. Perf-neutral expected (the solve is a tiny sliver
# of the 916us lm_tuner floor). Each variant: warmup + median-of-N via drive_generate.py.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY=.venv/bin/python

# GLASS_REF = glass-cholesky build (current HEAD); HAND_REF = the hand-rolled-cholesky baseline.
# e4bd3da is the last commit with the hand-rolled solver (same merged-GRiD pin, both __restrict__),
# so this isolates the cholesky implementation. Only src/hjcd_kernel.cu is swapped per variant.
GLASS_REF="${1:-HEAD}"
HAND_REF="${2:-e4bd3da}"

util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
if [ "${util:-0}" -gt 5 ]; then echo "GPU util=${util}% — NOT quiet. Aborting (rerun when free)."; exit 1; fi

time_variant () {
  local ref="$1" label="$2"
  echo "=== [$label] $ref ==="
  git checkout -q "$ref" -- src/hjcd_kernel.cu
  $PY -m pip install -e . --no-build-isolation >/dev/null 2>&1
  $PY scripts/perf/drive_generate.py
}

ORIG=$(git rev-parse --abbrev-ref HEAD)
time_variant "$GLASS_REF" "GLASS (new)"
time_variant "$HAND_REF"  "HAND-ROLLED (baseline)"
# restore working tree to HEAD's kernel + rebuild
git checkout -q "$ORIG" -- src/hjcd_kernel.cu
$PY -m pip install -e . --no-build-isolation >/dev/null 2>&1
echo "=== restored to $ORIG; compare the two medians above (perf-neutral = within noise) ==="
