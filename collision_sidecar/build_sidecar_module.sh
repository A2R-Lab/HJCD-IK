#!/usr/bin/env bash
# Build the standalone _sidecar pybind extension with nvcc (Checkpoint 2, Stage 6).
# Isolated from the HJCD _hjcdik build: single TU (sidecar_module.cu #includes collision_sidecar.cu),
# so the generated header's __constant__ arrays are defined once. Does not touch the parent build.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HJCD="$(dirname "$HERE")"
OUT="${1:-$HJCD/generated}"
PYINC=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND=$(python3 -c "import pybind11; print(pybind11.get_include())")
EXT=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
PTXAS_LOG="${OUT}/sidecar_ptxas.log"

/usr/local/cuda/bin/nvcc -std=c++17 -arch=sm_89 -O3 --compiler-options -fPIC -shared \
  --ptxas-options=-v \
  -I "$HJCD/generated" -I "$HJCD/src" -I "$PYINC" -I "$PYBIND" \
  "$HERE/sidecar_module.cu" -o "$OUT/_sidecar${EXT}" 2> "$PTXAS_LOG"
echo "built $OUT/_sidecar${EXT}"
echo "ptxas register/occupancy log -> $PTXAS_LOG"
grep -E "Function properties|registers|smem|Compiling entry|_kernel" "$PTXAS_LOG" | head -40 || true
