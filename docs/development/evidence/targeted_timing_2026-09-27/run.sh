#!/usr/bin/env bash
set -euo pipefail
timing_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
timing_python="$timing_dir/../timing-prebuild-2026-09-26/runtime/bin/python"
exec env -u PYTHONPATH -u LD_PRELOAD PYTHONSAFEPATH=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$timing_python" "$timing_dir/run.py" "$@"
