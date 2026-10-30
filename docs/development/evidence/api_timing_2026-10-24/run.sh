#!/usr/bin/env bash
# Precompiled A/B gate, 2026-10-24: pre-API build (main) vs current main (branch), plus the runtime-built
# Panda (branch-rt) and a 2 cm clearance margin (branch-margin). Never builds, pulls, pushes or retunes.
set -euo pipefail
ROOT=/home/plancher/Desktop/HJCD-IK
HERE="$ROOT/docs/open-tasks/gate_2026-10-24"
case "${1:-}" in
  --check|--run) mode="$1" ;;
  *) echo "Usage: bash $HERE/run.sh --check|--run" >&2; exit 2 ;;
esac
cd "$ROOT"
[[ -z "$(git status --porcelain)" ]] || { echo "Primary checkout is dirty; stop and re-stage." >&2; exit 2; }
for ep in main branch; do
  [[ -z "$(git -C "$HERE/$ep" status --porcelain)" ]] || { echo "Endpoint $ep checkout is dirty." >&2; exit 2; }
done
sha256sum --check "$HERE/SHA256SUMS"
.venv/bin/python scripts/perf/run_audit_gate.py --config "$HERE/gate.json" --out "$HERE/unused-preflight" --check
[[ "$mode" == --run ]] || exit 0
exec 9>/tmp/a2rlab-timing.lock
flock -n 9 || { echo "Another timing job holds /tmp/a2rlab-timing.lock; no run started." >&2; exit 3; }
for sample in 1 2 3; do
  util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)"
  [[ "$util" =~ ^[[:space:]]*0[[:space:]]*$ ]] || { echo "GPU is not idle (utilization: $util)." >&2; exit 3; }
  sleep 1
done
OUT="$HERE/results-$(date -u +%Y%m%dT%H%M%SZ)"
[[ ! -e "$OUT" ]] || { echo "Output already exists: $OUT" >&2; exit 2; }
exec > >(tee "$OUT.launcher.log") 2>&1
printf 'Launcher PID: %s\nResults: %s\n' "$$" "$OUT"
date -u
nvidia-smi --query-gpu=name,uuid,driver_version,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem --format=csv
exec .venv/bin/python -c '
import runpy, signal, sys
signal.signal(signal.SIGTERM, signal.default_int_handler)
signal.signal(signal.SIGINT, signal.default_int_handler)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
' "$ROOT/scripts/perf/run_audit_gate.py" --config "$HERE/gate.json" --out "$OUT" --quiet-window
