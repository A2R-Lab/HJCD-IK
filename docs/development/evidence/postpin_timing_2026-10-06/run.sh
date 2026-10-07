#!/usr/bin/env bash
# Local, precompiled post-pin confirmation. Never builds, pulls, pushes or retunes.
set -euo pipefail
ROOT=/home/plancher/Desktop/HJCD-IK
HERE="$ROOT/docs/open-tasks/postpin_timing_2026-10-05"
case "${1:-}" in
  --check|--run) mode="$1" ;;
  *) echo "Usage: bash $HERE/run.sh --check|--run" >&2; exit 2 ;;
esac
[[ $# == 1 ]] || { echo "Exactly one mode is required." >&2; exit 2; }
cd "$ROOT"
[[ -z "$(git status --porcelain)" ]] || { echo "Primary checkout is dirty; stop and re-stage." >&2; exit 2; }
[[ -z "$(git -C docs/open-tasks/audit_gate_2026-10-04/branch status --porcelain)" ]] || {
  echo "Reference checkout is dirty; stop and investigate." >&2; exit 2;
}
sha256sum --check "$HERE/SHA256SUMS"
.venv/bin/python scripts/perf/run_audit_gate.py --config "$HERE/gate.json" --out "$HERE/unused-preflight" --check
[[ "$mode" == --run ]] || exit 0

# Shared advisory lock: the coordinator must NOT hold it itself while launching this job.
exec 9>/tmp/a2rlab-timing.lock
flock -n 9 || { echo "Another timing job holds /tmp/a2rlab-timing.lock; no run started." >&2; exit 3; }
# Check aggregate utilization only BEFORE our workers start. During the run the existing
# gate checks foreign compute PIDs, not our own time-averaged utilization.
for sample in 1 2 3; do
  util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)"
  [[ "$util" =~ ^[[:space:]]*0[[:space:]]*$ ]] || {
    echo "GPU is not idle (utilization: $util). Preserve the quiet window; do not bypass." >&2; exit 3;
  }
  sleep 1
done
OUT="$HERE/results-$(date -u +%Y%m%dT%H%M%S%NZ)"
[[ ! -e "$OUT" ]] || { echo "Output already exists: $OUT" >&2; exit 2; }
exec > >(tee "$OUT.launcher.log") 2>&1
printf 'Launcher PID: %s\nResults: %s\n' "$$" "$OUT"
printf '%s\n' "$$" > "$OUT.pid"
date -u
nvidia-smi --query-gpu=name,uuid,driver_version,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem --format=csv
echo "Coordinator must ensure no concurrent GPU work or CPU-heavy compiles. No settings are changed."
# Translate TERM and INT into Python cleanup: run_audit_gate's finally block terminates
# and waits for ONLY its own worker. The parent keeps lock FD 9 until cleanup is done.
exec .venv/bin/python -c '
import runpy, signal, sys
signal.signal(signal.SIGTERM, signal.default_int_handler)
signal.signal(signal.SIGINT, signal.default_int_handler)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
' "$ROOT/scripts/perf/run_audit_gate.py" --config "$HERE/gate.json" --out "$OUT" --quiet-window
