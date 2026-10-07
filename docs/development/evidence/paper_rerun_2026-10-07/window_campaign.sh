#!/usr/bin/env bash
# Tomorrow's quiet-window job: the corrected full paper/competitor campaign in the isolated staging clone.
#   bash window_campaign.sh --check   # CPU-only preflight (clean clone at the expected commit, venv imports, GPU idle)
#   bash window_campaign.sh --run     # the campaign under /tmp/a2rlab-timing.lock (~25 min; reserve 30)
set -euo pipefail
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$S/hjcd"
EXPECT_COMMIT="${EXPECT_COMMIT:-$(git -C /home/plancher/Desktop/HJCD-IK rev-parse HEAD)}"
mode="${1:-}"; [[ "$mode" == --check || "$mode" == --run ]] || { echo "usage: $0 --check|--run" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "$EXPECT_COMMIT" ]] || { echo "staging clone at $(git rev-parse --short HEAD), expected ${EXPECT_COMMIT:0:7}"; exit 2; }
[[ -z "$(git status --porcelain | grep -v '^?? benchmark/reference/panda_visual' | grep -v 'csrc/generated/grid.cuh')" ]] || { echo "staging clone dirty:"; git status --short; exit 2; }
.venv/bin/python - <<'PY'
import importlib, sys
for m in ("hjcdik", "pyroki", "jaxls", "curobo", "ikflow", "tracikpy", "fcl", "trimesh", "rtree", "yaml", "pandas", "tabulate"):
    try: importlib.import_module(m)
    except Exception as e: print("MISSING", m, e); sys.exit(2)
import hjcdik; print("hjcdik", hjcdik.build_info())
PY
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -q . && { echo "GPU busy (foreign compute PID)"; [[ "$mode" == --check ]] || exit 3; }
echo "preflight OK ($mode)"; [[ "$mode" == --run ]] || exit 0
exec 9>/tmp/a2rlab-timing.lock; flock -n 9 || { echo "timing lock held; not starting"; exit 3; }
for i in 1 2 3; do u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits); [[ "$u" =~ ^[[:space:]]*0 ]] || { echo "GPU util $u, not idle"; exit 3; }; sleep 1; done
OUT_NAME="paper_results_$(date -u +%Y%m%dT%H%M%SZ)"
echo "results -> $S/$OUT_NAME (logs beside it: $OUT_NAME.run.log, $OUT_NAME.monitor.log)"; date -u
OUT_NAME="$OUT_NAME" bash "$S/run_paper_full.sh"
