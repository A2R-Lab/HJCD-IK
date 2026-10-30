#!/usr/bin/env bash
# Stage one precompiled Panda hand-frame endpoint in an isolated clone with its own venv.
# usage: stage.sh <label> <commit> [extra pip packages...]
set -euo pipefail
LABEL=$1; COMMIT=$2; shift 2
ROOT=/home/plancher/Desktop/HJCD-IK
HERE=$ROOT/docs/open-tasks/gate_2026-10-24
DIR=$HERE/$LABEL
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=protocol.file.allow GIT_CONFIG_VALUE_0=always
rm -rf "$DIR"
git clone -q "$ROOT" "$DIR"
cd "$DIR"
git checkout -q "$COMMIT"
git submodule init -q
for s in GRiD GLASS foam; do git config "submodule.external/$s.url" "$ROOT/external/$s"; done
git submodule update -q --init
( cd external/GRiD && git submodule init -q && \
  for s in GLASS URDFParser RBDReference; do git config "submodule.external/$s.url" "$ROOT/external/GRiD/external/$s"; done && \
  git submodule update -q --init external/GLASS external/URDFParser external/RBDReference )
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip >/dev/null
.venv/bin/pip install -q cmake ninja scikit-build-core pybind11 numpy sympy beautifulsoup4 lxml "$@"
# Hand-frame header (the gate's frame), foam spheres, the clone's own pinned GRiD codegen.
.venv/bin/python scripts/codegen/generate_grid.py csrc/urdf/panda.urdf -t panda_hand_joint --collision \
  --spherized-urdf external/foam/assets/panda/smaller_panda_spherized.urdf 2>&1 | grep -E "^\[generate_grid\]" | tail -2
PATH="$DIR/.venv/bin:$PATH" .venv/bin/pip install -q --no-build-isolation -e . 2>&1 | tail -1
git add csrc/generated/grid.cuh && git commit -q -m "gate: panda_hand_joint header" && echo "clone commit (header): $(git rev-parse HEAD) over $COMMIT"
.venv/bin/python - <<'PY'
import hashlib, json, hjcdik, hjcdik._hjcdik as m
info = hjcdik.build_info()
print(json.dumps(dict(label=__import__("os").path.basename(__import__("os").getcwd()), header_sha=info["grid_header_sha256"],
      binary=m.__file__, binary_sha=hashlib.sha256(open(m.__file__, "rb").read()).hexdigest(), ee=info["ee_target"])))
PY
