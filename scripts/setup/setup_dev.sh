#!/usr/bin/env bash
# One-shot local dev setup for HJCD-IK: pinned submodules, a project
# venv, codegen, and an editable build — so the whole pipeline runs against our
# own copy of GRiD/GLASS.
#
#   ./scripts/setup/setup_dev.sh
#
# Sets up everything needed to build the extension AND the docs site (Sphinx + Doxygen).
#
# Env overrides:
#   GLASS_LOCAL       optional sibling GLASS checkout to overlay (unset by default)
#   GLASS_BRANCH      GLASS branch to use (default main)
#   PYTHON            python interpreter (default python3)
#   SKIP_APT=1        skip the system (apt) deps step
#   SKIP_SUBMODULES=1 skip submodule init + GLASS overlay (use the current checkout)
#   SKIP_BUILD=1      set up env + codegen but skip the editable build
set -euo pipefail
cd "$(dirname "$0")/../.."
GLASS_LOCAL="${GLASS_LOCAL:-}"
GLASS_BRANCH="${GLASS_BRANCH:-main}"
PYTHON="${PYTHON:-python3}"

# System (C++/CUDA) build dependencies. The build needs the CUDA toolkit (nvcc) plus two
# header libraries: Eigen3 and nlohmann-json (the collision env parser includes it).
# Set SKIP_APT=1 to skip the apt step (e.g. on non-Debian systems — install the equivalents
# manually: cuda-toolkit, libeigen3-dev, nlohmann-json3-dev).
if [ "${SKIP_APT:-0}" != "1" ] && command -v apt-get >/dev/null 2>&1; then
  echo "[setup] (0/4) system deps (Eigen3, nlohmann-json, Doxygen) via apt ..."
  sudo apt-get install -y --no-install-recommends libeigen3-dev nlohmann-json3-dev doxygen \
    || echo "[setup] WARNING: apt install failed; install libeigen3-dev + nlohmann-json3-dev + doxygen manually"
else
  echo "[setup] (0/4) skipping apt; ensure these are installed: cuda-toolkit, libeigen3-dev, nlohmann-json3-dev, doxygen (for docs)"
fi

if [ "${SKIP_SUBMODULES:-0}" != "1" ]; then
  echo "[setup] (1/4) submodules ..."
  bash scripts/setup/bootstrap.sh
  # Explicit opt-in only: normal setup must reproduce the repository's committed pins.
  if [ -n "$GLASS_LOCAL" ] && [ -e "$GLASS_LOCAL/.git" ]; then
    echo "[setup] overlaying GLASS '$GLASS_BRANCH' from $GLASS_LOCAL"
    git -C external/GLASS fetch -q "$GLASS_LOCAL" "$GLASS_BRANCH"
    git -C external/GLASS checkout -q --detach FETCH_HEAD
  fi
else
  echo "[setup] (1/4) skipping submodules (SKIP_SUBMODULES=1) — using current checkout"
fi

echo "[setup] (2/4) venv + deps ..."
[ -d .venv ] || "$PYTHON" -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
pip install -q --upgrade pip
# Install codegen and test dependencies before regeneration; the editable build happens after codegen.
# Keep this explicit to avoid building once against the committed header and immediately rebuilding.
pip install -q numpy sympy beautifulsoup4 lxml pytest scipy pytest-gpu-proof pyyaml
# Docs toolchain (Sphinx + Breathe + pydata theme) so `make -C docs all` builds the site in this venv.
pip install -q -r docs/requirements.txt

echo "[setup] (3/4) generate grid.cuh ..."
python scripts/codegen/generate_grid.py csrc/urdf/panda.urdf -t panda_grasptarget_hand \
  --collision --spherized-urdf external/foam/assets/panda/smaller_panda_spherized.urdf

if [ "${SKIP_BUILD:-0}" = "1" ]; then
  echo "[setup] SKIP_BUILD=1 — skipping editable build."
else
  echo "[setup] (4/4) build (pip install -e .) ..."
  pip install -e .
fi

echo "[setup] done. Activate the env with:  source .venv/bin/activate"
