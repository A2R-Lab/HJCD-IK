#!/usr/bin/env bash
# One-shot local dev setup for HJCD-IK: submodules (on our branches), a project
# venv, codegen, and an editable build — so the whole pipeline runs against our
# own copy of GRiD/GLASS.
#
#   ./scripts/setup/setup_dev.sh
#
# Sets up everything needed to build the extension AND the docs site (Sphinx + Doxygen).
#
# Env overrides:
#   GLASS_LOCAL       sibling GLASS checkout to overlay our branch from (default ~/Desktop/GLASS)
#   GLASS_BRANCH      GLASS branch to use (default feat/warp-primitives)
#   PYTHON            python interpreter (default python3)
#   SKIP_APT=1        skip the system (apt) deps step
#   SKIP_SUBMODULES=1 skip submodule init + GLASS overlay (use the current checkout)
#   SKIP_BUILD=1      set up env + codegen but skip the editable build
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"

GLASS_LOCAL="${GLASS_LOCAL:-$HOME/Desktop/GLASS}"
GLASS_BRANCH="${GLASS_BRANCH:-feat/warp-primitives}"
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
  bash scripts/setup/bootstrap.sh                       # external/GRiD + nested GRiDCodeGenerator/URDFParser
  git submodule update --init external/GLASS
  # Overlay our local GLASS branch for dev (so we build against our warp primitives).
  if [ -e "$GLASS_LOCAL/.git" ]; then
    echo "[setup] overlaying GLASS '$GLASS_BRANCH' from $GLASS_LOCAL"
    git -C external/GLASS remote add local "$GLASS_LOCAL" 2>/dev/null || true
    if git -C external/GLASS fetch -q local "$GLASS_BRANCH"; then
      git -C external/GLASS checkout -q "$GLASS_BRANCH" || git -C external/GLASS checkout -q FETCH_HEAD
    fi
  fi
else
  echo "[setup] (1/4) skipping submodules (SKIP_SUBMODULES=1) — using current checkout"
fi

echo "[setup] (2/4) venv + deps ..."
[ -d .venv ] || "$PYTHON" -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
pip install -q --upgrade pip
# Codegen (GRiD) needs numpy/sympy/bs4/lxml; tests need pytest/scipy — mirrors pyproject
# `dependencies` + the `[dev]` extra. (Kept explicit so codegen at step 3 works before the build.)
pip install -q numpy sympy beautifulsoup4 lxml pytest scipy
# Docs toolchain (Sphinx + Breathe + pydata theme) so `make -C docs all` builds the site in this venv.
pip install -q -r docs/requirements.txt

echo "[setup] (3/4) generate grid.cuh ..."
python scripts/codegen/generate_grid.py csrc/urdf/panda.urdf -t panda_grasptarget_hand

if [ "${SKIP_BUILD:-0}" = "1" ]; then
  echo "[setup] SKIP_BUILD=1 — skipping editable build."
else
  echo "[setup] (4/4) build (pip install -e .) ..."
  pip install -e .
fi

echo "[setup] done. Activate the env with:  source .venv/bin/activate"
