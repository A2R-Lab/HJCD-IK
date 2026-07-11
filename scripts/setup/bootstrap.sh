#!/usr/bin/env bash
# Initialize HJCD-IK's submodules on the branches the pins track:
#   external/GLASS                    -> main               (GLASS)
#   external/GRiD                     -> modernizing-tests  (GRiD)
#   external/GRiD/{GRiDCodeGenerator,URDFParser,GLASS} -> modernizing-tests
#
# Idempotent: already-initialized submodules are left on their branch (not detached).
# Heavy GRiD submodules (RBDReference, pinocchio baselines) are skipped — not needed
# for codegen/build. Run scripts/setup/setup_dev.sh for the full venv+build flow.
set -euo pipefail
cd "$(dirname "$0")/../.."

GRID_BRANCH="${GRID_BRANCH:-modernizing-tests}"
GLASS_BRANCH="${GLASS_BRANCH:-main}"

# Check out (or create at the current pin) a branch in a submodule, without
# detaching an existing branch checkout.
checkout_branch() { # <dir> <branch>
  local d="$1" b="$2"
  [ -e "$d/.git" ] || return 0
  if [ "$(git -C "$d" symbolic-ref --short -q HEAD || true)" = "$b" ]; then return 0; fi
  if git -C "$d" rev-parse --verify --quiet "refs/heads/$b" >/dev/null; then
    git -C "$d" checkout -q "$b"
  else
    git -C "$d" checkout -q -b "$b"
  fi
}

echo "[bootstrap] init external/GLASS + external/GRiD (if needed)..."
[ -e external/GLASS/.git ] || git submodule update --init external/GLASS
[ -e external/GRiD/.git ]  || git submodule update --init external/GRiD

echo "[bootstrap] init GRiD codegen submodules (GRiDCodeGenerator, URDFParser, GLASS)..."
for s in GRiDCodeGenerator URDFParser GLASS; do
  [ -e "external/GRiD/$s/.git" ] || git -C external/GRiD submodule update --init "$s"
done

echo "[bootstrap] putting submodules on feature branches..."
checkout_branch external/GLASS "$GLASS_BRANCH"
checkout_branch external/GRiD  "$GRID_BRANCH"
checkout_branch external/GRiD/GRiDCodeGenerator "$GRID_BRANCH"
checkout_branch external/GRiD/URDFParser        "$GRID_BRANCH"
checkout_branch external/GRiD/GLASS             "$GRID_BRANCH"

echo "[OK] submodules ready (GLASS on main, GRiD + nested on modernizing-tests)"
