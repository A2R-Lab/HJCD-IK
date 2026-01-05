#!/usr/bin/env bash
set -euo pipefail

# Go to repo root (parent of scripts/)
cd "$(dirname "$0")/.."

echo "[bootstrap] syncing top-level submodules..."
git submodule sync --recursive

echo "[bootstrap] init/update GRiD submodule (top-level)..."
git submodule update --init external/GRiD

echo "[bootstrap] rewriting GRiD nested submodule URLs to HTTPS..."
git config -f external/GRiD/.gitmodules submodule.GRiDCodeGenerator.url https://github.com/A2R-Lab/GRiDCodeGenerator.git
git config -f external/GRiD/.gitmodules submodule.RBDReference.url      https://github.com/A2R-Lab/RBDReference.git
git config -f external/GRiD/.gitmodules submodule.URDFParser.url        https://github.com/A2R-Lab/URDFParser.git

echo "[bootstrap] syncing GRiD nested submodules..."
git -C external/GRiD submodule sync --recursive

echo "[bootstrap] cleaning partial nested submodule dirs (if any)..."
rm -rf external/GRiD/GRiDCodeGenerator \
       external/GRiD/RBDReference \
       external/GRiD/URDFParser || true

echo "[bootstrap] init/update all nested submodules recursively..."
git submodule update --init --recursive

echo "[OK] submodules ready"
