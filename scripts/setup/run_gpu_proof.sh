#!/usr/bin/env bash
# Generate the signed GPU-proof receipt for HJCD-IK's CUDA test suite. Run this on
# a machine with a real GPU and a built `hjcdik` (see scripts/setup/rebuild.sh).
# The pytest-gpu-proof plugin comes from PyPI — install it into the venv first:
#   pip install -e '.[dev]'      (or: pip install pytest-gpu-proof pyyaml)
#
#   scripts/setup/run_gpu_proof.sh                 # full receipt -> gpu-proof.json
#   PYTEST_ARGS="-k regression" scripts/setup/run_gpu_proof.sh   # scoped dry run
#
# Every HJCD-IK test carries the gpu_proof marker (auto-applied to the whole suite
# by tests/conftest.py, since each test drives the CUDA kernel). The receipt records
# their outcomes and signs the code fingerprint with your local SSH key; CI verifies
# it CPU-only via github.com/plancherb1.keys.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Refuse to sign a dirty tree: the fingerprint can't descend into the CUDA kernel's
# build or the GRiD/GLASS submodules, so a clean tree is what makes the receipt's
# commit SHA an honest description of the code under test (mirrors
# tests/gpu-proof-policy.yaml allow_dirty:false).
if [[ -n "$(git status --porcelain)" ]]; then
    echo "ERROR: working tree is dirty. Commit or stash before signing a receipt" >&2
    echo "       (a clean tree is what pins the kernel + GRiD/GLASS submodules via" >&2
    echo "        the receipt's commit SHA)." >&2
    exit 1
fi

PYTHON="${PYTHON:-.venv/bin/python}"

exec "$PYTHON" -m pytest \
    tests \
    -m gpu_proof \
    --gpu-proof-enable \
    --gpu-proof-out=gpu-proof.json \
    ${PYTEST_ARGS:-}
