#!/usr/bin/env bash
# Initialize the committed submodule revisions needed for building and codegen.
#
# Post packaging-fold GRiD layout: the codegen lives in the tracked grid_codegen/
# package at the GRiD root; URDFParser and the vendored-GLASS source are nested
# submodules under external/GRiD/external/. RBDReference + pinocchio baselines are
# skipped — not needed for codegen/build. Run scripts/setup/setup_dev.sh for the full flow.
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "[bootstrap] initialize external/GLASS + external/GRiD at committed revisions..."
git submodule update --init external/GLASS external/GRiD

echo "[bootstrap] init GRiD codegen deps (external/GLASS, external/URDFParser)..."
for s in external/GLASS external/URDFParser; do
  git -C external/GRiD submodule update --init "$s"
done

echo "[OK] submodules ready at committed revisions"
