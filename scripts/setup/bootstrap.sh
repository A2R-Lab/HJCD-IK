#!/usr/bin/env bash
# Initialize the committed submodule revisions needed for building and codegen.
#
# Post packaging-fold GRiD layout: the codegen lives in the tracked grid_codegen/
# package at the GRiD root; URDFParser and the vendored-GLASS source are nested
# submodules under external/GRiD/external/. RBDReference + pinocchio baselines are
# skipped — not needed for codegen/build. Run scripts/setup/setup_dev.sh for the full flow.
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "[bootstrap] initialize external/GLASS + external/GRiD + external/foam at committed revisions..."
git submodule update --init external/GLASS external/GRiD external/foam

echo "[bootstrap] init GRiD codegen deps (external/GLASS, external/URDFParser)..."
for s in external/GLASS external/URDFParser; do
  git -C external/GRiD submodule update --init "$s"
done

# grid.cuh is generated with vendor_glass=False, so it compiles against the top-level
# external/GLASS while GRiD's codegen is written against its own nested GLASS pin.
top_glass=$(git -C external/GLASS rev-parse HEAD)
nested_glass=$(git -C external/GRiD/external/GLASS rev-parse HEAD)
if [[ "$top_glass" != "$nested_glass" ]]; then
  echo "ERROR: external/GLASS ($top_glass) != external/GRiD/external/GLASS ($nested_glass)." >&2
  echo "       Bump the pins together, then regenerate grid.cuh (scripts/codegen/generate_grid.py)." >&2
  exit 1
fi

echo "[OK] submodules ready at committed revisions"
