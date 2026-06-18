#!/usr/bin/env bash
# Rebuild the CUDA extension AND install it where Python imports it.
#
# WHY THIS EXISTS: `ninja -C build` only updates build/_hjcdik*.so, but the imported module is the
# editable-install copy under .venv/.../site-packages/hjcdik/. Running ninja alone leaves the RUNNING
# binary stale — a trap that silently invalidated a full timing/correctness pass (2026-06-18). Always
# rebuild via this script (or `pip install -e . --no-build-isolation`) so what you test is what you built.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
NINJA=$(command -v ninja || echo .venv/bin/ninja)
"$NINJA" -C build
SO=$(ls build/_hjcdik*.so)
DST=$(.venv/bin/python -c "import sysconfig,glob,os; \
sp=sysconfig.get_paths()['purelib']; \
print(glob.glob(os.path.join(sp,'hjcdik','_hjcdik*.so'))[0])")
cp "$SO" "$DST"
echo "installed $SO -> $DST"
.venv/bin/python -c "import hjcdik; print('import OK:', hjcdik.__file__)"
