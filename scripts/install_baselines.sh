#!/usr/bin/env bash
# Install the competitor IK baselines (PyRoki, cuRobo) for benchmark/baseline_bench.py.
#
# These are OPTIONAL and heavy — HJCD-IK itself needs none of them. Install only the ones you want;
# each stage is independently skippable. See docs/BASELINES.md for what each baseline covers.
#
#   ./scripts/install_baselines.sh                 # install everything (PyRoki + cuRobo)
#   SKIP_CUROBO=1 ./scripts/install_baselines.sh   # PyRoki only (no torch/cuRobo build)
#   SKIP_PYROKI=1 ./scripts/install_baselines.sh   # cuRobo only
#
# Env overrides:
#   PYTHON        python interpreter / venv to install into (default: .venv/bin/python, else python3)
#   PYROKI_REF    pyroki git ref               (default: main)
#   JAXLS_REF     jaxls git ref                (default: main)
#   CUROBO_REF    cuRobo git ref               (default: main)
#   JAX_CUDA      jax CUDA extra to install    (default: jax[cuda12])
#
# NOTE: pinned sources/versions are best-effort. For an EXACT paper-matching environment, reconcile
# against the co-author's `pip freeze` (the harness was developed against their WSL setup).
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x .venv/bin/python ]; then PY=".venv/bin/python"; else PY="python3"; fi
fi
echo "[install_baselines] using interpreter: $PY"

PYROKI_REF="${PYROKI_REF:-main}"
JAXLS_REF="${JAXLS_REF:-main}"
# cuRobo API note: baseline_bench.py targets the cuRobo v2 API (curobo.inverse_kinematics.InverseKinematics
# + curobo.scene.Scene + curobo.robot_builder.RobotBuilder). That API is v0.8.0+ / `main` — the rewrite.
# (The classic v0.7.x curobo.wrap.reacher.ik_solver path is gone, and v0.7.6 won't build on CUDA13/gcc13:
# `lerp` vs C++23 `std::lerp`.) v2/main BUILDS + RUNS on Blackwell (sm_120). v2 needs a kernel backend at
# runtime: install `cuda-core` (no compilation) — done below.
CUROBO_REF="${CUROBO_REF:-main}"
CUDA_CORE_EXTRA="${CUDA_CORE_EXTRA:-cu13}"   # cuda-core wheel extra: cu13 (CUDA 13) or cu12
CUROBO_SRC="${CUROBO_SRC:-$HOME/.cache/curobo_src}"   # persistent (editable install points here; NOT /tmp)
JAX_CUDA="${JAX_CUDA:-jax[cuda12]}"

# Build + install tracikpy on a non-ROS box: clone, drop in the vendored ROS-free shims
# (benchmark/vendor/tracik/), patch setup.py to link urdfdom instead of the ROS urdf/kdl_parser libs,
# then pip install from the local checkout. Idempotent. Returns nonzero on failure.
_install_tracik_rosfree() {
  local src="${TRACIK_SRC:-$HOME/.cache/tracikpy_src}"
  local ven; ven="$(pwd)/benchmark/vendor/tracik"
  [ -d "$ven" ] || { echo "[install_baselines] vendored tracik shims missing at $ven"; return 1; }
  if [ ! -d "$src/.git" ]; then
    git clone --depth 1 https://github.com/mjd3/tracikpy.git "$src" || return 1
  fi
  mkdir -p "$src/tracikpy/include/kdl_parser" "$src/tracikpy/include/urdf"
  cp "$ven/include/kdl_parser/kdl_parser.hpp" "$src/tracikpy/include/kdl_parser/" || return 1
  cp "$ven/include/urdf/model.h"              "$src/tracikpy/include/urdf/"        || return 1
  cp "$ven/src/kdl_parser.cpp"                "$src/tracikpy/src/"                 || return 1
  # Patch setup.py (idempotent): swap ROS libs for urdfdom, add the vendored .cpp + urdfdom include dir.
  "$PY" - "$src/setup.py" <<'PYEOF' || return 1
import sys
p = sys.argv[1]
s = open(p).read()
s = s.replace('libraries=["orocos-kdl", "nlopt", "urdf", "kdl_parser"],',
              'libraries=["orocos-kdl", "nlopt", "urdfdom_model"],')
if "kdl_parser.cpp" not in s:
    s = s.replace('"tracikpy/src/kdl_tl.cpp",',
                  '"tracikpy/src/kdl_tl.cpp",\n        "tracikpy/src/kdl_parser.cpp",')
if "/usr/include/urdfdom" not in s:
    s = s.replace('"/usr/include/eigen3",',
                  '"/usr/include/eigen3",\n        "/usr/include/urdfdom",')
open(p, "w").write(s)
PYEOF
  "$PY" -m pip install --no-build-isolation --force-reinstall --no-deps "$src" || return 1
  return 0
}

# Lightweight tooling + PyRoki's pure-python friends (pip-resolvable) + plotting, via package extras.
echo "[install_baselines] (1/3) baseline tooling + plots extra (pyyaml/pandas/tabulate/jaxlie/yourdfpy/robot_descriptions/matplotlib)"
"$PY" -m pip install -e ".[baselines,plots]"

# ---- PyRoki (JAX): jax[cuda] + jaxls + pyroki ----
if [ "${SKIP_PYROKI:-0}" != "1" ]; then
  echo "[install_baselines] (2/3) PyRoki stack (jax-cuda + jaxls + pyroki)"
  "$PY" -m pip install -U "$JAX_CUDA"
  "$PY" -m pip install "jaxls @ git+https://github.com/brentyi/jaxls.git@${JAXLS_REF}"
  "$PY" -m pip install "pyroki @ git+https://github.com/chungmin99/pyroki.git@${PYROKI_REF}"
else
  echo "[install_baselines] (2/3) SKIP_PYROKI=1 — skipping PyRoki stack"
fi

# ---- cuRobo v2 (curobo@main): torch, a no-build-isolation source install, + the cuda-core runtime backend ----
if [ "${SKIP_CUROBO:-0}" != "1" ]; then
  echo "[install_baselines] (3/3) cuRobo v2 stack (torch + NVlabs/curobo@${CUROBO_REF} from source -> ${CUROBO_SRC})"
  "$PY" -m pip install torch
  # Re-clone if an existing checkout is at a different ref.
  if [ -d "$CUROBO_SRC/.git" ]; then
    cur="$(git -C "$CUROBO_SRC" rev-parse --abbrev-ref HEAD 2>/dev/null || echo none)"
    [ "$cur" = "$CUROBO_REF" ] || { echo "[install_baselines] cuRobo clone at '$cur' != '$CUROBO_REF' — re-cloning"; rm -rf "$CUROBO_SRC"; }
  fi
  if [ ! -d "$CUROBO_SRC/.git" ]; then
    git clone --depth 1 --branch "$CUROBO_REF" https://github.com/NVlabs/curobo.git "$CUROBO_SRC" \
      || echo "[install_baselines] WARNING: cuRobo clone of ${CUROBO_REF} failed."
  fi
  if MAX_JOBS="${CUROBO_MAX_JOBS:-4}" "$PY" -m pip install -e "$CUROBO_SRC" --no-build-isolation; then
    # v2 compiles no C++ by default — it JIT-compiles kernels at runtime via the cuda-core backend.
    "$PY" -m pip install "cuda-core[${CUDA_CORE_EXTRA}]" \
      || echo "[install_baselines] WARNING: cuda-core install failed — cuRobo will error at runtime ('No module named cuda.core'). Try CUDA_CORE_EXTRA=cu12."
  else
    echo "[install_baselines] WARNING: cuRobo build failed — see docs/BASELINES.md. cuRobo column will be absent."
  fi
else
  echo "[install_baselines] (3/3) SKIP_CUROBO=1 — skipping cuRobo stack"
fi

# ---- IKFlow: normalizing-flow IK baseline (Tables I + IV); torch model. Weights are loaded OFFLINE: ----
# baseline_ikflow.py merges benchmark/assets/ikflow/model_descriptions.yaml (co-author's registry) into the
# installed package and stages the local .pkl from benchmark/assets/ikflow/weights/ into ikflow's cache, so
# no download is needed (the public GCS bucket 403s). Drop the co-author's .pkl in that weights/ dir first.
if [ "${SKIP_IKFLOW:-0}" != "1" ]; then
  echo "[install_baselines] (extra) IKFlow"
  "$PY" -m pip install torch    # no-op if cuRobo stage already installed it
  "$PY" -m pip install ikflow \
    || echo "[install_baselines] WARNING: ikflow install failed (see docs/BASELINES.md); Tables I/IV IKFlow column will be absent."
  if ! ls benchmark/assets/ikflow/weights/*.pkl >/dev/null 2>&1; then
    echo "[install_baselines] NOTE: no IKFlow weights in benchmark/assets/ikflow/weights/ — add the co-author's .pkl there for offline load."
  fi
else
  echo "[install_baselines] (extra) SKIP_IKFLOW=1 — skipping IKFlow"
fi

# ---- TRAC-IK: ground-truth distribution for the MMD / Table IV metric (CPU) ----
# tracikpy is NOT on PyPI and ASSUMES a ROS environment: its C++ #includes <kdl_parser/kdl_parser.hpp>
# and <urdf/model.h>, both ROS packages absent on a bare Linux box. We build it ROS-free by vendoring
# two tiny shims (benchmark/vendor/tracik/) over urdfdom+KDL and patching setup.py. Needs system
# SWIG + KDL + NLopt + urdfdom (-dev). See docs/BASELINES.md ("TRAC-IK on a non-ROS box").
TRACIK_SRC="${TRACIK_SRC:-$HOME/.cache/tracikpy_src}"   # persistent checkout (NOT /tmp)
if [ "${SKIP_TRACIK:-0}" != "1" ]; then
  echo "[install_baselines] (extra) TRAC-IK (tracikpy, ROS-free build) for MMD ground truth -> ${TRACIK_SRC}"
  if [ "${SKIP_APT:-0}" != "1" ] && command -v apt-get >/dev/null 2>&1; then
    # liburdfdom-dev is the key one most non-ROS boxes lack (provides urdf_parser.h + liburdfdom_model.so).
    sudo apt-get install -y --no-install-recommends \
        swig liborocos-kdl-dev libnlopt-dev libnlopt-cxx-dev libeigen3-dev liburdfdom-dev liburdfdom-headers-dev \
      || echo "[install_baselines] (tracik apt deps failed; install swig + liborocos-kdl-dev + libnlopt-dev + liburdfdom-dev manually)"
  fi
  if _install_tracik_rosfree; then
    echo "[install_baselines] tracikpy installed."
  else
    echo "[install_baselines] WARNING: tracikpy failed (needs swig + liborocos-kdl-dev + libnlopt-dev + liburdfdom-dev; see docs/BASELINES.md). MMD/Table IV will lack ground truth."
  fi
else
  echo "[install_baselines] (extra) SKIP_TRACIK=1 — skipping TRAC-IK"
fi

echo "[install_baselines] done. Quick check:"
echo "    $PY benchmark/baseline_bench.py --mode pyroki --goal_file benchmark/targets/panda_open.yml"
