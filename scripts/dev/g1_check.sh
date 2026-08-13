#!/usr/bin/env bash
# Build the extension for the Unitree G1 (29-DoF, branched, mixed joint axes) and run the
# robot-agnostic FK / joint-axis tests against it, then restore the committed Panda build.
#
# G1 is the robot that breaks every single-robot assumption HJCD carried:
#   - branched tree (is_serial_chain False)  -> GRiD emits a baked parent table
#   - TOPOLOGY_HELPERS_COUNT = 175 (Panda 0) -> the old nullptr helper pointer was a null write
#   - joint axes 13y / 9x / 7z (Panda all-z) -> the old z-column axis read was wrong for 22/29
#
# Usage: scripts/dev/g1_check.sh          (restores Panda on exit, even on failure)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY=.venv/bin/python

G1_URDF=csrc/urdf/g1_29dof_rev_1_0.urdf
PANDA_SPHERIZED="external/foam/assets/panda/smaller_panda_spherized.urdf"

restore_panda() {
  echo ">> restoring the committed Panda build (7-DoF, WITH collision)"
  $PY scripts/codegen/generate_grid.py csrc/urdf/panda.urdf -t panda_grasptarget_hand \
      --collision --spherized-urdf "$PANDA_SPHERIZED" >/dev/null
  scripts/setup/rebuild.sh >/dev/null
  echo ">> Panda restored"
}
trap restore_panda EXIT

echo ">> generating grid.cuh + hjcd_targets.cuh for G1 (4 targets, no collision)"
# The four-target set, in order. Every offset is explicit and anchored on a movable joint, because
# these frames must match the CRAG bouldering MJCF's contact sites exactly (scripts/build_g1_mjcf.py
# CONTACT_SITES) -- CRAG's tests/test_hjcd_integration.py asserts HJCD/MuJoCo FK equivalence.
#
# Hands: mid-palm. NOT `fixed=left_hand_palm_joint`, which resolves to the rubber-hand LINK ORIGIN
# (i.e. the wrist) and lands the target ~86 mm short of where the hand grips. The offset is that
# fixed joint's own origin (0.0415, +/-0.003, 0) composed with the mid-palm offset (0.085, -/+0.015,
# 0.005) that the MJCF site uses.
#
# Feet: no fixed frame exists, so anchor at the ankle-roll joint with a sole offset:
#   x = 0.035   (longitudinal centroid of the 4 foot collision spheres)
#   y = 0       (lateral centroid)
#   z = -0.030  (sphere centers; matches the MJCF site. Note this is the centroid, not the contact
#                plane -- the spheres have radius 0.005, so the sole surface is 5 mm lower at -0.035.)
#
# -t still names the single-EE frame consumed by the not-yet-migrated solver path; it is unused by
# the target metadata and is a placeholder until Phase 3. It must name a FIXED joint.
$PY scripts/codegen/generate_grid.py "$G1_URDF" -t left_hand_palm_joint \
    --target "name=left_hand;anchor=left_wrist_yaw_joint;xyz=0.1265,-0.012,0.005;rpy=0,0,0" \
    --target "name=right_hand;anchor=right_wrist_yaw_joint;xyz=0.1265,0.012,0.005;rpy=0,0,0" \
    --target "name=left_foot;anchor=left_ankle_roll_joint;xyz=0.035,0,-0.030;rpy=0,0,0" \
    --target "name=right_foot;anchor=right_ankle_roll_joint;xyz=0.035,0,-0.030;rpy=0,0,0"

echo ">> rebuilding"
scripts/setup/rebuild.sh >/dev/null

echo ">> running FK / joint-axis / target / residual tests against G1"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 HJCD_TEST_URDF="$G1_URDF" \
  $PY -m pytest tests/test_joint_axis.py tests/test_targets.py tests/test_residuals.py tests/test_lm.py tests/test_incremental.py tests/test_lm_compat.py tests/test_diagnostics.py tests/test_coarse.py \
  -v -s -p no:cacheprovider
