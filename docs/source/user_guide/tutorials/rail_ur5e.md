# Illustrative rail-mounted UR5e

This example adds a one-metre X rail to the shipped nominal UR5e. It exercises a
seven-joint, mixed-unit chain without changing the packaged Panda or registering a
new production preset. It is an illustrative assembly, not a manufacturer-approved
robot or a model validated against physical hardware.

From a source checkout with the build extra installed (Python 3.10+, `nvcc` for
the first build):

```bash
python examples/rail_ur5e.py --cache /tmp/hjcd-rail-example
python examples/rail_ur5e.py --cache /tmp/hjcd-rail-example --load-only --precision fp64
```

An absolute script path works from another directory. The first command assembles
the two input URDFs from the installed UR5e assets and compiles through GRiD's store.
Subsequent builds reuse unchanged inputs; `--load-only` needs `grid-rbd`, but neither
a compiler nor the build step. Keep the example's `rail_ur5e_targets.json` beside it.

## Frames, coordinates and geometry

The fixed root is `world`. `rail_joint` translates `base_link` along world +X over
[0, 1] metres. The remaining six coordinates are the original UR5e joints in radians,
in `solver.info.joint_names` order. Tool poses refer to `tool0`, selected by the fixed
joint `flange-tool0`, and use `[x,y,z,qw,qx,qy,qz]` in the world/root frame.

The generator changes only `base_joint` from fixed to prismatic (renamed `rail_joint`)
and preserves all supplied sphere geometry, including `base_link_inertia`. Those
base spheres now move and participate in collision checking. GRiD excludes pairs on
the same or adjacent movable anchors; the example adds no exclusions. The fit can
reject configurations a more detailed physical model would accept.

Carriage, rail, supports, cabling and end-effector hardware are **not modeled**.
The obstacle scene contains one explicitly placed sphere, not a factory workspace.
The URDF is a kinematics example: the source's helper links have no inertials, and
the assembly does not claim a validated dynamics model. Parser inertia warnings do
not establish a kinematic failure; independent FK and collision placement are tested.

UR5e assets retain BSD-3-Clause attribution to Universal Robots ROS2 Description
at `22f055da2fa7e2158254426107d1f257fd56aebb`. The generated directory includes its
license. The shipped `csrc/collision/ur5e_provenance.json` reports sampled fidelity:
the sphere fit is not a mesh-level or physical-safety certificate. See
{doc}`custom_robot` for the nominal-calibration and geometry limitations.

## Workflow and acceptance

The example loads two frozen 32-target sets, installs a world, solves, and reuses
only candidates satisfying both 5 mm and 0.001 rad for neighboring target poses.
It then installs a fully blocking sphere, handles zero output without indexing an
absent row, restores the earlier world, and checks that old results remain valid.
A miss is reported as no qualifying solution found within the budget, not proof of
infeasibility. `count`/`valid` describe returned slots; `ok` tests pose accuracy.
Hard collision validity applies to the supplied sphere model only, not a trajectory.

Witness configurations and nearby poses come from fixed-seed independent NumPy FK
and sphere checks, with a 3 mm selection clearance, not selection by IK success.
Regenerate them with `python scripts/codegen/make_rail_targets.py` in the development
environment. The tests require at least 31/32 accurate targets per scene/precision
for cold and neighboring warm starts at B=2000/S=4; every hard result is independently
checked. There is no speed or identical-configuration requirement.

For the full installed-package check, separate from routine pytest:

```bash
.venv/bin/python scripts/setup/check_installed.py --out /tmp/hjcd-installed-check
```

Use a new output directory each time. It retains the source archive, wheel, hashes,
commands and logs. It tests packaged Panda without build extras, then the rail journey
in an isolated environment outside the checkout, and fresh-process loading with build
calls forbidden. This check needs pip access or cached dependencies; it performs no
performance measurements. The smaller `mixed_joint_warm_start.py` remains the minimal
two-joint introduction.
