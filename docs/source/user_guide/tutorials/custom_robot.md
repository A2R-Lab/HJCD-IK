# Custom robot

Two routes lead from a URDF to a solver. The runtime build is the one to use; regenerating the
committed header is the workflow of the packaged Panda build and of isolated native builds.

## Runtime build (recommended)

Install the build extra (`python -m pip install -e ".[build]"` adds GRiD's `grid-rbd` and
`pybind11`; `nvcc` must be on the path) and build once:

```bash
hjcdik build path/to/robot.urdf --ee-link <fixed joint of the tool frame> --name my-robot
hjcdik build path/to/robot.urdf --ee-link <joint> --spheres path/to/robot_spherized.urdf   # with collision
hjcdik list
```

`hjcdik.Solver("my-robot")` then loads the prepared build with no compiler. The build goes through
`grid_rbd.build_artifact`: GRiD generates the robot's `grid.cuh`, validates the sphere model against
the URDF when one is given, and compiles HJCD-IK's translation units against it into its store
(`~/.cache/grid-rbd/consumers/hjcdik/`), keyed by everything that shapes the binary. A second build
with the same inputs is a lookup; an edited URDF, sphere model, option or source re-keys it.

The sphere model must be expressed in the robot URDF's own frames. A model made for another URDF
of the same robot (foam's Panda, whose finger frames differ from ours) is adapted with
`grid-spherize --rebind-to robot.urdf spheres.urdf --out rebound.urdf`, which re-attaches each
sphere by link name and validates the result; `csrc/collision/panda_foam_rebound.urdf` is that file
for the packaged Panda and the `"panda_foam"` preset. Rebinding copies local sphere offsets;
it does not rotate them when link coordinate frames change. Such a frame change requires
transforming the sphere centers explicitly. Without a sphere model the build is
open-world and `collision` must be `None`.

The solver model is a fixed-base serial chain of 1 to 32 independent revolute, continuous or
prismatic joints, with arbitrary axes and fixed attachments. Floating bases, actuated branches
and mimic joints are rejected. The selected fixed tool joint must attach to the final actuated
joint. Revolute and prismatic limits must be finite and strictly ordered; continuous joints use
[-pi, pi] for sampling and clipping.
GRiD 0.6.0 cannot generate all-prismatic chains: at least one revolute or continuous joint is
required. The validator reports this upstream limitation before code generation (GRiD ask G13).

`solver.info.joint_names`, `joint_types`, `joint_units` and `joint_limits` are immutable tuples
in solver order. Use that order for `q_init`; revolute coordinates are radians and prismatic
coordinates are metres. The coarse translation step is limited to one quarter of joint travel.
LM step clipping and warm-start proximity retain the existing raw-coordinate norm, mixing
metres and radians. This is not a unit-invariant metric. Correctness tests cover 1 cm, 0.5 m and
5 m slider travel before and after a rotating joint, both precisions, cold and perturbed warm
starts. They do not establish convergence for every robot or scale. A failed solve means no
qualifying solution was found within its search budget, not that the target is infeasible.

Run `python examples/mixed_joint_warm_start.py` from a source checkout for a small analytic
mixed-joint example. It builds once, prints coordinate metadata, installs a world, checks
accuracy before reusing a result as a warm start, and handles zero-result output explicitly.
Its spheres describe the demonstration geometry only, not a physical robot.
For a seven-joint assembly with neighboring warm targets and world replacement, see
{doc}`rail_ur5e`. It retains the moving UR5e base spheres and documents omitted supports.

## Shipped robot geometry

`Solver("ur5e")` uses the pinned Universal Robots ROS2 description with its stock nominal
calibration, not calibration measured from a particular robot. Its tool is `tool0`, named by
the fixed joint `flange-tool0`. The URDF, sphere fit, license and mesh hashes ship in `csrc/`;
the source meshes remain upstream. See `csrc/collision/ur5e_provenance.json` for collision
and visual fidelity reports. Reproduce with `scripts/codegen/make_robot_presets.py` and the
pinned checkout stated there (xacro 2.1.1, GRiD 0.6.0, scipy, trimesh and pycollada).
The spherizer always reads collision geometry; `--geometry visual` changes only its report.

Z1 is not yet a shipped collision preset: the conservative fit overlaps two non-adjacent
wrist-link pairs in all sampled configurations. Fetch-with-torso is also not bundled while
the upstream description's noncommercial/share-alike distribution terms are unresolved.
Neither limitation changes support for user-supplied serial prismatic/general-axis models.

# Regenerating the committed header (GRiD codegen workflow)

Install the codegen dependencies first: `python -m pip install -e '.[codegen]'`.
The generated `grid.cuh` and sibling `hjcd_joint_model.h` fix the packaged robot, its named end-effector target and its collision
geometry at build time; nothing at runtime changes the compiled solver. Check `hjcdik.build_info()`
after rebuilding.

## Regenerate `grid.cuh`
```bash
python scripts/codegen/generate_grid.py path/to/robot.urdf -t <ee_target_frame>
```
- `-t` selects a fixed end-effector target frame attached to the final actuated joint
  (e.g. `panda_grasptarget_hand`).
- Output is GRiD's header plus HJCD's joint-axis/type metadata header. Never hand-edit either.
  CMake checks their pairing by hash, including for different robots with the same joint count.
- Robot constants (`NUM_JOINTS`, topology counts) are baked per-URDF; the solver reads them from the generated
  symbols, so don't hardcode sizes.

Then rebuild: `python -m pip install -e .`.

### Source archives without Git metadata

Pass `--glass-revision <full-commit-sha>` to label generated headers from a source
archive, or set `GRID_GLASS_REVISION` (also honored by the header-freshness check).
Use the nested GLASS revision recorded for that source release, not an unrelated
top-level checkout. With Git present, GRiD rejects a label that disagrees with the
nested checkout; without Git it explicitly reports the supplied label as unverified.
This preserves byte-identical generation without weakening freshness checks.

## Isolated native builds

Generate into a separate directory to keep the checkout's default Panda model intact.
For example, to build and test the bundled Fetch arm:

```bash
model_dir=$(mktemp -d /tmp/hjcd-fetch.XXXXXX)
python scripts/codegen/generate_grid.py csrc/urdf/fetch.urdf -t ee_fixed -o "$model_dir/grid.cuh"
cmake -S . -B "$model_dir/build" -DBUILD_PYTHON=OFF \
  -DHJCDIK_GRID_HEADER="$model_dir/grid.cuh" -DHJCDIK_BUILD_NATIVE_TESTS=ON
cmake --build "$model_dir/build" --parallel 2
ctest --test-dir "$model_dir/build" --output-on-failure
```

`HJCDIK_GRID_HEADER` must point to a generated file named `grid.cuh`, with its matching
`hjcd_joint_model.h` in the same directory. Each build directory
selects its own robot; use different directories for different models. Automatic codegen is
an opt-in (`HJCDIK_AUTO_CODEGEN=ON`) restricted to the default Panda workflow; generate
custom headers explicitly. Ordinary builds use the committed header without regeneration.

For a Python wheel, pass the same option through scikit-build-core and select a separate build directory:

```bash
python -m pip wheel . --no-deps -Cbuild-dir="$model_dir/python-build" \
  -Ccmake.define.HJCDIK_GRID_HEADER="$model_dir/grid.cuh" -w "$model_dir/wheels"
```

The wheel contains that one compiled robot. Install it in a separate virtual environment when
comparing robots. Base CUDA architecture defaults to the available GPU; `CUDAARCHS` or
`-DCMAKE_CUDA_ARCHITECTURES=...` can override it for cross-compilation.

## Collision (bring-your-own-URDF)
Add `--collision` to bake GRiD's `grid_collision` spheres (and self-collision ranges) into the same
`grid.cuh` — no hand-written per-robot collision code:
```bash
# spherize the URDF's own collision meshes:
python scripts/codegen/generate_grid.py path/to/robot.urdf -t <ee_target> --collision --collision-res 0.05
# or read a pre-spherized foam URDF (when the URDF's meshes don't resolve on disk, e.g. Panda):
python scripts/codegen/generate_grid.py path/to/robot.urdf -t <ee_target> --collision --spherized-urdf path/to/robot_spherized.urdf
```

## Caveats
- GRiD supports more general robot topologies than HJCD's fixed-base serial-chain solver.
  Codegen rejects floating bases, actuated branches and mimic joints before compilation.
- The generated header does **not** vendor its own copy of GLASS: `generate_grid.py` runs GRiD with
  `vendor_glass=False`, so `grid.cuh` includes the top-level `external/GLASS` (one GLASS per translation
  unit, ~9k fewer generated lines). GRiD's nested GLASS pin must therefore match `external/GLASS`;
  `scripts/setup/bootstrap.sh` initializes both at the committed revisions.
- The default codegen profile emits kinematics and optional collision only. Use `--profile all`
  only when the generated header is also needed by an external dynamics consumer.
- Without `--collision` the build runs open-world; the Python API rejects any `collision` other than `None`
  (the collision path is compiled out via the `HJCD_HAS_COLLISION` sentinel).
- Keep the EE-target frame consistent with `FLANGE_IDX` usage in the kernel.
