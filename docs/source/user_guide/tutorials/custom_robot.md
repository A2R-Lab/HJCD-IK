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
for the packaged Panda and the `"panda_foam"` preset. Without a sphere model the build is
open-world and `collision` must be `None`.

The solver model is a fixed-base serial chain of 1 to 32 revolute or continuous local-+Z joints plus
fixed frames; prismatic joints, floating bases and branched chains are not supported yet.

# Regenerating the committed header (GRiD codegen workflow)

Install the codegen dependencies first: `python -m pip install -e '.[codegen]'`.
The generated `grid.cuh` fixes the packaged robot, its named end-effector target and its collision
geometry at build time; nothing at runtime changes the compiled solver. Check `hjcdik.build_info()`
after rebuilding.

## Regenerate `grid.cuh`
```bash
python scripts/codegen/generate_grid.py path/to/robot.urdf -t <ee_target_frame>
```
- `-t` selects a fixed end-effector target frame attached to the final actuated joint
  (e.g. `panda_grasptarget_hand`).
- Output is the **stock** generated header — never hand-edit it.
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

`HJCDIK_GRID_HEADER` must point to a generated file named `grid.cuh`. Each build directory
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
- HJCD-IK currently supports fixed-base serial chains with 1–32 independent revolute or continuous
  joints, each rotating around its local +Z axis. Fixed joints may connect the links and tool frame.
  GRiD supports more general robots, but this solver's geometric Jacobian and suffix FK do not;
  codegen rejects floating bases, branches, mimic joints, prismatic joints, and other joint axes.
- The generated header does **not** vendor its own copy of GLASS: `generate_grid.py` runs GRiD with
  `vendor_glass=False`, so `grid.cuh` includes the top-level `external/GLASS` (one GLASS per translation
  unit, ~9k fewer generated lines). GRiD's nested GLASS pin must therefore match `external/GLASS`;
  `scripts/setup/bootstrap.sh` initializes both at the committed revisions.
- The default codegen profile emits kinematics and optional collision only. Use `--profile all`
  only when the generated header is also needed by an external dynamics consumer.
- Without `--collision` the build runs open-world; the Python API rejects any `collision` other than `None`
  (the collision path is compiled out via the `HJCD_HAS_COLLISION` sentinel).
- Keep the EE-target frame consistent with `FLANGE_IDX` usage in the kernel.
