# Custom robot (GRiD codegen workflow)

HJCD-IK's kinematics are generated from a URDF by GRiD into `csrc/generated/grid.cuh`.

## Regenerate `grid.cuh`
```bash
python scripts/codegen/generate_grid.py path/to/robot.urdf -t <ee_target_frame>
```
- `-t` selects the fixed end-effector target frame (e.g. `panda_grasptarget_hand`).
- Output is the **stock** generated header — never hand-edit it.
- Robot constants (`NUM_JOINTS`, topology counts) are baked per-URDF; the solver reads them from the generated
  symbols, so don't hardcode sizes.

Then rebuild: `python -m pip install -e .`.

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
- The solver currently assumes revolute/prismatic/fixed joints, no kinematic loops.
- Without `--collision` the build runs open-world; a `collision_free=True` request is ignored
  (the collision path is compiled out via the `HJCD_HAS_COLLISION` sentinel).
- Keep the EE-target frame consistent with `FLANGE_IDX` usage in the kernel.
