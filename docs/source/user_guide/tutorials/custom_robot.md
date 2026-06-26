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

## Caveats
- The solver currently assumes revolute/prismatic/fixed joints, no kinematic loops.
- **Collision** geometry is per-robot and only provided for Panda/Fetch. A new robot needs its own collision
  spheres in `csrc/robots/` (or run with `collision_free=False`).
- Keep the EE-target frame consistent with `FLANGE_IDX` usage in the kernel.
