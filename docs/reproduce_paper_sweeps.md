# Reproducing the HJCD-IK paper sweeps

Regenerates the HJCD-IK columns of the IROS-2026 paper (Tables I–III). Baselines (PyRoki/cuRobo/IKFlow,
TRAC-IK for MMD) are NOT installed here, so only HJCD-IK's own numbers are reproduced. Measured on an
RTX 5090 (the paper used an RTX 4060, so our absolute times are lower).

## Per-robot EE target frame (the key per-robot config)
The end-effector is a **named fixed-joint frame**, robot-specific. GRiD's codegen places it at an
`s_XmatsHom` index that **shifts with DoF**, so `scripts/generate_grid.py` resolves that index from the
generated `end_effector_pose_inner_<target>` epilogue and injects `grid::EE_FIXED_FRAME_IDX` into
`grid.cuh`; `include/hjcd_settings.h` consumes it (never hardcode the index).

| robot | URDF | `-t` target (fixed joint) | resolved `EE_FIXED_FRAME_IDX` |
|---|---|---|---|
| Panda 7-DoF | `panda.urdf` | `panda_grasptarget_hand` | 10 |
| Panda 12-DoF | `panda_ext_12dof.urdf` | `panda_hand_joint` | 14 |
| Panda 18-DoF | `panda_ext_18dof.urdf` | `panda_hand_joint` | 20 |
| Panda 24-DoF | `panda_ext_24dof.urdf` | `panda_grasptarget_hand` | 27 |
| Fetch 7-DoF | `fetch.urdf` | `ee_fixed` (→ `ee_link`, zero offset = arm end) | 7 |

GRiD's `-t` takes a **fixed-joint** name (it reaches that joint's child frame), e.g. Fetch's `ee_fixed`
reaches `ee_link`. Panda's grasptarget has a 10.5 cm TCP offset; Fetch's `ee_link` is at the wrist (zero
offset — this arm URDF has no gripper geometry).

## How to regenerate + run one robot
```bash
python scripts/generate_grid.py include/test_urdf/<robot>.urdf -t <target>   # injects EE_FIXED_FRAME_IDX
bash scripts/rebuild.sh                                                        # ninja + install (NOT ninja alone)
python benchmark/hjcd_ik_bench.py --skip-grid-codegen --batches 1,10,100,1000,2000 \
    --num-targets 100 --num-solutions 1 --csv-out /tmp/<robot>.csv
# collision-free (Panda): add  --collision-free --problems-json tests/mb_problems.json --problem-set bookshelf_thin_panda
# restore Panda afterward: generate_grid.py panda.urdf -t panda_grasptarget_hand && scripts/rebuild.sh
```
(The in-repo `benchmark/hjcd_ik_bench.py` built-in codegen path imports a stale `GRiD.GRiDCodeGenerator`
layout and fails with our integration — always codegen via `scripts/generate_grid.py`.)

## Reproduced results
Machine-specific timing/accuracy results are recorded in a **local, untracked** file (they depend on the
GPU and shouldn't live in the repo): `docs/open-tasks/paper_sweep_results_5090.md` (the `docs/open-tasks/`
folder is gitignored). Regenerate with the commands above on your own hardware.

## Not reproduced (need external deps)
- Baseline comparison columns + Pareto plots (Figs 4–5): need PyRoki / cuRobo / IKFlow / jax.
- Table IV (MMD/MMD²): needs TRAC-IK ground-truth + the MMD computation.
- Hardware (Fig 6): the `realworld` branch + a physical Franka.
