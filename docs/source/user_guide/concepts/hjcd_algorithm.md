# The HJCD-IK algorithm

```{figure} /_static/paper/design.png
:alt: HJCD-IK three-phase pipeline — initialization, polishing, collision filter
:width: 100%

The HJCD-IK pipeline: orientation-aware greedy coordinate-descent initialization (PO-CCD), parallel
Jacobian-based polishing (PJ-IK), and a parallel collision filter.
```

Hybrid Jacobian Coordinate Descent is a three-phase, sampling-based solver that combines cheap per-joint
coordinate updates with Jacobian-based refinement and a GPU collision filter:

1. **Coarse search (PO-CCD).** From hundreds-to-thousands of random restarts, an *orientation-aware* greedy
   coordinate-descent sweep picks the per-joint moves that most reduce position **and** orientation error.
   Cheap, fast, and good at escaping poor initializations — it produces a diverse set of coarse seeds.
2. **Levenberg–Marquardt refine (PJ-IK).** The best coarse candidates are polished: build the 6×N geometric
   Jacobian from the world-frame joint transforms (cross-products), form the normal equations
   `(JᵀJ + λ·diag)Δq = Jᵀr`, and solve them with a warp Cholesky. A trust region + backtracking line search
   (with dogleg / single-coordinate fallbacks) adapts `λ` to sub-millimetre, sub-degree accuracy.
3. **Collision filter.** The refined batch is filtered for feasibility (see below).

The end-effector error is computed as a **quaternion** orientation error plus a Euclidean position error.
See [arXiv:2510.07514](https://arxiv.org/abs/2510.07514) for the full method and evaluation.

## Batched execution & warp-locality

HJCD-IK launches **one CUDA block per IK problem** and processes **one candidate per warp**
(`warp_id = threadIdx.x >> 5`, `lane = threadIdx.x & 31`). This warp-locality is the core performance
contract — a single block sweeps many candidates concurrently:

- Forward kinematics, the Jacobian build, the reductions, and the normal-equations solve are all
  **warp-scoped**, using `__shfl_*_sync` / `__syncwarp` rather than block-wide barriers.
- The `SYNC()` macro selects `__syncwarp()` for single-warp blocks and `__syncthreads()` otherwise.

When refactoring math onto GRiD/GLASS, keep it warp-scoped: use `grid::ee_pose_inner_warp` and the
`glass::warp::` primitives — **not** block-scoped (`glass::`), cooperative-groups (`glass::cgrps::`), or
vendor (`glass::nvidia::`) paths, which lose parallelism or add overhead at these tiny, warp-dispatched
sizes.

## Collision avoidance

With `collision_free=True`, the refined candidates are scored against the environment *after* optimization
(never on the hot solver loop) using GRiD's URDF-driven `grid_collision`. The robot is approximated by
covering spheres baked into `grid.cuh` at codegen (`--collision`); obstacles (spheres / cuboids / cylinders)
come from the problem set. `soft` mode (default) adds a penetration cost that biases selection; `hard` mode
filters colliding candidates with `grid_collision::config_free` (self **and** environment); `both` combines
them (env `HJCD_CC_MODE`).

Because collision geometry is generated from the same URDF as the kinematics, adding a robot needs no
hand-written collision code — see {doc}`../tutorials/custom_robot`. A `grid.cuh` built without `--collision`
runs open-world (a `collision_free=True` request is ignored).
