# Batched execution & warp-locality

HJCD-IK launches **one CUDA block per IK problem** and processes **one candidate per warp**
(`warp_id = threadIdx.x >> 5`, `lane = threadIdx.x & 31`). This warp-locality is the core performance contract:

- Forward kinematics, the Jacobian build, the reductions, and the normal-equations solve are all **warp-scoped**,
  using `__shfl_*_sync` / `__syncwarp` rather than block-wide barriers.
- The `SYNC()` macro selects `__syncwarp()` for single-warp blocks and `__syncthreads()` otherwise.

When refactoring math onto GRiD/GLASS, keep it warp-scoped: use `grid::ee_pose_inner_warp` and the
`glass::warp::` primitives — **not** block-scoped (`glass::`), cooperative-groups (`glass::cgrps::`), or vendor
(`glass::nvidia::`) paths, which lose parallelism or add overhead at these tiny, warp-dispatched sizes.
