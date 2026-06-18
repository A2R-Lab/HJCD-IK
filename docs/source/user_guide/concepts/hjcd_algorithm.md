# The HJCD algorithm

Hybrid Jacobian Coordinate Descent combines cheap per-joint coordinate updates with Jacobian-based refinement:

1. **Coarse search.** From random restarts, greedily pick per-joint moves that most reduce position/orientation
   error (a coordinate-descent sweep, with a two-joint greedy step). Cheap, fast, escapes poor initializations.
2. **Levenberg–Marquardt refine.** For the best coarse candidates, build the 6×N geometric Jacobian from the
   world-frame joint transforms (cross-products), form the normal equations `(JᵀJ + λ·diag)Δq = Jᵀr`, and solve
   them with a warp Cholesky. Dogleg / line-search backtracking adapts `λ`.

The end-effector error is computed as a **quaternion** orientation error plus a Euclidean position error. Each
candidate is handled by one warp, so a block sweeps many candidates concurrently.

See [arXiv:2510.07514](https://arxiv.org/abs/2510.07514) for the full method and evaluation.
