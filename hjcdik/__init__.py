import os

if os.name == "nt":
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        cuda_bin = os.path.join(cuda_path, "bin")
        if os.path.isdir(cuda_bin):
            os.add_dll_directory(cuda_bin)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_bin = os.path.join(conda_prefix, "Library", "bin")
        if os.path.isdir(conda_bin):
            os.add_dll_directory(conda_bin)

import threading as _threading

import numpy as _np

from ._hjcdik import (
    Workspace as _Workspace,
    generate_solutions,
    sample_targets,
    num_joints,
    num_frames,
    num_targets,
    joint_limits,
    link_transforms,
    target_transforms,
    target_metadata,
    _target_residuals_raw,
    _normal_equations_raw,
    _lm_refine_raw,
    _coarse_search_raw,
    _solve_problems_raw,
    collision_free,
    _incremental_probe_raw,
    _bench_fk_raw,
)

__all__ = [
    "generate_solutions", "sample_targets", "num_joints", "num_frames", "num_targets", "joint_limits",
    "link_transforms", "target_transforms", "target_metadata", "target_residuals",
    "normal_equations", "refine", "coarse_search", "incremental_probe", "bench_fk",
    "pack_active_mask", "collision_free", "solve", "solve_problems", "HJCDSolver",
]

# The generated target order is FIXED and is the order of every [.., K, ..] axis in this API,
# in every result array, and in the benchmarks:
#     0 = left hand    1 = right hand    2 = left foot    3 = right foot
# (For a single-target robot such as Panda, K == 1 and index 0 is its one generated target.)
# Names for index k: target_metadata() -> see csrc/generated/hjcd_targets.json.


def pack_active_mask(mask, B, K):
    """Canonicalize an active-target mask to a [B] uint32 bitmask (bit k = target k active).

    Accepts:
      None            -> all K targets active in every problem
      [B] integer     -> already-packed bitmasks (validated)
      [K] bool        -> the same target subset for every problem
      [B, K] bool     -> per-problem subsets
    Packing always happens HERE, on the host; CUDA only ever sees the [B] uint32 form.
    """
    if mask is None:
        packed = _np.full(B, (1 << K) - 1, dtype=_np.uint32)
    else:
        a = _np.asarray(mask)
        if a.dtype == bool:
            if a.ndim == 1 and a.shape[0] == K:
                a = _np.broadcast_to(a, (B, K))
            elif a.ndim != 2 or a.shape != (B, K):
                raise ValueError(f"boolean active_target_mask must be [K]={K} or [B,K]={(B, K)}, "
                                 f"got {a.shape}")
            bits = (1 << _np.arange(K, dtype=_np.uint32))
            packed = (a.astype(_np.uint32) * bits).sum(axis=1).astype(_np.uint32)
        elif _np.issubdtype(a.dtype, _np.integer):
            if a.shape != (B,):
                raise ValueError(f"integer active_target_mask must be [B]={(B,)}, got {a.shape}")
            packed = a.astype(_np.uint32)
        else:
            raise TypeError(f"active_target_mask must be bool or integer, got {a.dtype}")

    valid = (1 << K) - 1
    if _np.any(packed & ~_np.uint32(valid)):
        bad = int(_np.argmax(packed & ~_np.uint32(valid)))
        raise ValueError(f"active_target_mask[{bad}] = 0x{packed[bad]:x} sets bits above "
                         f"K={K} (valid bits 0x{valid:x})")
    if _np.any(packed == 0):
        bad = int(_np.argmax(packed == 0))
        raise ValueError(f"active_target_mask[{bad}] is empty: a problem must activate at least "
                         f"one target")
    return _np.ascontiguousarray(packed, dtype=_np.uint32)


def _bcast_weights(w, B, K, name):
    """scalar | [K] | [B,K]  ->  contiguous [B,K] float64. Rejects negative / NaN / inf."""
    a = _np.asarray(w, dtype=_np.float64)
    if a.ndim == 0:
        a = _np.full((B, K), float(a))
    elif a.ndim == 1 and a.shape[0] == K:
        a = _np.broadcast_to(a, (B, K))
    elif a.ndim == 2 and a.shape == (B, K):
        pass
    else:
        raise ValueError(f"{name} must be scalar, [K]={K}, or [B,K]={(B, K)}; got shape {a.shape}")
    if not _np.all(_np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or inf")
    if _np.any(a < 0):
        raise ValueError(f"{name} contains negative values")
    return _np.ascontiguousarray(a, dtype=_np.float64)


def _bcast_tol(t, K, name):
    """scalar | [K] -> contiguous [K] float64, strictly positive."""
    a = _np.asarray(t, dtype=_np.float64)
    if a.ndim == 0:
        a = _np.full(K, float(a))
    elif a.ndim != 1 or a.shape[0] != K:
        raise ValueError(f"{name} must be scalar or [K]={K}; got shape {a.shape}")
    if not _np.all(_np.isfinite(a)) or _np.any(a <= 0):
        raise ValueError(f"{name} must be finite and > 0")
    return _np.ascontiguousarray(a, dtype=_np.float64)


# DEFAULT PRECISION IS "float32". It is 7.8-8.1x faster end-to-end than float64 with no measurable
# task-space accuracy loss (terminal errors ~1.5x inside the tolerances), no top-1 success
# regression, and identical collision behaviour. "float64" stays available for debugging, unusually
# tight tolerances, and numerical comparison.
_PRECISIONS = ("float64", "float32")
DEFAULT_PRECISION = "float32"

# ADAPTIVE STOPPING (Policy B) IS ON BY DEFAULT.
#
# A seed stops as soon as it either (a) satisfies every active tolerance -- that exit was always in
# the kernel -- or (b) makes negligible progress in E_phys for `stag_patience` consecutive
# iterations. E_phys is a TOLERANCE-NORMALISED PHYSICAL error, not the row-scaled cost: the row
# scales are re-frozen every iteration, so consecutive scaled costs are in different units and a
# relative improvement between them is not a real quantity.
#
# Measured on G1 K=4, B=2000: LM device time 65.1 -> 13.6 ms with top-1 unchanged at 100%, the same
# selected errors, and 98% of the solved-seed pool retained. A fixed lm=10 cap reaches similar speed
# but keeps only 27% of the pool -- it kills every seed at 10, including ones that would have solved
# at iteration 18 or 42. Policy B kills only the STALLED ones.
#
# stag_patience = 0 is the explicit opt-out and exactly restores the old fixed-cap behaviour.
# The hard caps (coarse_iters, lm_iters) still bound the worst case.
DEFAULT_STAG_PATIENCE = 2
DEFAULT_STAG_REL = 1e-3


def _precision_code(precision):
    """'float64' -> 0, 'float32' -> 1. Explicit and validated; NEVER a silent fallback to double.

    The value selects the GPU COMPUTE TYPE of the coarse and LM kernels (the template argument T),
    not merely the numpy dtype of the arrays you hand in.
    """
    if precision not in _PRECISIONS:
        raise ValueError(
            f"precision must be one of {_PRECISIONS!r}, got {precision!r}. "
            f"There is no automatic fallback: an unsupported value is an error, not a default.")
    return _PRECISIONS.index(precision)


def _split_precision(precision, coarse_precision, lm_precision):
    """`precision` sets both stages; coarse_precision / lm_precision override per stage.

    The split form exists for the Phase-0B ablation (fp64 coarse + fp32 LM). The normal public API
    is just precision=.
    """
    cp = coarse_precision if coarse_precision is not None else precision
    lp = lm_precision if lm_precision is not None else precision
    return _precision_code(cp), _precision_code(lp)


def target_residuals(q,
                     target_positions,
                     target_quaternions,
                     active_target_mask=None,
                     position_weights=1.0,
                     orientation_weights=1.0,
                     position_tol=1e-3,
                     orientation_tol=1e-3):
    """Per-target residuals, costs and success for a batch of configurations.

    Diagnostic / reference path: it runs one full GRiD FK, composes the K generated target frames,
    and evaluates the residual layer. It does NOT solve anything.

    Target index order is the generated order (0 left hand, 1 right hand, 2 left foot, 3 right foot).

    Args:
        q                    [B, N]     float64 joint configurations.
        target_positions     [B, K, 3]  float64 desired positions, world frame, metres.
        target_quaternions   [B, K, 4]  float64 desired orientations, WXYZ. Normalized here; a
                                        quaternion and its negation give identical residuals.
        active_target_mask   None | [B] int | [K] bool | [B, K] bool. Packed to a [B] uint32 bitmask
                                        before reaching CUDA. An empty mask is rejected.
        position_weights     scalar | [K] | [B, K]   >= 0, finite.
        orientation_weights  scalar | [K] | [B, K]   >= 0, finite.
        position_tol         scalar | [K]  success threshold on |e_p| (metres), > 0.
        orientation_tol      scalar | [K]  success threshold on |e_R| (radians), > 0.

    An ACTIVE target with both weights zero is rejected (it would contribute nothing yet be
    required to succeed).

    Returns dict:
        position_residuals    [B, K, 3]  e_p = p* - p           (unweighted, world)
        orientation_residuals [B, K, 3]  e_R = Log(R* R^T)      (unweighted, world)
        position_errors       [B, K]     |e_p|
        orientation_errors    [B, K]     |e_R|
        target_costs          [B, K]     w_p|e_p|^2 + w_R|e_R|^2
        cost_raw              [B]        sum of target_costs over ACTIVE targets
        cost_normalized       [B]        cost_raw / (sum_active(w_p + w_R) + 1e-12)   [reporting only]
        target_success        [B, K]     bool; False for inactive targets (not evaluated)
        success               [B]        bool; all active targets succeeded
        active_target_mask    [B]        uint32, as sent to CUDA

    Inactive targets contribute exactly zero residual, norm and cost.
    """
    N, K = num_joints(), num_targets()

    q = _np.ascontiguousarray(_np.asarray(q, dtype=_np.float64))
    if q.ndim != 2 or q.shape[1] != N:
        raise ValueError(f"q must be [B, {N}], got {q.shape}")
    B = q.shape[0]
    if B == 0:
        raise ValueError("q must contain at least one configuration")

    p = _np.asarray(target_positions, dtype=_np.float64)
    if p.shape != (B, K, 3):
        raise ValueError(f"target_positions must be [B,K,3] = {(B, K, 3)}, got {p.shape}")
    if not _np.all(_np.isfinite(p)):
        raise ValueError("target_positions contains NaN or inf")

    quat = _np.asarray(target_quaternions, dtype=_np.float64)
    if quat.shape != (B, K, 4):
        raise ValueError(f"target_quaternions must be [B,K,4] (wxyz) = {(B, K, 4)}, "
                         f"got {quat.shape}")
    if not _np.all(_np.isfinite(quat)):
        raise ValueError("target_quaternions contains NaN or inf")

    packed = pack_active_mask(active_target_mask, B, K)
    wp = _bcast_weights(position_weights, B, K, "position_weights")
    wo = _bcast_weights(orientation_weights, B, K, "orientation_weights")
    ep = _bcast_tol(position_tol, K, "position_tol")
    eo = _bcast_tol(orientation_tol, K, "orientation_tol")

    # An active target with no weight at all is a spec error, not a silent no-op.
    act = ((packed[:, None] >> _np.arange(K, dtype=_np.uint32)) & 1).astype(bool)
    dead = act & (wp == 0) & (wo == 0)
    if _np.any(dead):
        b, k = map(int, _np.argwhere(dead)[0])
        raise ValueError(f"target {k} is active in problem {b} but has both position_weights and "
                         f"orientation_weights == 0")

    # Quaternion normalization is defined HERE: unit-normalize on the host; the device does not
    # renormalize the target. Sign is left alone -- the device flips into the target's hemisphere,
    # so q and -q are identical inputs.
    nrm = _np.linalg.norm(quat, axis=-1, keepdims=True)
    if _np.any(nrm[act] < 1e-8):
        raise ValueError("target_quaternions contains a (near-)zero quaternion for an active target")
    quat = _np.where(nrm > 0, quat / _np.where(nrm > 0, nrm, 1.0), quat)

    return _target_residuals_raw(
        q, _np.ascontiguousarray(p), _np.ascontiguousarray(quat), packed, wp, wo, ep, eo)


def _canonical_problem(q, p, quat, mask, wp, wo, dtype=_np.float64, _canonical=False):
    """Shared validation/broadcast/packing for the residual, normal-equation and refine paths.

    `dtype` is the wire dtype handed to CUDA. For an fp32 solve it is float32, and the numpy buffers
    are then copied STRAIGHT to the device -- no host narrowing loop anywhere. float64 stays fully
    supported as the compatibility path (narrowed once, inside the launcher).
    """
    if _canonical:
        # solve() has already validated, broadcast, packed and cast these. Repeating the whole pass
        # inside coarse_search() and refine() cost ~0.6 ms EACH at B=2000 -- three passes per solve.
        return q, p, quat, mask, wp, wo
    N, K = num_joints(), num_targets()
    q = _np.ascontiguousarray(_np.asarray(q, dtype=dtype))
    if q.ndim != 2 or q.shape[1] != N:
        raise ValueError(f"q must be [B, {N}], got {q.shape}")
    B = q.shape[0]
    p = _np.asarray(p, dtype=dtype)
    quat = _np.asarray(quat, dtype=dtype)
    if p.shape != (B, K, 3):
        raise ValueError(f"target_positions must be {(B, K, 3)}, got {p.shape}")
    if quat.shape != (B, K, 4):
        raise ValueError(f"target_quaternions must be {(B, K, 4)} (wxyz), got {quat.shape}")
    if not (_np.all(_np.isfinite(p)) and _np.all(_np.isfinite(quat))):
        raise ValueError("target_positions / target_quaternions contain NaN or inf")
    packed = pack_active_mask(mask, B, K)
    wpa = _bcast_weights(wp, B, K, "position_weights").astype(dtype, copy=False)
    woa = _bcast_weights(wo, B, K, "orientation_weights").astype(dtype, copy=False)
    act = ((packed[:, None] >> _np.arange(K, dtype=_np.uint32)) & 1).astype(bool)
    dead = act & (wpa == 0) & (woa == 0)
    if _np.any(dead):
        b, k = map(int, _np.argwhere(dead)[0])
        raise ValueError(f"target {k} is active in problem {b} but has both weights == 0")
    nrm = _np.linalg.norm(quat, axis=-1, keepdims=True)
    if _np.any(nrm[act] < 1e-8):
        raise ValueError("target_quaternions contains a (near-)zero quaternion for an active target")
    quat = _np.where(nrm > 0, quat / _np.where(nrm > 0, nrm, 1.0), quat).astype(dtype, copy=False)
    return (q, _np.ascontiguousarray(p, dtype=dtype), _np.ascontiguousarray(quat, dtype=dtype),
            packed, _np.ascontiguousarray(wpa, dtype=dtype), _np.ascontiguousarray(woa, dtype=dtype))



class HJCDSolver:
    """A solver instance that OWNS its CUDA workspace.

    Why this exists: every solve used to cudaMalloc ~10 device buffers, free them, and copy the
    B x N configuration four times on the way out. At B=2000 that marshalling was ~5.2 ms of a
    ~20 ms solve. A solver keeps a capacity-based device arena alive across calls, so after warm-up
    a steady stream of same-or-smaller solves performs ZERO cudaMalloc and ZERO cudaFree.

    OWNERSHIP     The workspace is owned by this object and freed when it is garbage-collected.
    DEVICE        Bound to the CUDA device current at construction.
    STREAM        The default stream. There is no per-solver stream yet.
    THREAD SAFETY NOT thread-safe, and deliberately not silently so: exactly ONE call may be active
                  per instance. A concurrent or re-entrant call raises RuntimeError rather than
                  racing on the shared arena. Use one solver per thread.
    GROWTH        Capacity grows geometrically and never shrinks. A smaller batch reuses a larger
                  workspace with no allocation; a larger batch triggers exactly one growth.

    The free functions (hjcdik.solve/refine/coarse_search) wrap a THREAD-LOCAL solver, so they stay
    allocation-free too without ever sharing a workspace across threads.
    """

    def __init__(self):
        self._ws = _Workspace()
        self._busy = False

    def workspace_stats(self):
        """cuda_mallocs / cuda_frees / bytes / capacity_B / device."""
        return self._ws.stats()

    def _enter(self):
        if self._busy:
            raise RuntimeError(
                "HJCDSolver is not thread-safe and does not support re-entrant or concurrent calls: "
                "one call may be active per instance. Use a separate HJCDSolver per thread.")
        self._busy = True

    def _exit(self):
        self._busy = False

    def solve(self, *a, **kw):
        return solve(*a, _solver=self, **kw)

    def solve_problems(self, *a, **kw):
        return solve_problems(*a, _solver=self, **kw)

    def refine(self, *a, **kw):
        return refine(*a, _solver=self, **kw)

    def coarse_search(self, *a, **kw):
        return coarse_search(*a, _solver=self, **kw)


_tls = _threading.local()


def _default_solver():
    """Thread-local solver behind the free-function API. Never shared between threads, so the
    free functions are allocation-free AND race-free."""
    s = getattr(_tls, "solver", None)
    if s is None:
        s = HJCDSolver()
        _tls.solver = s
    return s


def normal_equations(q, target_positions, target_quaternions, active_target_mask=None,
                     position_weights=1.0, orientation_weights=1.0):
    """The accumulated LM normal equations A = sum_k Jk^T Wk Jk, b = sum_k Jk^T Wk e_k.

    Reference/diagnostic path -- it runs the SAME device accumulation the LM uses. Returns
    {"A": [B, N, N], "b": [B, N]}. Undamped (no lambda applied).
    """
    args = _canonical_problem(q, target_positions, target_quaternions, active_target_mask,
                              position_weights, orientation_weights)
    return _normal_equations_raw(*args)


def refine(q, target_positions, target_quaternions, active_target_mask=None,
           position_weights=1.0, orientation_weights=1.0,
           position_tol=1e-4, orientation_tol=1e-3, lambda_init=5e-3, max_iters=40,
           diagnostics=False, return_trace=False,
           precision="float32",
           stag_patience=2, stag_rel=1e-3, _solver=None, _canonical=False, _seeds_per_problem=1):
    """Multi-target Levenberg-Marquardt refinement of seed configurations. LM only, no coarse search.

    Solves (J^T W J + lambda diag(A)) dq = J^T W e per iteration, accumulating A and b target by
    target (never forming the stacked 6K x N Jacobian). Target index order is the generated order:
    0 left hand, 1 right hand, 2 left foot, 3 right foot.

    Args are as target_residuals(); q is [B, N] SEEDS (refined in the returned array, not in place).

    Returns dict:
        joint_config        [B, N]  refined configs (best cost seen, not merely the last iterate)
        position_errors     [B, K]  metres; 0 for inactive targets
        orientation_errors  [B, K]  radians; 0 for inactive
        cost                [B]     raw weighted total cost
        success             [B]     every active weighted component within tolerance
        active_target_mask  [B]     uint32

    LM diagnostics (only when diagnostics=True; otherwise these keys are ABSENT, no trace buffer is
    allocated, no trace store is executed, and the solve path is bit-for-bit the fast path). They are
    all DERIVED FROM THE TRACE, which is the authoritative source:
        lm_iterations       [B]     valid trace rows = outer LM linearizations (Jacobian rebuilds).
                                    0 when the seed was already converged -- no linearization ran.
        lm_trials           [B]     cumulative damped linear systems solved
        line_searches       [B]     cumulative backtracking step lengths evaluated
        accepted_lm_steps   [B]     cumulative accepted steps
        rejected_lm_steps   [B]     lm_iterations - accepted_lm_steps
        iterations          [B]     alias of lm_iterations (back-compat)
        trace               [B, max_iters, 10]  only when return_trace=True. Columns:
                                    0 valid, 1 it, 2 lm_trials, 3 accepted(this), 4 accepted(cum),
                                    5 cost, 6 max_pos_err, 7 max_ori_err, 8 lambda, 9 line_searches
                                    Row validity is column 0 -- NEVER inferred from cost or lambda.
    """
    pc = _precision_code(precision)
    wire = _np.float32 if pc == 1 else _np.float64
    qa, p, quat, packed, wp, wo = _canonical_problem(
        q, target_positions, target_quaternions, active_target_mask,
        position_weights, orientation_weights, dtype=wire, _canonical=_canonical)
    if max_iters <= 0:
        raise ValueError("max_iters must be > 0")
    for nm, v in (("position_tol", position_tol), ("orientation_tol", orientation_tol),
                  ("lambda_init", lambda_init)):
        if not _np.isfinite(v) or v <= 0:
            raise ValueError(f"{nm} must be finite and > 0")
    if return_trace and not diagnostics:
        raise ValueError("return_trace=True requires diagnostics=True")
    sv = _solver or _default_solver()
    sv._enter()
    try:
        return _lm_refine_raw(qa, p, quat, packed, wp, wo,
                              float(position_tol), float(orientation_tol),
                              float(lambda_init), int(max_iters),
                              bool(diagnostics), bool(return_trace), pc,
                              int(stag_patience), float(stag_rel), sv._ws,
                              int(_seeds_per_problem))
    finally:
        sv._exit()


def incremental_probe(q, updates, accept, target_positions, target_quaternions,
                      active_target_mask=None, position_weights=1.0, orientation_weights=1.0):
    """Run a SEQUENCE of coordinate updates through the incremental (subtree) FK path.

    Each step j <- v is applied in place: only joint j's subtree world transforms are recomputed,
    only the targets in JOINT_TARGET_MASK[j] & active are recomposed and rescored, and the total cost
    is folded incrementally. A step with accept=False is rolled back (joint restored, subtree
    recomputed, cached target state restored bitwise).

    This is the machinery Phase 5's coarse search will call. It is exposed so it can be validated
    against a fresh full FK independently of any optimizer.

    Args:
        q        [B, N]      starting configurations
        updates  [B, M, 2]   (joint_index, new_value) per step, applied in order
        accept   [B, M] bool True = keep the step, False = roll back
        (targets/mask/weights as in target_residuals)

    Returns dict: joint_config [B,N], joint_transforms [B,N,4,4], target_transforms [B,K,4,4],
    position_residuals, orientation_residuals, position_errors, orientation_errors, target_costs,
    cost_raw [B].
    """
    qa, p, quat, packed, wp, wo = _canonical_problem(
        q, target_positions, target_quaternions, active_target_mask,
        position_weights, orientation_weights)
    B, N = qa.shape
    u = _np.asarray(updates, dtype=_np.float64)
    if u.ndim != 3 or u.shape[0] != B or u.shape[2] != 2:
        raise ValueError(f"updates must be [B, M, 2], got {u.shape}")
    M = u.shape[1]
    uj = _np.ascontiguousarray(u[..., 0].astype(_np.int32))
    uv = _np.ascontiguousarray(u[..., 1])
    if M and (uj.min() < 0 or uj.max() >= N):
        raise ValueError(f"update joint index out of range [0, {N})")
    ac = _np.ascontiguousarray(_np.asarray(accept, dtype=bool))
    if ac.shape != (B, M):
        raise ValueError(f"accept must be [B, M] = {(B, M)}, got {ac.shape}")
    return _incremental_probe_raw(qa, uj, uv, ac, p, quat, packed, wp, wo)


def bench_fk(q, joint, iters, mode, target_positions, target_quaternions,
             active_target_mask=None, position_weights=1.0, orientation_weights=1.0):
    """Time `iters` coordinate updates on `joint`. mode=0 full FK path, mode=1 incremental. -> ms."""
    qa, p, quat, packed, wp, wo = _canonical_problem(
        q, target_positions, target_quaternions, active_target_mask,
        position_weights, orientation_weights)
    return _bench_fk_raw(qa, int(joint), int(iters), int(mode), p, quat, packed, wp, wo)


def coarse_search(q, target_positions, target_quaternions, active_target_mask=None,
                  position_weights=1.0, orientation_weights=1.0,
                  position_tol=1e-3, orientation_tol=1e-2,
                  lambda_coord=1e-6, h_min=1e-9, max_step=0.35,
                  max_iters=60, stall_lim=5, use_incremental=True, seed=0,
                  diagnostics=False, return_trace=False,
                  problems_json_text="", problem_set_name="", problem_idx=0,
                  max_pert_attempts=4, precision="float32", _solver=None, _canonical=False,
                  _seeds_per_problem=1):
    """Multi-target coarse search: aggregate weighted coordinate Gauss-Newton.

    Per outer iteration, every joint lane forms ONE aggregate scalar proposal

        delta_j = g_j / (h_j + lambda_coord)

        g_j = sum_{k in A_j} ( Jv_kj^T W_p,k e_p,k + Jw_kj^T W_R,k e_R,k )
        h_j = sum_{k in A_j} ( Jv_kj^T W_p,k Jv_kj + Jw_kj^T W_R,k Jw_kj )

    with A_j = JOINT_TARGET_MASK[j] & active_target_mask, and the Phase-3B row scaling
    W_{k,r} = w_{k,r} * s_{k,r}^2, s_{k,r} = 1/(||J_{k,r}|| + eps), FROZEN once per iteration and
    used for BOTH proposal ranking and the exact trial cost. de/dq = -J, so g_j carries a plus sign.

    The step is clipped to `max_step` and to the joint limits; its linearized predicted improvement
    is  pred_j = 2*delta_j*g_j - delta_j^2*h_j.  A joint with no affected target, or with curvature
    h_j <= h_min, emits an invalid proposal. A warp-wide reduction selects the single best proposal,
    and ONLY that one is evaluated exactly (subtree FK -> affected targets -> exact cost); it is
    accepted only if the exact cost improves, otherwise the Phase-4 rollback restores the state.

    use_incremental=False swaps the Phase-4 subtree FK for a full FK + full rescore (ablation only;
    same answer, more work).

    STALL PERTURBATIONS ARE COLLISION-GATED. A kick is exploratory -- it does NOT have to reduce the
    task cost to become the current state -- but it must satisfy every enabled hard constraint. On a
    stall the state is saved, a kick applied, FK/targets/residuals/costs fully refreshed, and the
    exact same grid_collision::config_free gate the proposals use is run. A colliding kick is
    rejected and the saved state restored exactly; up to `max_pert_attempts` kicks are tried,
    stopping at the first feasible one. If all of them collide the original state is restored, the
    stall counter is reset anyway (leaving it tripped would retry the kick every iteration and spin)
    and the event is counted as `coarse_perturbations_exhausted`. best_x -- and hence the returned
    configuration -- is only ever copied from a feasible state, the seed included.

    Returns joint_config, position_errors, orientation_errors, cost, success, active_target_mask.
    With diagnostics=True it also returns coarse_iterations, accepted_coarse_steps,
    rejected_coarse_steps, coarse_perturbations (kicks RETAINED), coarse_max_stall,
    coarse_perturbation_events, coarse_perturbation_attempts, coarse_perturbations_rejected,
    coarse_perturbations_exhausted -- ALL derived from the trace -- and with return_trace=True the
    trace itself, [B, max_iters, 13]:
        0 valid, 1 it, 2 joint, 3 delta, 4 predicted improvement,
        5 cost_before, 6 cost_after, 7 accepted, 8 stall, 9 perturbed(retained),
        10 pert_attempts, 11 pert_collision_rejects, 12 pert_exhausted
    """
    pc = _precision_code(precision)
    wire = _np.float32 if pc == 1 else _np.float64
    qa, p, quat, packed, wp, wo = _canonical_problem(
        q, target_positions, target_quaternions, active_target_mask,
        position_weights, orientation_weights, dtype=wire, _canonical=_canonical)
    if return_trace and not diagnostics:
        raise ValueError("return_trace=True requires diagnostics=True")
    for nm, v in (("position_tol", position_tol), ("orientation_tol", orientation_tol),
                  ("max_step", max_step)):
        if not _np.isfinite(v) or v <= 0:
            raise ValueError(f"{nm} must be finite and > 0")
    if max_iters <= 0:
        raise ValueError("max_iters must be > 0")
    sv = _solver or _default_solver()
    sv._enter()
    try:
        return _coarse_search_raw(qa, p, quat, packed, wp, wo,
                                  float(position_tol), float(orientation_tol),
                                  float(lambda_coord), float(h_min), float(max_step),
                                  int(max_iters), int(stall_lim), int(bool(use_incremental)),
                                  int(seed), bool(diagnostics), bool(return_trace),
                                  str(problems_json_text), str(problem_set_name), int(problem_idx),
                                  int(max_pert_attempts), pc, sv._ws, int(_seeds_per_problem))
    finally:
        sv._exit()


def solve(q, target_positions, target_quaternions, active_target_mask=None,
          position_weights=1.0, orientation_weights=1.0,
          position_tol=1e-4, orientation_tol=1e-3,
          coarse_mode="auto", coarse_iters=60, coarse_incremental=True,
          lm_iters=60, seed=0, diagnostics=False,
          problems_json_text="", problem_set_name="", problem_idx=0,
          precision="float32", coarse_precision=None, lm_precision=None,
          stag_patience=2, stag_rel=1e-3, _solver=None):
    """Multi-target end-to-end solve: auto-dispatched coarse stage -> multi-target LM.

    DISPATCH is on the number of ACTIVE TARGET BITS -- popcount(active_target_mask) -- never on the
    generated target count:

        coarse_mode="auto"          popcount == 1  -> LM only
                                    popcount >= 2  -> new multi-target coarse -> LM
        coarse_mode="none"          force LM only
        coarse_mode="multi_target"  force the new coarse search
        coarse_mode="legacy"        NOT available here (single-target generate_solutions only)

    Rationale, measured: on Panda K=1 the coarse search WORSENS terminal accuracy and costs time
    (0.000200 mm legacy-coarse vs 0.000123 mm LM-only); on G1 K=4 LM alone converges 0% from random
    restarts while coarse -> LM converges 100%.

    A batch may mix active masks. This is the simplest correct implementation: the coarse kernel runs
    on the WHOLE batch whenever ANY problem needs it, and LM-only problems are left untouched by it
    (their per-problem dispatch is honoured by skipping their coarse result). Partitioning the batch
    into popcount==1 and popcount>=2 groups is a profiling-driven optimization, not done yet -- see
    the divergence note in the Phase-5 report.

    With diagnostics=True the returned dict carries the combined counters. For an LM-only problem
    every coarse counter is exactly 0.

    Collision: pass problems_json_text/problem_set_name/problem_idx to gate the coarse search on
    exact collision-freedom of each winning proposal.
    """
    qa, p, quat, packed, wp, wo = _canonical_problem(
        q, target_positions, target_quaternions, active_target_mask,
        position_weights, orientation_weights,
        dtype=_np.float32 if _precision_code(precision) == 1 else _np.float64)
    if coarse_mode not in ("auto", "none", "multi_target"):
        raise ValueError(f"coarse_mode must be auto|none|multi_target, got '{coarse_mode}'")
    B, K = qa.shape[0], num_targets()

    popc = _np.array([bin(int(m)).count("1") for m in packed])
    if coarse_mode == "none":
        use_coarse = _np.zeros(B, dtype=bool)
    elif coarse_mode == "multi_target":
        use_coarse = _np.ones(B, dtype=bool)
    else:                                              # auto: popcount >= 2
        use_coarse = popc >= 2

    # coarse_iters == 0 means LM-ONLY: no coarse stage, and no empty kernel launched for it.
    if coarse_iters <= 0:
        use_coarse = _np.zeros(B, dtype=bool)

    cp_code, lp_code = _split_precision(precision, coarse_precision, lm_precision)
    cp, lp = _PRECISIONS[cp_code], _PRECISIONS[lp_code]

    cc_enabled = bool(problems_json_text) and bool(problem_set_name)

    # The coarse stage also runs whenever collision is enabled, EVEN IF the dispatch would not
    # otherwise call for it (a single active target goes LM-only). It is the only stage with a hard
    # collision gate, so it is the only source of a guaranteed-feasible fallback for a seed whose LM
    # result ends up colliding. Its output is used ONLY as that fallback here -- the LM is still
    # seeded exactly as before (`use_coarse` alone decides that), so LM-only accuracy is unchanged.
    # ...but with collision on, the coarse stage is still needed to manufacture a feasible fallback,
    # so coarse_iters == 0 is only honoured as "no coarse launch" in the open-world case.
    run_coarse = bool(use_coarse.any()) or cc_enabled

    seeds = qa
    cdiag = None
    c = None
    if run_coarse:
        c = coarse_search(qa, p, quat, active_target_mask=packed,
                          position_weights=wp, orientation_weights=wo,
                          position_tol=position_tol, orientation_tol=orientation_tol,
                          max_iters=coarse_iters, use_incremental=coarse_incremental, seed=seed,
                          diagnostics=diagnostics, return_trace=False,
                          problems_json_text=problems_json_text,
                          problem_set_name=problem_set_name, problem_idx=problem_idx,
                          precision=cp, _solver=_solver, _canonical=True)
        # Same wire dtype throughout: no widen/narrow round trip between the stages.
        cq = _np.ascontiguousarray(c["joint_config"], dtype=qa.dtype)
        seeds = cq if use_coarse.all() else _np.ascontiguousarray(
            _np.where(use_coarse[:, None], cq, qa))
        cdiag = c if diagnostics else None

    lm = refine(seeds, p, quat, active_target_mask=packed,
                position_weights=wp, orientation_weights=wo,
                position_tol=position_tol, orientation_tol=orientation_tol,
                max_iters=lm_iters, diagnostics=diagnostics, precision=lp,
                stag_patience=stag_patience, stag_rel=stag_rel, _solver=_solver, _canonical=True)

    out = dict(lm)
    out["seeds_after_coarse"] = seeds
    out["used_coarse"] = use_coarse
    out["collision_enabled"] = cc_enabled
    # Device stage times, CUDA-event measured, FROM THIS SAME INVOCATION. Never mix these with
    # independently-measured medians -- that is what produced the impossible
    # "coarse + LM > end-to-end" rows in the Phase-0B report.
    out["coarse_kernel_ms"] = float(c["kernel_ms"]) if c is not None else 0.0
    out["lm_kernel_ms"] = float(lm["kernel_ms"])

    # -------------------------------------------------------------------------------------------
    # FINAL COLLISION FILTER (only when collision is enabled -- otherwise not a single extra kernel).
    #
    # The coarse stage is hard-gated, so its answer is feasible. The LM is NOT gated: it happily
    # refines a feasible coarse state straight back into the shelf. Measured on bookshelf_small_panda
    # with verified-free seeds: coarse out 0/1000 colliding, LM out 63/1000. So the LM's task-optimal
    # answer must be VALIDATED, not trusted.
    #
    # Per seed: keep the LM result if it is collision-free; otherwise fall back to that seed's own
    # collision-free coarse result. A colliding candidate is never returned, and -- because a
    # fallback carries its COARSE task cost, which is worse -- a better-task-but-colliding LM
    # candidate can never out-rank a feasible one under the existing argmin(cost) ranking. The
    # ranking rule itself is untouched; only the candidate set is filtered.
    # -------------------------------------------------------------------------------------------
    if cc_enabled:
        import time as _time
        _t0 = _time.perf_counter()

        lm_q = _np.asarray(lm["joint_config"], dtype=_np.float64)
        lm_free = collision_free(lm_q, problems_json_text, problem_set_name, problem_idx)

        coarse_q = _np.asarray(c["joint_config"], dtype=_np.float64)
        # Only the seeds whose LM collided need a fallback, so only those get checked.
        need = ~lm_free
        coarse_ok = _np.zeros(B, dtype=bool)
        if need.any():
            coarse_ok[need] = collision_free(coarse_q[need], problems_json_text,
                                             problem_set_name, problem_idx)

        use_fb = need & coarse_ok           # LM colliding, coarse feasible -> take coarse
        infeasible = need & ~coarse_ok      # nothing feasible for this seed (a colliding SEED that
                                            # the coarse gate could never escape). Flagged, never
                                            # selected: its cost is +inf so argmin skips it.
        _t_filter = (_time.perf_counter() - _t0) * 1e3

        take_coarse = use_fb | infeasible
        dt = lm["joint_config"].dtype
        out["joint_config"] = _np.where(take_coarse[:, None],
                                        _np.asarray(c["joint_config"], dtype=dt),
                                        lm["joint_config"])
        # A fallback candidate reports ITS OWN (coarse) metrics -- not the LM's, which describe a
        # configuration we are not returning.
        for k in ("position_errors", "orientation_errors"):
            out[k] = _np.where(take_coarse[:, None], c[k], lm[k])
        out["cost"] = _np.where(take_coarse, c["cost"], lm["cost"])
        out["success"] = _np.where(take_coarse, c["success"], lm["success"]) & ~infeasible
        out["cost"][infeasible] = _np.inf          # unrankable: argmin(cost) can never pick it

        out["collision_free"] = ~infeasible
        out["lm_collision_free"] = lm_free
        out["used_coarse_fallback"] = use_fb
        out["n_lm_colliding"] = int(need.sum())
        out["n_lm_collision_free"] = int(lm_free.sum())
        out["n_coarse_fallbacks"] = int(use_fb.sum())
        out["n_infeasible"] = int(infeasible.sum())
        out["collision_filter_ms"] = _t_filter
    if diagnostics:
        z = _np.zeros(B, dtype=int)
        def _c(key):
            if cdiag is None:
                return z.copy()
            v = _np.asarray(cdiag[key]).copy()
            v[~use_coarse] = 0          # an LM-only problem has ZERO coarse counters, by definition
            return v
        out["coarse_iterations"] = _c("coarse_iterations")
        out["accepted_coarse_steps"] = _c("accepted_coarse_steps")
        out["rejected_coarse_steps"] = _c("rejected_coarse_steps")
        out["coarse_perturbations"] = _c("coarse_perturbations")
        out["coarse_max_stall"] = _c("coarse_max_stall")
        out["coarse_perturbation_events"] = _c("coarse_perturbation_events")
        out["coarse_perturbation_attempts"] = _c("coarse_perturbation_attempts")
        out["coarse_perturbations_rejected"] = _c("coarse_perturbations_rejected")
        out["coarse_perturbations_exhausted"] = _c("coarse_perturbations_exhausted")
    return out


# =================================================================================================
# BATCHED-PROBLEM API (Milestone 2). Solve P distinct multi-target IK problems in parallel, each
# with its own targets/mask and S candidate seeds, in ONE GPU submission (no Python loop).
#
#   b = p*S + s   flattens (problem p, seed s) into a candidate index. The candidate kernels see
#   B = P*S blocks; a block reads its PROBLEM-level data (targets, mask, weights) via pid = gp/S and
#   its CANDIDATE-level data (seed, all outputs) via gp. Targets/masks are stored ONCE per problem
#   ([P, K, ...] / [P]) and never broadcast to [P, S, K, ...].
# =================================================================================================
def _canonical_problems(target_poses, active_masks, seed_configs, dtype):
    """Validate + canonicalize the batched-problem inputs. Returns
        seeds_flat [B, N] (wire dtype, C-contig),
        tgt_p [P, K, 3], tgt_q [P, K, 4]  (split from the [P,K,7] poses; quats WXYZ, normalized),
        packed [P] uint32 masks,
        P, S.
    Raises a clear ValueError/TypeError BEFORE any CUDA work on malformed input.
    """
    N, K = num_joints(), num_targets()

    tp = _np.asarray(target_poses)
    if tp.ndim != 3 or tp.shape[1] != K or tp.shape[2] != 7:
        raise ValueError(f"target_poses must be [P, K={K}, 7], got {tp.shape}")
    P = int(tp.shape[0])

    sc = _np.asarray(seed_configs)
    if sc.ndim != 3 or sc.shape[2] != N:
        raise ValueError(f"seed_configs must be [P, S, N={N}], got {sc.shape}")
    if sc.shape[0] != P:
        raise ValueError(f"seed_configs P={sc.shape[0]} != target_poses P={P}")
    S = int(sc.shape[1])

    am = _np.asarray(active_masks)
    if am.ndim != 1 or am.shape[0] != P:
        raise ValueError(f"active_masks must be [P={P}], got {am.shape}")

    if P <= 0 or S <= 0:
        raise ValueError(f"need P>0 and S>0, got P={P}, S={S}")
    # B = P*S must not overflow a C int (the kernel grid and workspace index it as int/size_t).
    if P * S > 2**31 - 1:
        raise ValueError(f"P*S = {P*S} overflows int32; reduce the batch")

    if not _np.all(_np.isfinite(tp)):
        raise ValueError("target_poses contains NaN or inf")
    if not _np.all(_np.isfinite(sc)):
        raise ValueError("seed_configs contains NaN or inf")

    # masks: pack + validate bits (reuses the single-problem validator, which rejects empty masks
    # and bits above K).
    packed = pack_active_mask(am.astype(_np.int64) if _np.issubdtype(am.dtype, _np.integer)
                              else am, P, K)

    # Split pose7 -> position + quaternion; normalize (WXYZ). The quaternion is cast to the WIRE
    # dtype BEFORE normalizing, exactly as _canonical_problem does -- otherwise normalizing in
    # float64 and then narrowing differs by ~1 float32 ulp, which Policy B amplifies and breaks the
    # bitwise P=B,S=1 == candidate-specific-solve identity.
    pos = _np.ascontiguousarray(tp[:, :, 0:3], dtype=dtype)
    quat = _np.asarray(tp[:, :, 3:7], dtype=dtype)
    nrm = _np.linalg.norm(quat, axis=-1, keepdims=True)
    act = ((packed[:, None] >> _np.arange(K, dtype=_np.uint32)) & 1).astype(bool)
    if _np.any(nrm[act] < 1e-8):
        raise ValueError("target_poses has a (near-)zero quaternion for an active target")
    quat = _np.where(nrm > 0, quat / _np.where(nrm > 0, nrm, 1.0), quat).astype(dtype, copy=False)

    seeds_flat = _np.ascontiguousarray(sc.reshape(P * S, N), dtype=dtype)
    return seeds_flat, _np.ascontiguousarray(pos), _np.ascontiguousarray(quat), packed, P, S


def solve_problems(target_poses, active_masks, seed_configs,
                   num_solutions=1, precision="float32", coarse_mode="auto",
                   coarse_iters=120, lm_iters=60, seed=0, diagnostics=False,
                   stag_patience=2, stag_rel=1e-3,
                   position_tol=1e-4, orientation_tol=1e-3,
                   position_weights=1.0, orientation_weights=1.0,
                   problems_json_text="", problem_set_name="", problem_idx=0,
                   return_all_candidates=False, _solver=None):
    """Solve P distinct multi-target IK problems in parallel, returning the top-1 per problem.

    Args:
        target_poses  [P, K, 7]  per problem: K target poses [x,y,z, qw,qx,qy,qz] (WXYZ).
        active_masks  [P]        per problem: uint bitmask, bit k = target k active.
        seed_configs  [P, S, N]  per problem: S candidate seed configurations.

    Selection (on the GPU, one block per problem) ranks candidates by the deterministic three-class
    key R = (class, E_phys, seed): class 0 solved < class 1 valid-unsolved < class 2 invalid, then
    lower E_phys, then lower seed index. E_phys is the STABLE tolerance-normalised physical error --
    the row-scaled LM cost is NEVER used for selection (carried only as cost_lm).

    Returns (num_solutions=1), keeping the solution dimension for M4 top-M compatibility:
        joint_config        [P, 1, N]     the selected config, in the requested precision
        success             [P, 1]        the selected candidate met every active tolerance
        valid               [P, 1]        a valid (finite, feasible) candidate was selected
        cost_physical       [P, 1]        E_phys of the selected candidate (+inf if none valid)
        cost_lm             [P, 1]        row-scaled LM cost of the selected candidate (diagnostic)
        position_errors     [P, 1, K]     of the selected candidate
        orientation_errors  [P, 1, K]
        selected_seed_ids   [P, 1]        seed index within the problem, or -1 if none valid
        problem_success     [P]           at least one solved candidate existed
        num_solved          [P]  num_valid [P]
        active_masks        [P]
      when collision is enabled, also collision_free [P,1], used_coarse_fallback [P,1], and the
      per-problem counts num_collision_free / num_lm_colliding / num_coarse_fallbacks / num_infeasible.

    ALL-INVALID FILL: if every candidate of a problem is invalid, the returned config is that
    problem's FIRST input seed (if finite) else zeros, with selected_seed_id=-1, cost=+inf,
    success=valid=False, and (collision) collision_free=False -- never marked feasible.

    return_all_candidates=True additionally returns the full [P, S, ...] post-fallback candidate
    arrays under all_* keys, for debugging. The SELECTED outputs are identical either way.
    """
    if coarse_mode not in ("auto", "none", "multi_target"):
        raise ValueError(f"coarse_mode must be auto|none|multi_target, got '{coarse_mode}'")
    pc = _precision_code(precision)
    wire = _np.float32 if pc == 1 else _np.float64
    N, K = num_joints(), num_targets()

    seeds_flat, pos, quat, packed, P, S = _canonical_problems(
        target_poses, active_masks, seed_configs, wire)
    B = P * S
    if not (1 <= num_solutions <= S):
        raise ValueError(f"num_solutions must be in [1, S={S}], got {num_solutions}")

    wp = _bcast_weights(position_weights, P, K, "position_weights").astype(wire, copy=False)
    wo = _bcast_weights(orientation_weights, P, K, "orientation_weights").astype(wire, copy=False)
    wp = _np.ascontiguousarray(wp); wo = _np.ascontiguousarray(wo)

    cc_enabled = bool(problems_json_text) and bool(problem_set_name)

    # PER-PROBLEM dispatch, expanded to per-candidate [B] uint8. use_coarse[p] from popcount(mask[p]).
    popc = _np.array([bin(int(m)).count("1") for m in packed])
    if coarse_mode == "none":
        use_coarse_p = _np.zeros(P, dtype=bool)
    elif coarse_mode == "multi_target":
        use_coarse_p = _np.ones(P, dtype=bool)
    else:
        use_coarse_p = popc >= 2
    if coarse_iters <= 0:
        use_coarse_p = _np.zeros(P, dtype=bool)
    use_coarse = _np.ascontiguousarray(_np.repeat(use_coarse_p, S).astype(_np.uint8))   # [B]
    run_coarse = bool(use_coarse.any()) or cc_enabled

    sv = _solver or _default_solver()
    sv._enter()
    try:
        out = _solve_problems_raw(
            seeds_flat, pos, quat, packed, wp, wo, use_coarse, bool(run_coarse),
            float(position_tol), float(orientation_tol),
            1e-6, 1e-9, 0.35, int(coarse_iters), 5, 1, int(seed), 4,
            5e-3, int(lm_iters), int(stag_patience), float(stag_rel),
            int(num_solutions), pc, bool(return_all_candidates),
            str(problems_json_text), str(problem_set_name), int(problem_idx), sv._ws)
    finally:
        sv._exit()

    out["num_problems"] = P
    out["seeds_per_problem"] = S
    if return_all_candidates:
        # E_phys per candidate for the debug arrays (matches the device selection metric).
        act = ((packed[:, None] >> _np.arange(K, dtype=_np.uint32)) & 1).astype(bool)   # [P,K]
        act_b = act[:, None, :]                                                          # [P,1,K]
        pe = out["all_position_errors"]; oe = out["all_orientation_errors"]
        out["all_cost_physical"] = (((pe / position_tol) ** 2 + (oe / orientation_tol) ** 2)
                                    * act_b).sum(axis=2)
    return out
