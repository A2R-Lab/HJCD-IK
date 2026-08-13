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

from . import _hjcdik            # module handle for the Checkpoint 3 sidecar entry points
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
    "self_collision_info", "joint_names", "target_names", "target_axes",
    "ORI_NONE", "ORI_AXIS", "ORI_FULL",
]

# The generated target order is FIXED and is the order of every [.., K, ..] axis in this API, in
# every result array, and in the benchmarks. It is whatever order the targets were declared to
# scripts/codegen/generate_grid.py in -- for the current G1 build, left hand / right hand / left
# foot / right foot; for a single-target robot such as Panda, K == 1 and index 0 is its one
# generated target. Call target_names() for the compiled build's actual order rather than assuming
# one; likewise joint_names() for the configuration-vector order.


# --- generated model metadata (names) ------------------------------------------------------------
# num_joints()/num_targets()/target_metadata() come from the compiled extension and carry no
# strings: the CUDA hot path never sees a name. The names are resolved at codegen time and written
# to the JSON sidecar below, which ships as package data. Read it through importlib.resources so
# this works identically from a wheel, a site-packages install and an editable checkout -- never
# relative to the CWD, and never by walking back up to a repository layout.

_METADATA_RESOURCE = "hjcd_targets.json"

_metadata_lock = _threading.Lock()
_metadata_cache = None
_names_cache = {}


def _read_metadata_text():
    """Return the packaged metadata sidecar as text.

    Resolved through importlib.resources against this package, so it works identically from a
    wheel, a site-packages install and an editable checkout. Isolated in its own function to give
    the tests a seam for injecting malformed metadata without touching the real file.
    """
    try:
        from importlib.resources import files as _files
    except ImportError:                      # Python 3.8; the project floor is 3.9, so belt-and-braces
        from importlib_resources import files as _files   # type: ignore[import-not-found]

    return (_files(__package__) / _METADATA_RESOURCE).read_text(encoding="utf-8")


def _generated_metadata():
    """Parse (once) and return the generated metadata sidecar as a dict."""
    global _metadata_cache
    if _metadata_cache is not None:
        return _metadata_cache
    with _metadata_lock:
        if _metadata_cache is not None:      # another thread won the race
            return _metadata_cache

        import json as _json

        try:
            text = _read_metadata_text()
        except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:
            raise RuntimeError(
                f"Generated metadata {_METADATA_RESOURCE!r} is missing from the hjcdik package. "
                "It is written by scripts/codegen/generate_grid.py; regenerate the build "
                "(scripts/dev/g1_check.sh, or generate_grid.py directly) to restore it."
            ) from exc

        try:
            meta = _json.loads(text)
        except ValueError as exc:
            raise RuntimeError(
                f"Generated metadata {_METADATA_RESOURCE!r} is not valid JSON: {exc}. "
                "Regenerate it with scripts/codegen/generate_grid.py."
            ) from exc

        if not isinstance(meta, dict):
            raise RuntimeError(
                f"Generated metadata {_METADATA_RESOURCE!r} must be a JSON object, "
                f"got {type(meta).__name__}."
            )

        _metadata_cache = meta
        return _metadata_cache


def _generated_names(key, expected, what):
    """Read `key` from the sidecar, validating it against the compiled extension.

    A stale sidecar is the failure mode that matters: the .so is rebuilt for a different robot but
    the JSON is left behind, and every name is then silently wrong-by-one. Checking the length
    against the extension's own count catches that at the first call rather than downstream.

    The validated tuple is memoized, so callers get the identical object every time and the
    validation runs once.
    """
    cached = _names_cache.get(key)
    if cached is not None:
        return cached

    meta = _generated_metadata()

    if key not in meta:
        raise RuntimeError(
            f"Generated metadata {_METADATA_RESOURCE!r} has no {key!r} field. It predates the "
            "metadata-names change; regenerate it with scripts/codegen/generate_grid.py."
        )

    names = meta[key]
    if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
        raise RuntimeError(
            f"Generated metadata {key!r} must be a list of strings, got {type(names).__name__}."
        )

    if not names:
        raise RuntimeError(f"Generated metadata {key!r} is empty; the build has no {what}.")

    if len(set(names)) != len(names):
        seen, dupes = set(), []
        for name in names:
            if name in seen and name not in dupes:
                dupes.append(name)
            seen.add(name)
        raise RuntimeError(
            f"Generated metadata {key!r} contains duplicate {what} names: {dupes}. "
            "Names index the generated order, so they must be unique."
        )

    if len(names) != expected:
        raise RuntimeError(
            f"Generated metadata is stale: {key!r} has {len(names)} entries but the compiled "
            f"extension reports {expected} {what}. The .so and {_METADATA_RESOURCE!r} came from "
            "different codegen runs; re-run scripts/codegen/generate_grid.py and rebuild."
        )

    result = tuple(names)
    _names_cache[key] = result
    return result


def joint_names():
    """Generated joint names, in configuration-vector order.

    ``joint_names()[j]`` names joint ``j`` -- the same index used by ``joint_limits()``, by the
    ``q`` vectors passed to and returned by the solver, and by every per-joint device array.

    Returns a tuple of ``num_joints()`` unique strings. Raises RuntimeError if the generated
    metadata is missing, malformed, or inconsistent with the compiled extension.
    """
    return _generated_names("joint_names", num_joints(), "joints")


def target_names():
    """Generated target names, in target order.

    ``target_names()[k]`` names target ``k`` -- the index of every ``[.., K, ..]`` axis in this API.

    Returns a tuple of ``num_targets()`` unique strings. Raises RuntimeError if the generated
    metadata is missing, malformed, or inconsistent with the compiled extension.
    """
    return _generated_names("target_names", num_targets(), "targets")


def target_axes():
    """Generated per-target contact axes, in target order: float64 ``[K, 3]``.

    ``target_axes()[k]`` is target k's default ORI_AXIS direction, expressed in that target's own
    frame and unit-normalized at codegen time. It is the axis
    ``solve_problems(orientation_modes="axis")`` aligns unless an ``orientation_axes=`` override is
    given.

    A target that declared no ``orientation_axis=`` at codegen gets a ZERO row. That is a sentinel,
    not a direction -- no unit axis is zero -- and requesting AXIS mode for such a target raises
    rather than silently constraining nothing. Builds generated before this field existed return
    all-zero, i.e. "no target declares an axis", which is the correct answer for them.
    """
    K = num_targets()
    meta = _generated_metadata()
    raw = meta.get("target_orientation_axes")
    if raw is None:
        return _np.zeros((K, 3), dtype=_np.float64)
    a = _np.asarray(raw, dtype=_np.float64)
    if a.shape != (K, 3):
        raise RuntimeError(
            f"Generated metadata 'target_orientation_axes' has shape {a.shape}, expected "
            f"{(K, 3)} for the compiled build's {K} targets. The .so and the metadata came from "
            "different codegen runs; re-run scripts/codegen/generate_grid.py.")
    return a


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


# --- per-target orientation modes -----------------------------------------------------------
# A contact generally pins position and a contact NORMAL, not a full pose: the twist about that
# normal is a free DoF of the contact, and constraining it turns a feasible multi-contact stance
# into an infeasible one. ORI_AXIS expresses that; ORI_FULL is the legacy full-quaternion
# constraint and stays the default so no existing call changes behaviour.
ORI_NONE, ORI_AXIS, ORI_FULL = 0, 1, 2

_ORI_MODE_NAMES = {"none": ORI_NONE, "axis": ORI_AXIS, "full": ORI_FULL}


def _ori_mode_code(v, name):
    """'none'|'axis'|'full' (any case) or 0|1|2 -> int code."""
    if isinstance(v, str):
        try:
            return _ORI_MODE_NAMES[v.strip().lower()]
        except KeyError:
            raise ValueError(
                f"{name}: unknown orientation mode {v!r}; "
                f"expected one of {sorted(_ORI_MODE_NAMES)} or 0|1|2") from None
    iv = int(v)
    if iv not in (ORI_NONE, ORI_AXIS, ORI_FULL):
        raise ValueError(f"{name}: orientation mode must be 0 (NONE), 1 (AXIS) or 2 (FULL), got {iv}")
    return iv


def _bcast_ori_modes(modes, P, K, name="orientation_modes"):
    """None | scalar | [K] | [P,K] of names/codes -> contiguous [P,K] int32, or None for all-FULL.

    None returns None rather than a FULL-filled array so the whole feature stays a null pointer
    down to the kernel: a legacy caller must not merely *behave* like the old path, it must take it.
    """
    if modes is None:
        return None
    a = _np.asarray(modes, dtype=object)
    if a.ndim == 0:
        out = _np.full((P, K), _ori_mode_code(a.item(), name), dtype=_np.int32)
    elif a.ndim == 1 and a.shape[0] == K:
        row = _np.array([_ori_mode_code(v, name) for v in a], dtype=_np.int32)
        out = _np.broadcast_to(row, (P, K))
    elif a.ndim == 2 and a.shape == (P, K):
        out = _np.array([[_ori_mode_code(v, name) for v in r] for r in a], dtype=_np.int32)
    else:
        raise ValueError(f"{name} must be scalar, [K]={K}, or [P,K]={(P, K)}; got shape {a.shape}")
    return _np.ascontiguousarray(out, dtype=_np.int32)


def _bcast_ori_axes(axes, P, K, name="orientation_axes"):
    """None | [3] | [K,3] | [P,K,3] -> contiguous [P,K,3] float64 of UNIT axes, or None.

    None falls back to the generated per-target default (target_axes()); a target with neither an
    override nor a generated axis yields a zero row, which only matters if that target is AXIS --
    _require_axes_for_axis_mode is what turns that into a loud error.
    """
    if axes is None:
        default = target_axes()                      # [K,3]; zero row == "no generated axis"
        if not _np.any(default):
            return None                              # nothing to send: no target declares an axis
        out = _np.broadcast_to(default, (P, K, 3))
    else:
        a = _np.asarray(axes, dtype=_np.float64)
        if a.ndim == 1 and a.shape == (3,):
            out = _np.broadcast_to(a, (P, K, 3))
        elif a.ndim == 2 and a.shape == (K, 3):
            out = _np.broadcast_to(a, (P, K, 3))
        elif a.ndim == 3 and a.shape == (P, K, 3):
            out = a
        else:
            raise ValueError(
                f"{name} must be [3], [K,3]={(K, 3)}, or [P,K,3]={(P, K, 3)}; got shape {a.shape}")
        if not _np.all(_np.isfinite(out)):
            raise ValueError(f"{name} contains NaN or inf")
    out = _np.array(out, dtype=_np.float64, copy=True)
    n = _np.linalg.norm(out, axis=-1, keepdims=True)
    nz = n[..., 0] > 0
    out[nz] /= n[nz]                                 # normalize; zero rows stay zero
    return _np.ascontiguousarray(out)


def _require_axes_for_axis_mode(modes, axes, active, P, K):
    """AXIS without a usable axis is a silent wrong answer, so make it a loud one.

    A zero-length axis cannot define a direction: R a would be the zero vector and the residual
    would be identically zero, i.e. the target would appear perfectly solved while constraining
    nothing at all. Checked only for ACTIVE targets, since an inactive target is never evaluated.
    """
    if modes is None:
        return
    is_axis = (modes == ORI_AXIS)
    act = ((_np.asarray(active, dtype=_np.uint32)[:, None] >> _np.arange(K, dtype=_np.uint32)) & 1)
    need = is_axis & act.astype(bool)
    if not need.any():
        return
    if axes is None:
        bad = _np.argwhere(need)
        raise ValueError(
            f"orientation_modes requests AXIS for target(s) {sorted(set(int(k) for _, k in bad))} "
            "but no axis is available: the build's generated metadata declares no orientation_axis "
            "for them and no orientation_axes= override was passed. Give the target an axis at "
            "codegen time (--target '...;orientation_axis=x,y,z') or pass orientation_axes=.")
    norms = _np.linalg.norm(axes, axis=-1)
    bad = _np.argwhere(need & (norms <= 0))
    if bad.size:
        raise ValueError(
            "orientation_modes requests AXIS for (problem, target) "
            f"{[(int(p), int(k)) for p, k in bad]} but the corresponding orientation_axes entry is "
            "a zero vector, which defines no direction.")


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
                  _seeds_per_problem=1,
                  hard_self_collision=0, hard_top_k=3, hard_margin=0.0, hard_max_reseed=8,
                  hard_diagnostics=False, hard_oracle_every=0,
                  hard_reseed_mode=1, hard_reseed_candidates=16, hard_reseed_rounds=2,
                  hard_reseed_scales=(0.10, 0.20, 0.35, 0.50)):
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
                                  int(max_pert_attempts), pc, sv._ws, int(_seeds_per_problem),
                                  int(hard_self_collision), int(hard_top_k), float(hard_margin),
                                  int(hard_max_reseed), bool(hard_diagnostics),
                                  int(hard_oracle_every), int(hard_reseed_mode),
                                  int(hard_reseed_candidates), int(hard_reseed_rounds),
                                  [float(x) for x in hard_reseed_scales])
    finally:
        sv._exit()


# =================================================================================================
# SELF-COLLISION SIDECAR (Checkpoint 3). The validated GPU self-collision sidecar is compiled into
# the _hjcdik extension (Stage 3A). These helpers upload its model data lazily -- ONLY when a caller
# requests self_collision_mode != "off", so off mode allocates nothing and launches no sidecar work.
# =================================================================================================
_SELF_COLLISION_READY = False


def _sidecar_gen_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "generated")


def _ensure_self_collision_sidecar():
    """Upload torso/pelvis SDF grids + convex vertices into the extension's sidecar (once)."""
    global _SELF_COLLISION_READY
    if _SELF_COLLISION_READY:
        return
    import json as _json
    gen = _sidecar_gen_dir()
    for cid, fn in ((0, "g1_torso_sdf.npz"), (1, "g1_pelvis_sdf.npz")):
        z = _np.load(os.path.join(gen, fn), allow_pickle=True)
        _hjcdik.sidecar_upload_sdf(cid, _np.ascontiguousarray(
            z["sdf_i16"].astype(_np.int16).ravel(order="C")))
    # canonical convex verts (piece order == PIECE_VERT_OFF) written by emit_cuda_header.py
    verts = _np.ascontiguousarray(_np.load(os.path.join(gen, "g1_convex_verts.npy")).astype(_np.float64))
    _hjcdik.sidecar_upload_convex(verts)
    _SELF_COLLISION_READY = True


def self_collision_info():
    """Model information for the compiled self-collision sidecar: compiled flag, supported modes,
    hard-enabled flag, and all artifact hashes (URDF / joint-order / proxy / SDFs / convex / policy).

    `hard_enabled` reflects what this BUILD can actually do -- the sidecar model is G1-specific, so
    a build whose grid.cuh is a different robot reports False and solve(self_collision_mode="hard")
    raises rather than checking one robot's geometry against another's kinematics.
    """
    d = dict(_hjcdik.sidecar_model_info())
    hard = bool(_hjcdik.hard_available())
    d["hard_enabled"] = hard
    d["supported_modes"] = ["off", "final"] + (["hard"] if hard else [])
    d["incremental_checker"] = hard
    d["top_k_max"] = int(_hjcdik.hard_max_top_k())
    d["hard_workspace_allocations"] = int(_hjcdik.hard_ws_nalloc())
    d["hard_workspace_capacity"] = int(_hjcdik.hard_ws_capacity())
    d["hard_counter_stride"] = int(_hjcdik.hard_ctr_stride())
    d["geometry_validated"] = _geometry_artifacts_ok()
    return d


def _geometry_artifacts_ok():
    """Do the on-disk sidecar artifacts still match the hashes compiled into the extension?

    The generated .cuh carries the hashes of the URDF / proxy / SDF / convex / pair-policy it came
    from. If someone regenerates an artifact without rebuilding, the kernel would be checking STALE
    geometry and every collision guarantee here would be void -- so this is checked, not assumed.
    Returns None when the artifact carries no hashes to compare against.
    """
    import json as _json
    try:
        art = _json.load(open(os.path.join(_sidecar_gen_dir(), "g1_collision_sidecar.json")))
    except (OSError, ValueError):
        return None
    compiled = dict(_hjcdik.sidecar_model_info())["hashes"]
    # The artifact stores the two hashes it is itself the authority for; the SDF/convex/URDF hashes
    # are compiled in from their own generators and have no on-disk counterpart here.
    pairs = [("joint_order", art.get("joint_order_hash")),
             ("proxy_yaml", art.get("proxy_yaml_hash"))]
    pairs = [(k, v) for k, v in pairs if v is not None]
    if not pairs:
        return None
    return all(compiled.get(k) == v for k, v in pairs)


def solve(q, target_positions, target_quaternions, active_target_mask=None,
          position_weights=1.0, orientation_weights=1.0,
          position_tol=1e-4, orientation_tol=1e-3,
          coarse_mode="auto", coarse_iters=60, coarse_incremental=True,
          lm_iters=60, seed=0, diagnostics=False,
          problems_json_text="", problem_set_name="", problem_idx=0,
          precision="float32", coarse_precision=None, lm_precision=None,
          stag_patience=2, stag_rel=1e-3, _solver=None,
          self_collision_mode="off", collision_top_k=3,
          collision_reseed_candidates=16, collision_reseed_rounds=2,
          collision_reseed_scales=(0.10, 0.20, 0.35, 0.50), collision_reseed_mode=1,
          _hard_oracle_every=0):
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
    # -- self-collision gate (Checkpoint 3). Validate BEFORE any compute; off == exact baseline. --
    if self_collision_mode not in ("off", "final", "hard"):
        raise ValueError(f"self_collision_mode must be off|final|hard, got '{self_collision_mode!r}'")
    if isinstance(collision_top_k, bool) or not isinstance(collision_top_k, int) \
            or collision_top_k < 1:
        raise ValueError(f"collision_top_k must be a positive int, got {collision_top_k!r}")
    if self_collision_mode == "hard":
        if not _hjcdik.hard_available():
            raise NotImplementedError(
                "self_collision_mode='hard' is unavailable in this build: the compiled sidecar "
                "model is G1-specific and this grid.cuh is a different robot")
        _kmax = int(_hjcdik.hard_max_top_k())
        if collision_top_k > _kmax:
            raise ValueError(f"collision_top_k must be <= {_kmax}, got {collision_top_k}")
        _ensure_self_collision_sidecar()

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

    if self_collision_mode not in ("off", "final"):
        raise ValueError(
            f"solve_problems supports self_collision_mode off|final, got {self_collision_mode!r}"
            " (hard mode is a single-problem solve() feature)")
    if self_collision_mode == "final":
        # Uploaded BEFORE the solve now, not after: the sidecar's SDF/convex tables have to be
        # resident when solve_problems_batched calls the device check between LM and selection.
        _ensure_self_collision_sidecar()

    cc_enabled = bool(problems_json_text) and bool(problem_set_name)

    # The coarse stage also runs whenever collision is enabled, EVEN IF the dispatch would not
    # otherwise call for it (a single active target goes LM-only). It is the only stage with a hard
    # collision gate, so it is the only source of a guaranteed-feasible fallback for a seed whose LM
    # result ends up colliding. Its output is used ONLY as that fallback here -- the LM is still
    # seeded exactly as before (`use_coarse` alone decides that), so LM-only accuracy is unchanged.
    # ...but with collision on, the coarse stage is still needed to manufacture a feasible fallback,
    # so coarse_iters == 0 is only honoured as "no coarse launch" in the open-world case.
    #
    # Hard mode ALWAYS runs the coarse stage for the same reason: it is the only stage that produces
    # a VERIFIED collision-free configuration, and section 8's LM fallback has nothing to fall back
    # to without one.
    run_coarse = bool(use_coarse.any()) or cc_enabled or self_collision_mode == "hard"

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
                          precision=cp, _solver=_solver, _canonical=True,
                          hard_self_collision=1 if self_collision_mode == "hard" else 0,
                          hard_top_k=collision_top_k, hard_diagnostics=diagnostics,
                          hard_oracle_every=int(_hard_oracle_every),
                          hard_reseed_mode=int(collision_reseed_mode),
                          hard_reseed_candidates=int(collision_reseed_candidates),
                          hard_reseed_rounds=int(collision_reseed_rounds),
                          hard_reseed_scales=collision_reseed_scales)
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

    # -------------------------------------------------------------------------------------------
    # STAGE 3C -- FINAL SELF-COLLISION GATE. Separate from the env-collision fallback above. Runs
    # ONE batched sidecar full-check on the ALREADY-PRODUCED candidate q; coarse/LM/ranking are
    # untouched. Colliding candidates are marked unsuccessful; q values are NEVER modified, so every
    # free candidate is byte-identical to off mode. No persistent per-seed state (Stage 3D+).
    # -------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------
    # STAGE 3D/3E -- HARD MODE FINAL CHECK AND FALLBACK (spec section 8).
    #
    # The coarse stage returned a configuration that is collision-free BY CONSTRUCTION (a verified
    # seed, then only collision-gated commits). The LM that refines it is NOT gated -- exactly as in
    # the environment-collision case, it happily refines a feasible state straight back into a
    # collision -- so its answer is VALIDATED, never trusted:
    #
    #   LM free      -> return the LM result unchanged
    #   LM colliding -> return that seed's last collision-free coarse configuration
    #   no free coarse state (Stage 3D exhausted its retries) -> the seed is FAILED
    #
    # A fallback pose is re-measured against its own targets. Returning the LM's error metadata for
    # a configuration we are not returning is exactly the bug the environment-collision fallback
    # above had to fix, and it would silently report a pose that was never evaluated.
    # -------------------------------------------------------------------------------------------
    if self_collision_mode == "hard":
        import time as _time
        B = qa.shape[0]
        lm_q32 = _np.ascontiguousarray(_np.asarray(lm["joint_config"], dtype=_np.float32))
        _t0 = _time.perf_counter()
        lm_free = ~_np.asarray(_hjcdik.sidecar_full_check(lm_q32, 0.0)).any(axis=1)
        _t_lm_chk = (_time.perf_counter() - _t0) * 1e3

        # hard_flags is a uint8 bitfield: bit0 = committed state valid (Stage 3D found a free seed),
        # bit1 = a collision-free coarse configuration exists for this seed.
        raw_flags = _np.asarray(c["hard_flags"]).astype(_np.uint8)
        seed_ok = (raw_flags & 0x1) != 0
        has_free = (raw_flags & 0x2) != 0
        qfree = _np.asarray(c["hard_qfree"], dtype=_np.float32)          # [B, 29]

        use_fb = (~lm_free) & has_free
        failed = (~lm_free) & (~has_free)

        dt = lm["joint_config"].dtype
        out_q = _np.array(lm["joint_config"], dtype=dt, copy=True)
        out_q[use_fb] = qfree[use_fb].astype(dt)
        out["joint_config"] = out_q

        # A fallback row reports the COARSE stage's own metrics, because the pose it returns IS the
        # coarse stage's answer: hard mode's last_collision_free_coarse_q is published from best_x,
        # and out_pn/out_on/out_cost were computed from that same best_x after a full refresh. So
        # these are that pose's measured errors, not the LM's -- reusing the LM's numbers here would
        # describe a configuration we are not returning. (test_fallback_metadata_recomputed pins
        # the qfree == coarse joint_config identity this depends on.)
        for k in ("position_errors", "orientation_errors"):
            out[k] = _np.where(use_fb[:, None], c[k], lm[k])
        out["cost"] = _np.where(use_fb, c["cost"], lm["cost"])
        # Final success still requires the existing IK tolerance: a collision-free fallback that
        # misses it is a FAILED candidate, not a successful collision-free solution.
        out["success"] = (_np.where(use_fb, _np.asarray(c["success"]).astype(bool),
                                    _np.asarray(lm["success"]).astype(bool))
                          & ~failed & seed_ok)

        out["hard_last_free_coarse_q"] = qfree
        out["self_collision_free"] = lm_free | use_fb
        out["used_collision_fallback"] = use_fb
        out["hard_seed_ok"] = seed_ok

        sc = dict(mode="hard",
                  top_k=int(collision_top_k),
                  candidates_checked=int(B),
                  lm_collision_free=int(lm_free.sum()),
                  lm_colliding=int((~lm_free).sum()),
                  used_collision_fallback=int(use_fb.sum()),
                  fallback_success=int(_np.asarray(out["success"])[use_fb].sum())
                                   if use_fb.any() else 0,
                  seed_failures=int((~seed_ok).sum()),
                  unrecoverable=int(failed.sum()),
                  lm_check_ms=float(_t_lm_chk))
        sc.update(dict(c["hard"]))
        if diagnostics and "hard_counters" in c:
            ctr = _np.asarray(c["hard_counters"])
            KMAX = int(_hjcdik.hard_max_top_k())
            R0 = 11                      # == HARD_CTR_ACCEPT_RANK0 in collision_sidecar_hard.cuh
            for i, name in enumerate(("proposals_checked", "all_k_colliding", "proposals_rejected",
                                      "gjk_pairs", "gjk_iters", "nongjk_pairs", "sdf_evals",
                                      "perturbations_skipped", "trials_without_gjk",
                                      "oracle_checks", "oracle_mismatches")):
                sc[name] = int(ctr[:, i].sum())
            sc["accept_by_rank"] = [int(ctr[:, R0 + r].sum()) for r in range(KMAX)]
            sc["reject_by_joint"] = ctr[:, R0 + KMAX:].sum(axis=0).tolist()
            sc["counters"] = ctr
        out["self_collision"] = sc

    if self_collision_mode == "final":
        import time as _time
        _ensure_self_collision_sidecar()
        q32 = _np.ascontiguousarray(_np.asarray(out["joint_config"], dtype=_np.float32))
        _t0 = _time.perf_counter()
        verdict = _np.asarray(_hjcdik.sidecar_full_check(q32, 0.0))     # [B, n_pairs] uint8
        _t_sc = (_time.perf_counter() - _t0) * 1e3
        colliding = verdict.any(axis=1)
        prev_success = _np.asarray(out["success"]).astype(bool)
        out["success"] = prev_success & ~colliding
        out["self_collision_free"] = ~colliding
        sc = dict(mode="final",
                  candidates_checked=int(q32.shape[0]),
                  n_colliding=int(colliding.sum()),
                  candidates_rejected=int((prev_success & colliding).sum()),
                  kernel_ms=float(_t_sc))
        if diagnostics and colliding.any():
            gen = _sidecar_gen_dir()
            import json as _json
            pairs = _json.load(open(os.path.join(gen, "g1_collision_sidecar.json")))["checked_link_pairs"]
            bi = int(_np.argmax(colliding))
            pj = int(_np.argmax(verdict[bi]))
            sc["first_colliding"] = dict(candidate=bi, pair_index=pj, links=list(pairs[pj]))
        out["self_collision"] = sc
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
# The public base-update controls. Names deliberately mirror base_update.py's BaseUpdateConfig --
# it is the host reference for this exact solve, and a reader comparing them should not have to
# translate. Defaults are conservative and are to be tuned on benchmark evidence (M7), not
# intuition. `enabled` defaults to False: floating_base alone CARRIES a base without optimizing it
# (the externally-sampled-base behaviour, moved on-device), and turning the optimizer on is a
# separate, explicit decision.
_BASE_UPDATE_DEFAULTS = dict(enabled=False, interval=1, damping=1e-3, scale_p=1.0, scale_R=1.0,
                             step_scale=1.0, max_translation_step=0.05, max_rotation_step=0.10)
_UNBOUNDED = 1e30


def _canonical_base_update(base_update, base_bounds, floating_base):
    """Validate the public base-update config + bounds -> the raw binding kwargs.

    Everything is rejected BEFORE any CUDA work, with a message naming the offender: a bad base
    config that reaches the kernel does not raise, it silently solves a different problem.
    """
    if not floating_base:
        # Accepting these under a fixed base would be a silent no-op, which is how the base_bounds
        # kwarg spent all of M4 lying to callers.
        if base_update is not None:
            raise ValueError("base_update requires floating_base=True: there is no base to move")
        if base_bounds is not None:
            raise ValueError("base_bounds requires floating_base=True: there is no base to bound")
        return {}

    cfg = dict(_BASE_UPDATE_DEFAULTS)
    if base_update is not None:
        if not isinstance(base_update, dict):
            raise TypeError(f"base_update must be a dict of {sorted(_BASE_UPDATE_DEFAULTS)}, "
                            f"got {type(base_update).__name__}")
        unknown = sorted(set(base_update) - set(_BASE_UPDATE_DEFAULTS))
        if unknown:                      # a typo'd key must not be a silently ignored default
            raise ValueError(f"unknown base_update keys {unknown}; "
                             f"expected {sorted(_BASE_UPDATE_DEFAULTS)}")
        cfg.update(base_update)

    def _pos(name, allow_none=False):
        v = cfg[name]
        if v is None and allow_none:
            return 0.0                   # the kernel reads <= 0 as "no clip"
        v = float(v)
        if not (v > 0.0) or not _np.isfinite(v):
            raise ValueError(f"base_update['{name}'] must be > 0 and finite, got {cfg[name]}"
                             + (" (or None for no clipping)" if allow_none else ""))
        return v

    interval = int(cfg["interval"])
    if interval < 1:
        raise ValueError(f"base_update['interval'] must be >= 1, got {cfg['interval']}")
    damping = float(cfg["damping"])
    if not (damping >= 0.0) or not _np.isfinite(damping):
        raise ValueError(f"base_update['damping'] must be >= 0 and finite, got {cfg['damping']}")
    # scale_p/scale_R enter the damping metric D = diag(s_p^-2 I3, s_R^-2 I3) as s^-2, and their
    # positivity is what makes lambda*D positive definite -- which is what lets the kernel factor
    # H + lambda*D unconditionally. Zero is not "no scaling", it is a division by zero.
    scale_p, scale_R = _pos("scale_p"), _pos("scale_R")
    step_scale = _pos("step_scale")
    max_t = _pos("max_translation_step", allow_none=True)
    max_r = _pos("max_rotation_step", allow_none=True)

    lo, hi = [-_UNBOUNDED] * 3, [_UNBOUNDED] * 3
    if base_bounds is not None:
        try:
            lower, upper = base_bounds
        except (TypeError, ValueError):
            raise ValueError("base_bounds must be a (lower, upper) pair of 3-vectors, "
                             f"got {base_bounds!r}") from None
        lower = _np.asarray(lower, dtype=float).reshape(-1)
        upper = _np.asarray(upper, dtype=float).reshape(-1)
        if lower.shape != (3,) or upper.shape != (3,):
            raise ValueError(f"base_bounds must be two 3-vectors, got shapes "
                             f"{lower.shape} and {upper.shape}")
        if not (_np.all(_np.isfinite(lower)) and _np.all(_np.isfinite(upper))):
            raise ValueError("base_bounds must be finite; omit base_bounds for an unbounded base")
        bad = _np.nonzero(lower > upper)[0]
        if bad.size:
            raise ValueError(f"base_bounds lower must be <= upper on every axis; "
                             f"violated on axis {bad.tolist()} "
                             f"(lower={lower.tolist()}, upper={upper.tolist()})")
        lo, hi = lower.tolist(), upper.tolist()

    return dict(base_update_enabled=bool(cfg["enabled"]), base_update_interval=interval,
                base_damping=damping, base_step_scale=step_scale,
                base_damping_scale_p=scale_p, base_damping_scale_R=scale_R,
                base_max_translation_step=max_t, base_max_rotation_step=max_r,
                base_position_lower=lo, base_position_upper=hi)


def _canonical_problems(target_poses, active_masks, seed_configs, dtype, floating_base=False):
    """Validate + canonicalize the batched-problem inputs. Returns
        seeds_flat [B, N] (wire dtype, C-contig),
        tgt_p [P, K, 3], tgt_q [P, K, 4]  (split from the [P,K,7] poses; quats WXYZ, normalized),
        packed [P] uint32 masks,
        P, S,
        base_p [B, 3], base_q [B, 4] (wxyz, unit) when floating_base -- else EMPTY arrays,
        which the binding reads as "fixed base" and leaves null all the way down.
    Raises a clear ValueError/TypeError BEFORE any CUDA work on malformed input.
    """
    N, K = num_joints(), num_targets()

    tp = _np.asarray(target_poses)
    if tp.ndim != 3 or tp.shape[1] != K or tp.shape[2] != 7:
        raise ValueError(f"target_poses must be [P, K={K}, 7], got {tp.shape}")
    P = int(tp.shape[0])

    sc = _np.asarray(seed_configs)
    # Floating base: the seed carries the FULL state [x,y,z, qw,qx,qy,qz, N joints], not an
    # ambiguous 35-vector -- the optimization is over a 6D tangent, but the STATE is 7+N.
    want = (N + 7) if floating_base else N
    if sc.ndim != 3 or sc.shape[2] != want:
        raise ValueError(
            f"seed_configs must be [P, S, {want}], got {sc.shape}" + (
                f" -- floating_base=True expects [x,y,z, qw,qx,qy,qz, {N} joints]"
                if floating_base else ""))
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

    if floating_base:
        flat = sc.reshape(P * S, N + 7)
        # copy=True is REQUIRED, not defensive: the binding takes base_p as IN/OUT
        # (in.base_p = base_p.mutable_data(), then returns it as "base_position"), so this array
        # must be one we own. ascontiguousarray would alias the caller's seed_configs at B == 1 --
        # a [1,3] column slice counts as C-contiguous because a leading dim of 1 makes its stride
        # irrelevant -- and the refined base would land back in the caller's seeds.
        base_p = _np.array(flat[:, 0:3], dtype=dtype, order="C", copy=True)   # [B,3]
        bq = _np.asarray(flat[:, 3:7], dtype=_np.float64)                 # [B,4] wxyz
        nrm = _np.linalg.norm(bq, axis=1)
        if _np.any(nrm < 1e-8):
            raise ValueError("seed_configs has a (near-)zero base quaternion; "
                             "expected [x,y,z, qw,qx,qy,qz, joints] with a unit quaternion")
        # Normalize here so the kernel can assume unit -- it does, and never renormalizes
        # (the same contract tgt_q already has). q and -q are the same rotation; both are fine.
        base_q = _np.ascontiguousarray(bq / nrm[:, None], dtype=dtype)
        joints = flat[:, 7:]
    else:
        base_p = _np.zeros((0,), dtype=dtype)      # empty => the binding leaves in.base_* null
        base_q = _np.zeros((0,), dtype=dtype)
        joints = sc.reshape(P * S, N)

    # copy=True for the same reason base_p above copies: seed_configs is the CALLER's array and
    # solve_problems must never write to it. The joints happen to be safe today (the binding
    # uploads them via a const q.data() and downloads into its own output), but that is the
    # binding's private choice, not a promise to this layer -- and ascontiguousarray would hand
    # over a live view of the caller's array at B == 1 the moment it changed its mind.
    seeds_flat = _np.array(joints, dtype=dtype, order="C", copy=True)
    return (seeds_flat, _np.ascontiguousarray(pos), _np.ascontiguousarray(quat), packed, P, S,
            base_p, base_q)


def solve_problems(target_poses, active_masks, seed_configs,
                   problem_seeds=None,
                   num_solutions=1, precision="float32", coarse_mode="auto",
                   coarse_iters=120, lm_iters=60, seed=0, diagnostics=False,
                   stag_patience=2, stag_rel=1e-3,
                   position_tol=1e-4, orientation_tol=1e-3,
                   position_weights=1.0, orientation_weights=1.0,
                   problems_json_text="", problem_set_name="", problem_idx=0,
                   return_all_candidates=False, floating_base=False, base_bounds=None,
                   base_update=None, self_collision_mode="off",
                   self_collision_margin=0.0, self_collision_eligible_tol=None,
                   orientation_modes=None, orientation_axes=None,
                   _solver=None):
    """Solve P distinct multi-target IK problems in parallel, returning the top-1 per problem.

    Args:
        target_poses  [P, K, 7]  per problem: K target poses [x,y,z, qw,qx,qy,qz] (WXYZ).
        active_masks  [P]        per problem: uint bitmask, bit k = target k active.
        seed_configs  [P, S, N]  per problem: S candidate seed configurations.
                                 floating_base=True instead expects [P, S, 7+N]:
                                 [x,y,z, qw,qx,qy,qz, N joints]. The optimization is over a 6-D
                                 tangent but the STATE is 7+N, so the seed carries all of it.

    Floating base (all optional; a fixed-base call is unaffected by every one of them):
        floating_base  False  each candidate carries its own world base pose. The targets stay in
                              the WORLD frame; each candidate's copy is expressed in its own base.
        base_update    None   dict of controls; keys mirror base_update.py's BaseUpdateConfig:
                                enabled              False  optimize the base, not just carry it
                                interval             1      take a base step every N LM iterations
                                damping              1e-3   lambda in H_lambda = H + lambda*D
                                scale_p, scale_R     1.0    D = diag(s_p^-2 I3, s_R^-2 I3), > 0
                                step_scale           1.0    alpha on the accepted step
                                max_translation_step 0.05   m,   None => unclipped
                                max_rotation_step    0.10   rad, None => unclipped
        base_bounds    None   (lower3, upper3) world box the base position is clamped into.

    THE BASE IS NOT UNIQUELY IDENTIFIABLE from position targets. With N joints free there are many
    (base, joints) pairs that put the K contacts in the same place, and this returns one of them --
    measured on G1: a task solved to 2.9e-05 m with a base 0.044 m from the one it was built from.
    So judge a returned base by the error it produces, never by comparing it to a base you expected.
    A base is only meaningful WITH the joints it was scored against; the two are returned as a pair
    and must be used as one.

    Base updates run in LM REFINEMENT ONLY -- never in the coarse sweep, whose greedy per-joint
    accept/rollback a base move (which perturbs every target at once) would fight. coarse_iters
    therefore does no base optimization, and a coarse-only run leaves the base at its seed.

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
      and when floating_base=True, gathered to the SAME [P, M, ...] selection as joint_config:
        base_position       [P, 1, 3]     the selected candidate's world base position
        base_quaternion     [P, 1, 4]     its world base orientation (WXYZ, unit)
        problem_success     [P]           at least one solved candidate existed
        num_solved          [P]  num_valid [P]
        active_masks        [P]
      when collision is enabled, also collision_free [P,1], used_coarse_fallback [P,1], and the
      per-problem counts num_collision_free / num_lm_colliding / num_coarse_fallbacks / num_infeasible.

    ALL-INVALID FILL: if every candidate of a problem is invalid, the returned config is that
    problem's FIRST input seed (if finite) else zeros, with selected_seed_id=-1, cost=+inf,
    success=valid=False, and (collision) collision_free=False -- never marked feasible. The base
    follows the same rule: base_position/base_quaternion are that problem's FIRST INPUT seed base,
    so the returned (base, joints) stay the pair the fill describes rather than a live base bolted
    to a fallback config.

    return_all_candidates=True additionally returns the full [P, S, ...] post-fallback candidate
    arrays under all_* keys, for debugging. The SELECTED outputs are identical either way.
    """
    if coarse_mode not in ("auto", "none", "multi_target"):
        raise ValueError(f"coarse_mode must be auto|none|multi_target, got '{coarse_mode}'")
    pc = _precision_code(precision)
    wire = _np.float32 if pc == 1 else _np.float64
    N, K = num_joints(), num_targets()
    floating_base = bool(floating_base)
    _bu = _canonical_base_update(base_update, base_bounds, floating_base)

    seeds_flat, pos, quat, packed, P, S, base_p, base_q = _canonical_problems(
        target_poses, active_masks, seed_configs, wire, floating_base=floating_base)
    # base_p/base_q are IN/OUT at the binding (in.base_p = base_p.mutable_data(), returned as
    # "base_position"), so the seed base is GONE after the call. Keep it: the all-invalid fill
    # below owes the caller its first input seed's base, not whatever the kernel left there.
    base_p_in = base_p.copy() if floating_base else None
    base_q_in = base_q.copy() if floating_base else None
    B = P * S
    # Always collected when floating: the kernel writes them once in the epilogue from values it
    # already holds (measured: 0 extra registers on sm_89, where there are only 3 to spare), and
    # without them an acceptance RATE is unobservable -- a base update that proposes every sweep
    # and is rejected every time looks exactly like one that never ran.
    base_diag = _np.zeros((B, 3), dtype=_np.int32) if floating_base else _np.zeros((0,), _np.int32)
    if not (1 <= num_solutions <= S):
        raise ValueError(f"num_solutions must be in [1, S={S}], got {num_solutions}")

    wp = _bcast_weights(position_weights, P, K, "position_weights").astype(wire, copy=False)
    wo = _bcast_weights(orientation_weights, P, K, "orientation_weights").astype(wire, copy=False)
    om = _bcast_ori_modes(orientation_modes, P, K)
    oa = _bcast_ori_axes(orientation_axes, P, K) if om is not None else None
    _require_axes_for_axis_mode(om, oa, packed, P, K)
    if oa is not None:
        oa = oa.astype(wire, copy=False)
    wp = _np.ascontiguousarray(wp); wo = _np.ascontiguousarray(wo)

    if self_collision_mode not in ("off", "final"):
        raise ValueError(
            f"solve_problems supports self_collision_mode off|final, got {self_collision_mode!r}"
            " (hard mode is a single-problem solve() feature)")
    if self_collision_mode == "final":
        # Uploaded BEFORE the solve now, not after: the sidecar's SDF/convex tables have to be
        # resident when solve_problems_batched calls the device check between LM and selection.
        _ensure_self_collision_sidecar()

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
        # Checkpoint 5D.14c: [P] uint32 SEMANTIC per-problem RNG roots. None => legacy
        # slot-derived fallback inside the kernel, which is NOT reproducible across batch
        # size/order; callers that care about determinism must supply them.
        if problem_seeds is None:
            _pseeds = _np.empty(0, dtype=_np.uint32)
        else:
            _pseeds = _np.ascontiguousarray(problem_seeds, dtype=_np.uint32).reshape(-1)
            if _pseeds.size != P:
                raise ValueError(f"problem_seeds must have length P={P}, got {_pseeds.size}")
        out = _solve_problems_raw(
            seeds_flat, pos, quat, packed, wp, wo, use_coarse, bool(run_coarse),
            float(position_tol), float(orientation_tol),
            1e-6, 1e-9, 0.35, int(coarse_iters), 5, 1, int(seed), 4,
            5e-3, int(lm_iters), int(stag_patience), float(stag_rel),
            int(num_solutions), pc, bool(return_all_candidates),
            str(problems_json_text), str(problem_set_name), int(problem_idx), sv._ws,
            base_p, base_q, base_diag, problem_seeds=_pseeds,
            orientation_modes=om, orientation_axes=oa,
            self_collision=(self_collision_mode == "final"),
            self_collision_margin=float(self_collision_margin),
            self_collision_eligible_tol=(-1.0 if self_collision_eligible_tol is None
                                         else float(self_collision_eligible_tol)), **_bu)
    finally:
        sv._exit()

    out["num_problems"] = P
    out["seeds_per_problem"] = S
    if floating_base:
        # The kernel is candidate-major: it returns the base per CANDIDATE [B,...] while
        # joint_config is the SELECTED [P,M,...]. They are paired ONLY through
        # b = p*S + selected_seed_ids[p,m]; that rule lives here, once, and nowhere else -- a
        # caller open-coding it is a caller who will one day drop the p*S stride and read a
        # neighbouring problem's base, which is a real base and so looks entirely plausible.
        # The raw candidate-major arrays stay available under private keys for tests and
        # debugging that genuinely reason per candidate.
        bp = _np.asarray(out["base_position"])          # [B,3] candidate-major
        bq = _np.asarray(out["base_quaternion"])        # [B,4]
        out["_base_position_candidates"] = bp
        out["_base_quaternion_candidates"] = bq
        sid = _np.asarray(out["selected_seed_ids"])     # [P,M], -1 == nothing valid
        pofs = (_np.arange(P, dtype=_np.int64) * S)[:, None]
        idx = pofs + _np.where(sid >= 0, sid, 0)        # [P,M]; the -1 slots are overwritten below
        gp, gq = bp[idx], bq[idx]                       # [P,M,3], [P,M,4]
        bad = sid < 0
        if bad.any():                                   # ALL-INVALID FILL: the first INPUT seed
            fill = _np.broadcast_to(pofs, sid.shape)
            gp = _np.where(bad[..., None], base_p_in[fill], gp)
            gq = _np.where(bad[..., None], base_q_in[fill], gq)
        out["base_position"] = _np.ascontiguousarray(gp)
        out["base_quaternion"] = _np.ascontiguousarray(gq)

        # Diagnostics, gathered to the same [P,M] selection. Counts are per CANDIDATE (they are
        # what that candidate's own LM did), so they gather by the same rule as the base itself.
        d = _np.asarray(out.pop("base_diag", base_diag)).reshape(B, 3)
        out["_base_diag_candidates"] = d
        dsel = d[idx]                                          # [P,M,3]
        out["base_updates_attempted"] = _np.ascontiguousarray(dsel[..., 0])
        out["base_updates_accepted"] = _np.ascontiguousarray(dsel[..., 1])
        out["base_numerical_failures"] = _np.ascontiguousarray(dsel[..., 2])

        # How far the selected base actually travelled from ITS OWN seed. Host-side: the seed is
        # base_p_in[b], the answer is gp -- no kernel state needed. Reported because "the base
        # moved" and "the base update helped" are different claims and only the second matters.
        seed_p, seed_q = base_p_in[idx], base_q_in[idx]        # [P,M,3], [P,M,4]
        out["base_translation_moved"] = _np.linalg.norm(gp - seed_p, axis=-1)
        # Angle of the relative rotation q_out * q_seed^-1, via |<q_out, q_seed>|: the abs folds
        # q and -q together (the same rotation), so a hemisphere flip is not read as a 180 deg move.
        dot = _np.abs(_np.sum(gq * seed_q, axis=-1)).clip(0.0, 1.0)
        out["base_rotation_moved"] = 2.0 * _np.arccos(dot)
        if bad.any():                                          # unfilled slots did not move
            out["base_translation_moved"] = _np.where(bad, 0.0, out["base_translation_moved"])
            out["base_rotation_moved"] = _np.where(bad, 0.0, out["base_rotation_moved"])
    if return_all_candidates:
        # E_phys per candidate for the debug arrays (matches the device selection metric).
        act = ((packed[:, None] >> _np.arange(K, dtype=_np.uint32)) & 1).astype(bool)   # [P,K]
        act_b = act[:, None, :]                                                          # [P,1,K]
        pe = out["all_position_errors"]; oe = out["all_orientation_errors"]
        out["all_cost_physical"] = (((pe / position_tol) ** 2 + (oe / orientation_tol) ** 2)
                                    * act_b).sum(axis=2)

    # -------------------------------------------------------------------------------------------
    # CHECKPOINT 7: self-collision now gates ELIGIBILITY on the device, BEFORE segmented top-M
    # selection, so this layer no longer re-checks or re-rejects the winner.
    #
    # Old flow:  solve -> select winner -> gather base -> check ONLY the winner -> fail if it hit.
    #            A colliding best candidate failed the whole problem even when a slightly worse
    #            collision-free candidate existed, and `checked` was 1 per problem.
    # New flow:  solve -> device sidecar over ALL B = P*S candidates -> self_collision_free[B]
    #            -> segmented_topM_kernel ANDs it into feasibility -> cand_better() ranks the
    #            survivors -> ONE gather of q/base/diagnostics from the final selected_seed_ids.
    #
    # cand_better() therefore remains the only ranking implementation; nothing here re-ranks.
    # The per-candidate configs never reach the host: d_lq feeds the sidecar kernel directly.
    # -------------------------------------------------------------------------------------------
    if self_collision_mode == "final":
        B_total = int(P * S)
        n_elig = int(out.pop("self_collision_eligible", B_total))
        n_check = int(out.pop("self_collision_checked", B_total))
        cfree_sel = _np.asarray(out.get("collision_free"))
        nfree = _np.asarray(out.get("num_collision_free"))
        n_free_total = int(nfree.sum()) if nfree is not None else 0

        # Three states, kept distinguishable rather than collapsed into one boolean:
        #   not checked   permanently selection-ineligible (non-finite q/errors, or above the
        #                 caller's acceptance tolerance). NOT counted as colliding.
        #   checked, colliding
        #   checked, collision-free
        out["self_collision_free"] = cfree_sel          # the SELECTED candidate(s)
        out["self_collision"] = dict(
            mode="final",
            selection="pre-selection (device eligibility gate)",
            candidates_total=B_total,
            candidates_collision_eligible=n_elig,
            candidates_checked=n_check,
            candidates_not_checked=B_total - n_check,   # ineligible, NOT "colliding"
            num_collision_free=n_free_total,
            num_colliding=n_check - n_free_total,       # checked AND hit, nothing else
            num_infeasible=(None if out.get("num_infeasible") is None
                            else _np.asarray(out["num_infeasible"]).tolist()),
            per_problem_collision_free=(None if nfree is None else nfree.tolist()),
            margin=float(self_collision_margin),
            eligible_tol=(None if self_collision_eligible_tol is None
                          else float(self_collision_eligible_tol)),
            native_collision_tolerance_m=abs(float(self_collision_margin)),
            semantics="native self-collision prefilter passed; MuJoCo remains authoritative")
    return out
