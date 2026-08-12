"""Host-side reference for the floating-base update (Architecture B).

This is the ORACLE the GPU implementation is tested against, and the executable
statement of the conventions it must match. It is pure numpy and needs no CUDA, which is
deliberate: the base update's mathematics is independent of the forward kinematics, so it
can be validated exhaustively on a CPU-only box (see below).

WHY THE BASE NEEDS NO FLOATING-BASE FK
--------------------------------------
For kinematics, the floating base is a RIGID TRANSFORM on top of the fixed-base FK, not a
chain element:

    x_i(p_b, q_b, q_j) = R_b * fk_i(q_j) + p_b                            (1)

so `fk_i(q_j)` enters only through the contact point. Every function here therefore takes
the contact points as data and never calls a forward kinematics. That is also why HJCD keeps
its fixed-base 29-joint codegen: GRiD will not emit `ee_pose_inner_{thread,warp}` for a
floating-base robot at all (GRiDCodeGenerator/algorithms/_eepose_gradient_hessian.py:2823),
and it does not need to.

CONVENTIONS (the GPU code must match these EXACTLY)
--------------------------------------------------
state          x = (p_b in R^3, q_b in S^3 as WXYZ unit, q_j in R^29)
               Quaternions are WXYZ to match the rest of hjcdik (target_quaternions).
perturbation   WORLD-frame LEFT, about the BASE ORIGIN:

                   p_b+ = p_b + alpha*dp,     R_b+ = exp([alpha*dphi]x) * R_b        (2)

               The choice matters: rotating about the base origin (rather than the world
               origin) is what produces the (x_i - p_b) lever arm in (3). A body-frame RIGHT
               perturbation would instead give J = [I, -R_b [fk_i]x].
base Jacobian  differentiating (1) under (2), with exp([dphi]x) ~= I + [dphi]x:

                   dx_i = dp - [x_i - p_b]x dphi
                   J_b,i = [ I3 , -[x_i - p_b]x ]   in R^{3x6}                        (3)

               Note J_b depends on the joints ONLY through x_i -- there is no Jacobian of
               the robot in it. Verified against finite differences to 1.8e-10 on the real
               G1 FK, and to ~1e-11 here (tests/test_base_update.py).
residual       r_i = x*_i - x_i   (target minus current), stacked over K targets.
normal eqs     H = J^T W J + lambda*I,   b = J^T W r,   H dxi = b                     (4)

               lambda*I (Tikhonov), NOT lambda*diag(H) (Marquardt). The GPU side must
               therefore instantiate
                   glass::warp::posv<T, 6, 1, REGULARIZE=true, CHECK=true, REG_DIAG=false>
               REG_DIAG=false is the `A[i] += rho` branch = rho*I
               (GLASS/src/base/L3/trsm.cuh:237-245). REG_DIAG=true would silently be
               Marquardt and would not match this reference.

ORIENTATION IS NOT USED (yet)
-----------------------------
The update is POSITION-DRIVEN: only position residuals enter (4). This is a deliberate
first step, not an oversight. The orientation block is available and simple -- a world-frame
left perturbation rotates every end-effector frame by the same dphi, so J_ori,k = [0, I] up
to the right-Jacobian of the Log residual parameterization -- but making Jr^-1 consistent
with the kernel's hemisphere-flipped `quat_err_rotvec` deserves its own validation. The
translation block is exactly orthogonal to orientation (d e_R / d p_b = 0), so adding it
later cannot disturb what is here.

INACTIVE TARGETS
----------------
Handled by ZERO WEIGHT, matching the kernel's "a zero-weight channel is don't care"
semantics: an inactive target contributes nothing to H, b, or the cost.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = [
    "skew", "so3_exp", "quat_to_mat", "mat_to_quat", "quat_mul", "quat_normalize",
    "contact_points_world", "base_jacobian", "base_normal_equations",
    "solve_base_update", "clip_base_step", "apply_base_update", "base_cost",
    "base_update_step", "BaseUpdateConfig", "MIN_DAMPING",
]

# Smallest damping an escalation may step to. Matches the LM's lambda_min
# (hjcd_kernel.cu:1578) so both solvers escalate over the same range.
MIN_DAMPING = 1e-12


# ----------------------------------------------------------------------------------------
# SO(3) / quaternion primitives
# ----------------------------------------------------------------------------------------

def skew(v) -> np.ndarray:
    """[v]x, the matrix with [v]x a = v cross a."""
    x, y, z = (float(c) for c in np.asarray(v, dtype=float).reshape(3))
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(w) -> np.ndarray:
    """exp([w]x) -- Rodrigues. Small-angle branch keeps this exact to first order at w->0.

    There is no SO(3) exp on the GPU side today (only the Log, via quat_err_rotvec), so M4
    must add one; this is its reference.
    """
    w = np.asarray(w, dtype=float).reshape(3)
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        # I + [w]x is the exact first-order term; the quadratic term is O(th^2) and below
        # double precision here. Avoids 0/0 in w/th.
        return np.eye(3) + skew(w)
    K = skew(w / th)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def quat_normalize(q) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        raise ValueError(f"quaternion is degenerate (norm {n:.3e})")
    return q / n


def quat_to_mat(q) -> np.ndarray:
    """WXYZ unit quaternion -> 3x3 rotation matrix."""
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat(R) -> np.ndarray:
    """3x3 rotation -> WXYZ unit quaternion. Branch on the largest diagonal term.

    The branching is not cosmetic: the naive w = sqrt(1+trace)/2 form loses all precision
    when trace -> -1 (a 180-degree rotation), which the rotation-only tests hit.
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                      (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                      0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return quat_normalize(q)


def quat_mul(a, b) -> np.ndarray:
    """Hamilton product, WXYZ. quat_mul(a, b) is the rotation 'apply b, then a'."""
    aw, ax, ay, az = np.asarray(a, dtype=float).reshape(4)
    bw, bx, by, bz = np.asarray(b, dtype=float).reshape(4)
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


# ----------------------------------------------------------------------------------------
# The base update
# ----------------------------------------------------------------------------------------

def contact_points_world(c_base, p_b, q_b) -> np.ndarray:
    """(K,3) world contacts from (K,3) BASE-frame contacts: x_i = R_b c_i + p_b. Eq. (1)."""
    c_base = np.asarray(c_base, dtype=float).reshape(-1, 3)
    R = quat_to_mat(q_b)
    return c_base @ R.T + np.asarray(p_b, dtype=float).reshape(1, 3)


def base_jacobian(x_world, p_b) -> np.ndarray:
    """(3K, 6) stacked base Jacobian, Eq. (3). Rows are [dx_i/dp_b | dx_i/dphi_b].

    No weighting or masking here -- those enter through W in base_normal_equations, so this
    stays a pure statement of the geometry and can be finite-difference checked directly.
    """
    x = np.asarray(x_world, dtype=float).reshape(-1, 3)
    p_b = np.asarray(p_b, dtype=float).reshape(3)
    K = x.shape[0]
    J = np.zeros((3 * K, 6))
    for i in range(K):
        J[3 * i:3 * i + 3, 0:3] = np.eye(3)
        J[3 * i:3 * i + 3, 3:6] = -skew(x[i] - p_b)
    return J


def _weight_vector(K: int, weights=None, active_mask=None) -> np.ndarray:
    """(K,) per-target weights with inactive targets forced to 0.

    active_mask may be a bitmask int (bit k = target k, matching the kernel's `active`) or a
    boolean/array-like of length K.
    """
    w = np.ones(K) if weights is None else np.broadcast_to(
        np.asarray(weights, dtype=float), (K,)).astype(float).copy()
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    if active_mask is None:
        return w
    if isinstance(active_mask, (int, np.integer)):
        if int(active_mask) >> K:
            raise ValueError(f"active_mask sets bits above K={K}: {int(active_mask):#x}")
        act = np.array([bool((int(active_mask) >> k) & 1) for k in range(K)])
    else:
        act = np.asarray(active_mask).reshape(-1).astype(bool)
        if act.size != K:
            raise ValueError(f"active_mask must have {K} entries, got {act.size}")
    w[~act] = 0.0
    return w


def damping_matrix(scale_p: float = 1.0, scale_R: float = 1.0) -> np.ndarray:
    """D = diag(s_p^-2 I3, s_R^-2 I3) (6,6) -- the metric the damping is measured in.

    This is what makes lambda dimensionless and the two blocks comparable. H mixes units: its
    translation columns are dimensionless (I3) while its rotation columns carry metres (the lever
    arms [x-p_b]x), so a bare lambda*I adds the same absolute number to quantities that are not
    the same kind of quantity, and what it does depends on whether you measure the robot in metres
    or millimetres. With D, the penalty is lambda*(||dp||^2/s_p^2 + ||dphi||^2/s_R^2): each block
    is measured against its OWN characteristic scale (s_p metres, s_R radians) and the sum is
    dimensionless.

    s_p = s_R = 1 gives D = I exactly, i.e. the plain Tikhonov shift, which is a legitimate
    choice for G1 (a ~0.5 m lever arm makes 1 rad and 1 m comparable in contact displacement)
    and is the default -- but it is now a CHOICE of scales, not an accident of units.
    """
    sp, sR = float(scale_p), float(scale_R)
    if not (sp > 0.0 and np.isfinite(sp)):
        raise ValueError(f"scale_p must be a positive finite length, got {scale_p}")
    if not (sR > 0.0 and np.isfinite(sR)):
        raise ValueError(f"scale_R must be a positive finite angle, got {scale_R}")
    return np.diag(np.array([1.0 / sp**2] * 3 + [1.0 / sR**2] * 3))


def base_normal_equations(J, r, weights=None, active_mask=None,
                          damping: float = 0.0, scale_p: float = 1.0,
                          scale_R: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """H = J^T W J + lambda*D (6,6) and b = J^T W r (6,). Eq. (4).

    W is block-diagonal with w_k repeated 3x (one weight per TARGET, not per row), which is
    the same semantics as the kernel's s_wp[k]. D is `damping_matrix(scale_p, scale_R)`.

    THE ORACLE CONTRACT: the kernel (`base_update_warp`) forms this same H_lambda = H + lambda*D
    and factors it with REGULARIZE=false, so posv adds nothing of its own. Host and device
    therefore implement one algorithm. Do not reintroduce a shift here (or a REG_DIAG=true there)
    without changing both: an oracle that solves a different system silently stops being an oracle.
    """
    J = np.asarray(J, dtype=float)
    r = np.asarray(r, dtype=float).reshape(-1)
    if J.shape[0] != r.size or J.shape[1] != 6:
        raise ValueError(f"J must be (3K,6) and r (3K,), got {J.shape} and {r.shape}")
    K = r.size // 3
    w = np.repeat(_weight_vector(K, weights, active_mask), 3)      # (3K,)
    Jw = J * w[:, None]
    H = J.T @ Jw + float(damping) * damping_matrix(scale_p, scale_R)
    b = Jw.T @ r
    return H, b


def solve_base_update(H, b) -> Tuple[np.ndarray, bool]:
    """Solve H dxi = b by Cholesky. Returns (dxi, ok); dxi is zeros when not SPD.

    Cholesky (not lstsq/pinv) on purpose: it mirrors glass::warp::posv, so a matrix that
    fails here fails there too, and the failure path is exercised by the same tests. A
    non-SPD H is reported, never papered over with a pseudo-inverse the GPU cannot compute.
    """
    H = np.asarray(H, dtype=float).reshape(6, 6)
    b = np.asarray(b, dtype=float).reshape(6)
    try:
        L = np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        return np.zeros(6), False
    y = np.linalg.solve(L, b)
    dxi = np.linalg.solve(L.T, y)
    if not np.all(np.isfinite(dxi)):
        return np.zeros(6), False
    return dxi, True


def clip_base_step(dxi, max_translation: float, max_rotation: float) -> np.ndarray:
    """Clip translation and rotation SEPARATELY, each by its own norm.

    Separately because they carry different units (m vs rad); one joint norm would make the
    clip depend on an arbitrary choice of length scale. Scaling each block by a factor
    preserves its DIRECTION -- clipping components independently would not.
    """
    dxi = np.asarray(dxi, dtype=float).reshape(6).copy()
    nt = float(np.linalg.norm(dxi[0:3]))
    nr = float(np.linalg.norm(dxi[3:6]))
    if max_translation is not None and nt > max_translation > 0.0:
        dxi[0:3] *= max_translation / nt
    if max_rotation is not None and nr > max_rotation > 0.0:
        dxi[3:6] *= max_rotation / nr
    return dxi


def apply_base_update(p_b, q_b, dxi, step_scale: float = 1.0,
                      position_lower=None, position_upper=None) -> Tuple[np.ndarray, np.ndarray]:
    """(p_b, q_b) + dxi -> (p_b+, q_b+) per Eq. (2), with bounds and renormalization."""
    p_b = np.asarray(p_b, dtype=float).reshape(3)
    dxi = np.asarray(dxi, dtype=float).reshape(6)
    a = float(step_scale)
    p_new = p_b + a * dxi[0:3]
    if position_lower is not None or position_upper is not None:
        lo = -np.inf if position_lower is None else np.asarray(position_lower, float).reshape(3)
        hi = +np.inf if position_upper is None else np.asarray(position_upper, float).reshape(3)
        if np.any(np.asarray(lo) > np.asarray(hi)):
            raise ValueError(f"base position bounds are inverted: {lo} > {hi}")
        p_new = np.clip(p_new, lo, hi)
    # R+ = exp([a*dphi]x) R  -- LEFT multiply. Renormalize every step: the quaternion is the
    # stored state, so drift compounds across repeated updates.
    R_new = so3_exp(a * dxi[3:6]) @ quat_to_mat(q_b)
    return p_new, quat_normalize(mat_to_quat(R_new))


def base_cost(x_world, targets, weights=None, active_mask=None) -> float:
    """sum_k w_k ||x*_k - x_k||^2 over ACTIVE targets -- the quantity the update must reduce."""
    x = np.asarray(x_world, dtype=float).reshape(-1, 3)
    t = np.asarray(targets, dtype=float).reshape(-1, 3)
    w = _weight_vector(x.shape[0], weights, active_mask)
    return float(np.sum(w * np.sum((t - x) ** 2, axis=1)))


class BaseUpdateConfig:
    """Controls for one base update. Defaults are conservative by instruction; they are to
    be tuned only on benchmark evidence (M7), never by intuition."""

    def __init__(self, damping: float = 1e-3, step_scale: float = 1.0,
                 max_translation_step: float = 0.05, max_rotation_step: float = 0.10,
                 position_lower=None, position_upper=None,
                 accept_only_on_improvement: bool = True,
                 damping_escalation: float = 10.0, max_damping: float = 1e6,
                 scale_p: float = 1.0, scale_R: float = 1.0):
        self.damping = float(damping)
        # Validated here (not at first use) so a bad scale is rejected before any solving, and
        # rejected identically to the binding's check -- these must stay the same contract.
        damping_matrix(scale_p, scale_R)
        self.scale_p = float(scale_p)
        self.scale_R = float(scale_R)
        self.step_scale = float(step_scale)
        self.max_translation_step = max_translation_step
        self.max_rotation_step = max_rotation_step
        self.position_lower = position_lower
        self.position_upper = position_upper
        self.accept_only_on_improvement = bool(accept_only_on_improvement)
        self.damping_escalation = float(damping_escalation)
        self.max_damping = float(max_damping)


def base_update_step(c_base, targets, p_b, q_b, cfg: Optional[BaseUpdateConfig] = None,
                     weights=None, active_mask=None) -> dict:
    """One damped-least-squares base update. Returns a diagnostics dict.

    `c_base` is the (K,3) BASE-frame contact points, i.e. fk_i(q_j) -- constant here, since
    the joints do not move during a base update. That is the whole of Architecture B's
    alternation: joints are held while the base takes a Gauss-Newton step, and vice versa.

    Rejection is on the COST, never on the linear system succeeding: a solvable step that
    makes the pose worse (linearization error, an over-long step) is discarded.
    """
    cfg = cfg or BaseUpdateConfig()
    c_base = np.asarray(c_base, dtype=float).reshape(-1, 3)
    targets = np.asarray(targets, dtype=float).reshape(-1, 3)
    if targets.shape != c_base.shape:
        raise ValueError(f"targets {targets.shape} must match contacts {c_base.shape}")

    x0 = contact_points_world(c_base, p_b, q_b)
    cost0 = base_cost(x0, targets, weights, active_mask)
    r = (targets - x0).reshape(-1)
    J = base_jacobian(x0, p_b)

    lam, failures, dxi, ok = cfg.damping, 0, np.zeros(6), False
    while True:
        H, b = base_normal_equations(J, r, weights, active_mask, damping=lam,
                                     scale_p=cfg.scale_p, scale_R=cfg.scale_R)
        dxi, ok = solve_base_update(H, b)
        if ok:
            break
        # Not SPD: escalate damping and retry. H + lambda*D is SPD for any lambda > 0 (H is PSD
        # and D is PD), so this terminates on the first escalation off zero; the cap stops an
        # infinite loop on a NaN H.
        # The MIN_DAMPING floor is load-bearing, not cosmetic: escalating multiplicatively
        # from lambda == 0 stays at 0 forever and never reaches the cap -- an infinite loop
        # in the kernel. Mirrors the LM's own lambda_min (hjcd_kernel.cu:1578).
        failures += 1
        lam = max(lam * cfg.damping_escalation, MIN_DAMPING)
        if lam > cfg.max_damping:
            break

    if not ok:
        return dict(accepted=False, p_b=np.asarray(p_b, float).reshape(3).copy(),
                    q_b=quat_normalize(q_b), cost_before=cost0, cost_after=cost0,
                    dxi=np.zeros(6), damping=lam, numerical_failure=True,
                    failures=failures, clipped=False)

    dxi_clipped = clip_base_step(dxi, cfg.max_translation_step, cfg.max_rotation_step)
    clipped = not np.allclose(dxi_clipped, dxi, rtol=0, atol=0)
    p_new, q_new = apply_base_update(p_b, q_b, dxi_clipped, cfg.step_scale,
                                     cfg.position_lower, cfg.position_upper)
    x1 = contact_points_world(c_base, p_new, q_new)
    cost1 = base_cost(x1, targets, weights, active_mask)

    accepted = (cost1 < cost0) if cfg.accept_only_on_improvement else True
    if not accepted:
        p_new, q_new, cost1 = (np.asarray(p_b, float).reshape(3).copy(),
                               quat_normalize(q_b), cost0)
    return dict(accepted=bool(accepted), p_b=p_new, q_b=q_new, cost_before=cost0,
                cost_after=cost1, dxi=dxi_clipped, damping=lam, numerical_failure=False,
                failures=failures, clipped=bool(clipped))
