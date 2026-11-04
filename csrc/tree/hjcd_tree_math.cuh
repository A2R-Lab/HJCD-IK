#pragma once
// Multi-target residuals, orientation-mode Jacobians, row scaling and normal-equation assembly for
// fixed-base TREE models (humanoid-v1 M3). Ported from the humanoid branch (bbc53a9,
// csrc/kernel/hjcd_kernel.cu: target_residual, target_ori_jacobian, hjcd_twist_deadzone,
// hjcd_bounded_twist_aux, compute_row_scales_warp, weighted_cost_warp,
// accumulate_normal_equations_warp, eval_targets_warp, e_phys) with the numerics unchanged.
// Adaptations: GRiD 0.6.0 + hjcd_targets.cuh metadata instead of the old generated tables; joint
// axes from main's hjcd_joint_model.h; tools and bounded-twist specs are per-call arguments
// (no __constant__ state). Everything is warp-scoped: one warp per configuration, lane j = joint j.
#include "grid.cuh"
#include "hjcd_joint_model.h"
#include "hjcd_targets.cuh"
#include "glass.cuh"

namespace hjcd_tree {

constexpr int N = grid::NUM_JOINTS;
constexpr int K = hjcd_targets::NUM_TARGETS;
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;
enum : int { ORI_NONE = 0, ORI_AXIS = 1, ORI_FULL = 2, ORI_AXIS_BOUNDED_TWIST = 3 };

// ---- small math (bit-for-bit the humanoid helpers; quaternion ops delegate to GLASS as main does) ----
template<typename T> __device__ __forceinline__
void rotate_by_X(const T* __restrict__ X, const T* __restrict__ a, T* __restrict__ n) {
    n[0] = X[0]*a[0] + X[4]*a[1] + X[8] *a[2];
    n[1] = X[1]*a[0] + X[5]*a[1] + X[9] *a[2];
    n[2] = X[2]*a[0] + X[6]*a[1] + X[10]*a[2];
}
template<typename T> __device__ __forceinline__
void rotate_by_quat(const T* __restrict__ q, const T* __restrict__ a, T* __restrict__ n) {
    const T w = q[0], vx = q[1], vy = q[2], vz = q[3];
    const T tx = (T)2 * (vy*a[2] - vz*a[1]);
    const T ty = (T)2 * (vz*a[0] - vx*a[2]);
    const T tz = (T)2 * (vx*a[1] - vy*a[0]);
    n[0] = a[0] + w*tx + (vy*tz - vz*ty);
    n[1] = a[1] + w*ty + (vz*tx - vx*tz);
    n[2] = a[2] + w*tz + (vx*ty - vy*tx);
}
template<typename T> __device__ __forceinline__
void quat_mul_wxyz(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ out) {
    out[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
    out[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
    out[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
    out[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
}
template<typename T> __device__ __forceinline__ void mat_to_quat(const T* __restrict__ C, T* __restrict__ q) {
    glass::thread::rot_to_quat<T, glass::QuatLayout::wxyz, /*LDA=*/4>(C, q);
}
// World-frame residual log(q_goal (x) q_cur^-1); GLASS WORLD pins (q, q_des) = (q_goal, q_cur).
template<typename T> __device__ __forceinline__ void quat_err_rotvec(const T* q_cur, const T* q_goal, T* w) {
    glass::thread::quat_error<T, glass::QuatLayout::wxyz, glass::ErrorFrame::WORLD>(q_goal, q_cur, w);
}
template<typename T> __device__ __forceinline__ void joint_world_axis(const T* Ci, int j, T* ax) {
    T a[3]; hjcd_model::local_axis(j, a);
    for (int r = 0; r < 3; ++r) ax[r] = Ci[r]*a[0] + Ci[4+r]*a[1] + Ci[8+r]*a[2];
}

// ---- bounded twist: deadzone on the circle (zero inside [lo,hi], slope +1 outside) ----
template<typename T> __device__ __forceinline__
T twist_deadzone(T phi, T lo, T hi, T* __restrict__ slope) {
    const T PI_ = (T)3.14159265358979323846, TWO_PI_ = (T)6.28318530717958647692;
    const T centre = (T)0.5 * (lo + hi), half = (T)0.5 * (hi - lo);
    T off = phi - centre + PI_;
    off = off - TWO_PI_ * floor(off / TWO_PI_) - PI_;
    if (off <= -PI_) off = PI_;
    const T outside = fabs(off) - half;
    if (!(outside > (T)0)) { *slope = (T)0; return (T)0; }
    *slope = (T)1;
    return off > (T)0 ? outside : -outside;
}
// tw = (grip axis in target frame [3], lo, hi)
template<typename T> __device__ __forceinline__
void bounded_twist_aux(const T* __restrict__ X, const T* __restrict__ tgt_q, const T* __restrict__ a_local,
                       const T* __restrict__ tw, T* __restrict__ d, T* __restrict__ kdir, T* __restrict__ slope) {
    T h[3], t[3], b[3];
    rotate_by_quat<T>(tgt_q, a_local, d);
    rotate_by_quat<T>(tgt_q, tw, t);
    rotate_by_X<T>(X, tw, h);
    b[0] = d[1]*t[2] - d[2]*t[1]; b[1] = d[2]*t[0] - d[0]*t[2]; b[2] = d[0]*t[1] - d[1]*t[0];
    const T u = b[0]*h[0] + b[1]*h[1] + b[2]*h[2];
    const T w = t[0]*h[0] + t[1]*h[1] + t[2]*h[2];
    twist_deadzone<T>(atan2(u, w), tw[3], tw[4], slope);
    const T scale = u*u + w*w;
    if (!(scale > (T)1e-12)) { kdir[0] = kdir[1] = kdir[2] = (T)0; return; }
    T tg[3], bg[3];
    tg[0] = t[1]*h[2] - t[2]*h[1];  tg[1] = t[2]*h[0] - t[0]*h[2];  tg[2] = t[0]*h[1] - t[1]*h[0];
    bg[0] = b[1]*h[2] - b[2]*h[1];  bg[1] = b[2]*h[0] - b[0]*h[2];  bg[2] = b[0]*h[1] - b[1]*h[0];
    const T inv = (T)1 / scale;
    for (int r = 0; r < 3; ++r) kdir[r] = (u*tg[r] - w*bg[r]) * inv;
}

// ---- residual of one target: e_p = p* - p; e_R per mode; *ang = reported angle ----
template<typename T> __device__ __forceinline__
void target_residual(const T* __restrict__ X, const T* __restrict__ tgt_p, const T* __restrict__ tgt_q,
                     T* __restrict__ e_p, T* __restrict__ e_R, int mode, const T* __restrict__ a_local,
                     const T* __restrict__ tw, T* __restrict__ ang) {
    e_p[0] = tgt_p[0] - X[12]; e_p[1] = tgt_p[1] - X[13]; e_p[2] = tgt_p[2] - X[14];
    if (mode == ORI_AXIS_BOUNDED_TWIST) {
        T h[3], d[3], t[3], b[3];
        rotate_by_quat<T>(tgt_q, a_local, d);
        rotate_by_quat<T>(tgt_q, tw, t);
        rotate_by_X<T>(X, tw, h);
        b[0] = d[1]*t[2] - d[2]*t[1]; b[1] = d[2]*t[0] - d[0]*t[2]; b[2] = d[0]*t[1] - d[1]*t[0];
        T slope;
        const T phi = atan2(b[0]*h[0] + b[1]*h[1] + b[2]*h[2], t[0]*h[0] + t[1]*h[1] + t[2]*h[2]);
        const T v = twist_deadzone<T>(phi, tw[3], tw[4], &slope);
        const T half_psi = (T)0.5 * (phi - v);
        const T s_ = sin(half_psi);
        const T dq[4] = { cos(half_psi), s_*d[0], s_*d[1], s_*d[2] };
        T qeff[4]; quat_mul_wxyz<T>(dq, tgt_q, qeff);          // LEFT: world about d
        T qee[4]; mat_to_quat(X, qee);
        if (qee[0]*qeff[0] + qee[1]*qeff[1] + qee[2]*qeff[2] + qee[3]*qeff[3] < (T)0)
            for (int i = 0; i < 4; ++i) qee[i] = -qee[i];
        quat_err_rotvec(qee, qeff, e_R);
        *ang = sqrt(e_R[0]*e_R[0] + e_R[1]*e_R[1] + e_R[2]*e_R[2]);
        return;
    }
    if (mode == ORI_NONE) { e_R[0] = e_R[1] = e_R[2] = (T)0; *ang = (T)0; return; }
    if (mode == ORI_AXIS) {
        T n_cur[3], n_tgt[3];
        rotate_by_X<T>(X, a_local, n_cur);
        rotate_by_quat<T>(tgt_q, a_local, n_tgt);
        for (int r = 0; r < 3; ++r) e_R[r] = n_tgt[r] - n_cur[r];
        const T half_chord = (T)0.5 * sqrt(e_R[0]*e_R[0] + e_R[1]*e_R[1] + e_R[2]*e_R[2]);
        *ang = (T)2 * asin(half_chord < (T)1 ? half_chord : (T)1);
        return;
    }
    T qee[4]; mat_to_quat(X, qee);
    if (qee[0]*tgt_q[0] + qee[1]*tgt_q[1] + qee[2]*tgt_q[2] + qee[3]*tgt_q[3] < (T)0)
        for (int i = 0; i < 4; ++i) qee[i] = -qee[i];
    quat_err_rotvec(qee, tgt_q, e_R);
    *ang = sqrt(e_R[0]*e_R[0] + e_R[1]*e_R[1] + e_R[2]*e_R[2]);
}

// d(current orientation quantity)/dq for one joint with world angular axis jw (J = d prediction / dq).
template<typename T> __device__ __forceinline__
void target_ori_jacobian(int mode, const T* __restrict__ jw, const T* __restrict__ n, T* __restrict__ out,
                         const T* __restrict__ d, const T* __restrict__ kdir, T slope) {
    if (mode == ORI_AXIS_BOUNDED_TWIST) {
        const T f = ((T)1 - slope) * (kdir[0]*jw[0] + kdir[1]*jw[1] + kdir[2]*jw[2]);
        for (int r = 0; r < 3; ++r) out[r] = jw[r] - f*d[r];
        return;
    }
    if (mode == ORI_AXIS) {
        out[0] = jw[1]*n[2] - jw[2]*n[1];
        out[1] = jw[2]*n[0] - jw[0]*n[2];
        out[2] = jw[0]*n[1] - jw[1]*n[0];
    } else {
        out[0] = jw[0]; out[1] = jw[1]; out[2] = jw[2];
    }
}

// Per-call problem for one configuration (warp-shared).
template<typename T>
struct Problem {
    T tgt_p[K * 3], tgt_q[K * 4];   // desired poses (base frame == world for a fixed base)
    T wp[K], wo[K];
    int mode[K];
    T axis[K * 3];                  // target-local unit axis (AXIS / BOUNDED)
    T twist[K * 5];                 // bounded-twist spec: grip axis [3], lo, hi
    T tool[K * 16];                 // effective anchor->target tools, column-major (model or override)
    unsigned int active;
};

// Lane j's column of target k (6 scalars). Non-ancestor joints are exactly zero.
template<typename T> __device__ __forceinline__
bool target_column(const Problem<T>& P, int k, const T* __restrict__ jointX, const T* __restrict__ Xt,
                   int lane, T* c) {
    for (int r = 0; r < 6; ++r) c[r] = (T)0;
    if (!(lane < N && ((hjcd_targets::TARGET_ANCESTOR_MASK[k] >> lane) & 1u))) return false;
    const int md = P.mode[k];
    T nk[3] = {0, 0, 0}, dk[3] = {0, 0, 0}, kk[3] = {0, 0, 0}, sl = (T)0;
    if (md == ORI_AXIS || md == ORI_AXIS_BOUNDED_TWIST) rotate_by_X<T>(Xt, &P.axis[3*k], nk);
    if (md == ORI_AXIS_BOUNDED_TWIST) bounded_twist_aux<T>(Xt, &P.tgt_q[4*k], &P.axis[3*k], &P.twist[5*k], dk, kk, &sl);
    const T* Ci = &jointX[16 * lane];
    T ax[3]; joint_world_axis<T>(Ci, lane, ax);
    const bool prism = hjcd_model::is_prismatic(lane);
    const T rx = Xt[12] - Ci[12], ry = Xt[13] - Ci[13], rz = Xt[14] - Ci[14];
    c[0] = prism ? ax[0] : (ax[1]*rz - ax[2]*ry);
    c[1] = prism ? ax[1] : (ax[2]*rx - ax[0]*rz);
    c[2] = prism ? ax[2] : (ax[0]*ry - ax[1]*rx);
    T jw[3] = { prism ? (T)0 : ax[0], prism ? (T)0 : ax[1], prism ? (T)0 : ax[2] };
    target_ori_jacobian<T>(md, jw, nk, &c[3], dk, kk, sl);
    return true;
}

// Row-norm preconditioner s_{k,r} = 1/||J_{k,r}|| (1 where the row is ~0 or the target inactive).
template<typename T> __device__ __forceinline__
void compute_row_scales_warp(T* __restrict__ s_scale, const Problem<T>& P, const T* __restrict__ jointX,
                             const T* __restrict__ targetX) {
    const int lane = threadIdx.x & 31;
    for (int k = 0; k < K; ++k) {
        if (!((P.active >> k) & 1u)) { if (lane < 6) s_scale[6*k + lane] = (T)1; continue; }
        T c[6]; target_column<T>(P, k, jointX, &targetX[16*k], lane, c);
        for (int r = 0; r < 6; ++r) {
            T v = c[r] * c[r];
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(FULL_MASK, v, off);
            v = __shfl_sync(FULL_MASK, v, 0);
            if (lane == r) s_scale[6*k + r] = (v > (T)1e-18) ? rsqrt(v) : (T)1;
        }
        __syncwarp(FULL_MASK);
    }
    __syncwarp(FULL_MASK);
}

// A (N*N, row-major, lane i owns row i) and b assembled target by target; s_scale null => raw J^T W J.
// colnorm2[j] = sum_k w_p|Jv_kj|^2 + w_R|Jo_kj|^2 (unscaled; ORI_NONE => w_R = 0), for the null-column pin.
template<typename T> __device__ __forceinline__
void accumulate_normal_equations_warp(T* __restrict__ A, T* __restrict__ b, const Problem<T>& P,
                                      const T* __restrict__ jointX, const T* __restrict__ targetX,
                                      const T* __restrict__ e_pos, const T* __restrict__ e_ori,
                                      const T* __restrict__ s_scale, T* __restrict__ colnorm2) {
    const int lane = threadIdx.x & 31;
    for (int idx = lane; idx < N * N; idx += 32) A[idx] = (T)0;
    if (lane < N) { b[lane] = (T)0; if (colnorm2) colnorm2[lane] = (T)0; }
    __syncwarp(FULL_MASK);
    for (int k = 0; k < K; ++k) {
        if (!((P.active >> k) & 1u)) continue;
        const int md = P.mode[k];
        const T wp = P.wp[k];
        const T wo = (md == ORI_NONE) ? (T)0 : P.wo[k];
        if (wp == (T)0 && wo == (T)0) continue;
        T Wp[3], Wo[3];
        for (int r = 0; r < 3; ++r) {
            const T sp = s_scale ? s_scale[6*k + r] : (T)1, so = s_scale ? s_scale[6*k + 3 + r] : (T)1;
            Wp[r] = wp * sp * sp; Wo[r] = wo * so * so;
        }
        T c[6];
        const bool mine = target_column<T>(P, k, jointX, &targetX[16*k], lane, c);
        if (mine) {
            b[lane] += Wp[0]*c[0]*e_pos[3*k+0] + Wp[1]*c[1]*e_pos[3*k+1] + Wp[2]*c[2]*e_pos[3*k+2]
                     + Wo[0]*c[3]*e_ori[3*k+0] + Wo[1]*c[4]*e_ori[3*k+1] + Wo[2]*c[5]*e_ori[3*k+2];
            if (colnorm2) colnorm2[lane] += wp * (c[0]*c[0] + c[1]*c[1] + c[2]*c[2]) + wo * (c[3]*c[3] + c[4]*c[4] + c[5]*c[5]);
        }
        for (int j = 0; j < N; ++j) {
            T o[6];
            for (int r = 0; r < 6; ++r) o[r] = __shfl_sync(FULL_MASK, c[r], j);
            if (mine)
                A[lane * N + j] += Wp[0]*c[0]*o[0] + Wp[1]*c[1]*o[1] + Wp[2]*c[2]*o[2]
                                 + Wo[0]*c[3]*o[3] + Wo[1]*c[4]*o[4] + Wo[2]*c[5]*o[5];
        }
        __syncwarp(FULL_MASK);
    }
    __syncwarp(FULL_MASK);
}

// Null-column pin (bbc53a9): freeze joints that are not ancestors of any active target or whose
// unscaled column norm^2 <= eps * max over active-ancestor joints. Row/col -> identity, b -> 0.
template<typename T> __device__ __forceinline__ T null_column_eps();
template<> __device__ __forceinline__ double null_column_eps<double>() { return 1e-12; }
template<> __device__ __forceinline__ float  null_column_eps<float>()  { return 1e-10f; }

template<typename T> __device__ __forceinline__
unsigned int pin_null_columns_warp(T* __restrict__ A, T* __restrict__ b, const T* __restrict__ colnorm2,
                                   unsigned int active) {
    const int lane = threadIdx.x & 31;
    unsigned int act_joints = 0u;
    for (int k = 0; k < K; ++k) if ((active >> k) & 1u) act_joints |= hjcd_targets::TARGET_ANCESTOR_MASK[k];
    const bool anc = (lane < N) && ((act_joints >> lane) & 1u);
    const T cj = anc ? colnorm2[lane] : (T)0;
    T cmax = cj;
    for (int off = 16; off > 0; off >>= 1) cmax = fmax(cmax, __shfl_xor_sync(FULL_MASK, cmax, off));
    const bool scaled = (cmax > (T)0) && isfinite(cmax);
    const bool null_col = anc && scaled && (cj <= null_column_eps<T>() * cmax);
    const bool pin = (lane < N) && (!anc || null_col);
    const unsigned int pinned = __ballot_sync(FULL_MASK, pin);
    if (lane < N) {
        if (pin) { for (int j = 0; j < N; ++j) A[lane * N + j] = (T)0; A[lane * N + lane] = (T)1; b[lane] = (T)0; }
        else for (unsigned int p = pinned; p != 0u; p &= p - 1u) A[lane * N + (__ffs(p) - 1)] = (T)0;
    }
    __syncwarp(FULL_MASK);
    return pinned;
}

// ---- FK: template (block prologue) + per-warp joint world transforms + target frames ----
template<typename T> __device__ __forceinline__
void tree_fk_warp(const T* __restrict__ s_xhom, const T* __restrict__ q, T* __restrict__ local, T* __restrict__ world) {
    const int lane = threadIdx.x & 31;
    if (lane < N) grid::update_XmatHom_joint<T>(&local[16 * lane], s_xhom, lane, q[lane]);
    __syncwarp(FULL_MASK);
    for (int d = 0; d <= hjcd_targets::MAX_DEPTH; ++d) {
        if (lane < N && hjcd_targets::JOINT_DEPTH[lane] == d) {
            const int p = hjcd_targets::JOINT_PARENT[lane];
            if (p < 0) for (int m = 0; m < 16; ++m) world[16 * lane + m] = local[16 * lane + m];
            else glass::thread::gemm<T, 4, 4, 4>(T(1), &world[16 * p], &local[16 * lane], &world[16 * lane]);
        }
        __syncwarp(FULL_MASK);
    }
}
template<typename T> __device__ __forceinline__
void compose_targets_warp(const T* __restrict__ world, const T* __restrict__ tool, T* __restrict__ targetX) {
    const int lane = threadIdx.x & 31;
    if (lane < K) glass::thread::gemm<T, 4, 4, 4>(T(1), &world[16 * hjcd_targets::TARGET_ANCHOR_JID[lane]],
                                                   &tool[16 * lane], &targetX[16 * lane]);
    __syncwarp(FULL_MASK);
}

// Residuals of every active target into e_pos/e_ori, reported norms into pn/on (pn = |e_p|, on = angle).
template<typename T> __device__ __forceinline__
void eval_targets_warp(const Problem<T>& P, const T* __restrict__ targetX, T* __restrict__ e_pos,
                       T* __restrict__ e_ori, T* __restrict__ pn, T* __restrict__ on) {
    const int lane = threadIdx.x & 31;
    if (lane < K) {
        const int k = lane;
        T ep[3] = {0, 0, 0}, eR[3] = {0, 0, 0}, p_ = 0, o_ = 0;
        if ((P.active >> k) & 1u) {
            target_residual<T>(&targetX[16*k], &P.tgt_p[3*k], &P.tgt_q[4*k], ep, eR, P.mode[k],
                               &P.axis[3*k], &P.twist[5*k], &o_);
            p_ = sqrt(ep[0]*ep[0] + ep[1]*ep[1] + ep[2]*ep[2]);
        }
        for (int r = 0; r < 3; ++r) { e_pos[3*k + r] = ep[r]; e_ori[3*k + r] = eR[r]; }
        pn[k] = p_; on[k] = o_;
    }
    __syncwarp(FULL_MASK);
}

// Row-scaled weighted cost (the LM's acceptance/gain-ratio objective, frozen scales).
template<typename T> __device__ __forceinline__
T weighted_cost(const Problem<T>& P, const T* e_pos, const T* e_ori, const T* s_scale) {
    T c = (T)0;
    for (int k = 0; k < K; ++k) {
        if (!((P.active >> k) & 1u)) continue;
        for (int r = 0; r < 3; ++r) {
            const T sp = s_scale[6*k + r], so = s_scale[6*k + 3 + r];
            c += P.wp[k] * sp * sp * e_pos[3*k + r] * e_pos[3*k + r] + P.wo[k] * so * so * e_ori[3*k + r] * e_ori[3*k + r];
        }
    }
    return c;
}
template<typename T> __device__ __forceinline__
void worst_active_errors(const Problem<T>& P, const T* pn, const T* on, T* mp, T* mo) {
    *mp = (T)0; *mo = (T)0;
    for (int k = 0; k < K; ++k) {
        if (!((P.active >> k) & 1u)) continue;
        if (P.wp[k] > (T)0 && pn[k] > *mp) *mp = pn[k];
        if (P.wo[k] > (T)0 && on[k] > *mo) *mo = on[k];
    }
}
template<typename T> __device__ __forceinline__
bool all_active_converged(const Problem<T>& P, const T* pn, const T* on, T eps_p, T eps_o) {
    if (P.active == 0u) return false;
    for (int k = 0; k < K; ++k) {
        if (!((P.active >> k) & 1u)) continue;
        if (P.wp[k] > (T)0 && !(pn[k] <= eps_p)) return false;
        if (P.wo[k] > (T)0 && !(on[k] <= eps_o)) return false;
    }
    return true;
}
template<typename T> __device__ __forceinline__
T e_phys(const Problem<T>& P, const T* pn, const T* on, T eps_p, T eps_o) {
    T e = (T)0;
    for (int k = 0; k < K; ++k) {
        if (!((P.active >> k) & 1u)) continue;
        const T rp = pn[k] / eps_p, ro = on[k] / eps_o;
        e += rp * rp + ro * ro;
    }
    return e;
}

// ---- floating base (external; bbc53a9 hjcd_settings.h / hjcd_kernel.cu, numerics unchanged) ----
template<typename T> __device__ __forceinline__
void quat_rotate_inv(const T* __restrict__ q, const T* __restrict__ v, T* __restrict__ out) {   // R(q)^T v
    const T qv[3] = { -q[1], -q[2], -q[3] };
    T t[3] = { qv[1]*v[2] - qv[2]*v[1], qv[2]*v[0] - qv[0]*v[2], qv[0]*v[1] - qv[1]*v[0] };
    t[0] += t[0]; t[1] += t[1]; t[2] += t[2];
    const T c[3] = { qv[1]*t[2] - qv[2]*t[1], qv[2]*t[0] - qv[0]*t[2], qv[0]*t[1] - qv[1]*t[0] };
    for (int i = 0; i < 3; ++i) out[i] = v[i] + q[0] * t[i] + c[i];
}
template<typename T> __device__ __forceinline__ void quat_to_mat3(const T* __restrict__ q, T* __restrict__ R) {   // column-major
    const T w = q[0], x = q[1], y = q[2], z = q[3];
    R[0] = (T)1 - (T)2*(y*y + z*z);  R[1] = (T)2*(x*y + w*z);         R[2] = (T)2*(x*z - w*y);
    R[3] = (T)2*(x*y - w*z);         R[4] = (T)1 - (T)2*(x*x + z*z);  R[5] = (T)2*(y*z + w*x);
    R[6] = (T)2*(x*z + w*y);         R[7] = (T)2*(y*z - w*x);         R[8] = (T)1 - (T)2*(x*x + y*y);
}
template<typename T> __device__ __forceinline__ void quat_from_rotvec(const T* __restrict__ w, T* __restrict__ q) {
    const T th = sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
    if (th < (T)1e-12) { q[0] = (T)1; q[1] = (T)0.5*w[0]; q[2] = (T)0.5*w[1]; q[3] = (T)0.5*w[2]; }
    else { const T s = sin((T)0.5*th) / th; q[0] = cos((T)0.5*th); q[1] = w[0]*s; q[2] = w[1]*s; q[3] = w[2]*s; }
    const T n = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    for (int i = 0; i < 4; ++i) q[i] /= n;
}
// ^B T* = (^W T_B)^-1 ^W T*:  p = R_b^T (p* - p_b),  q = q_b^-1 (x) q*  (unit q_b => conjugate).
template<typename T> __device__ __forceinline__
void world_target_to_base(const T* __restrict__ tgt_p_w, const T* __restrict__ tgt_q_w, const T* __restrict__ base_p,
                          const T* __restrict__ base_q, T* __restrict__ out_p, T* __restrict__ out_q) {
    const T d[3] = { tgt_p_w[0] - base_p[0], tgt_p_w[1] - base_p[1], tgt_p_w[2] - base_p[2] };
    quat_rotate_inv(base_q, d, out_p);
    const T qinv[4] = { base_q[0], -base_q[1], -base_q[2], -base_q[3] };
    quat_mul_wxyz(qinv, tgt_q_w, out_q);
}

}  // namespace hjcd_tree
