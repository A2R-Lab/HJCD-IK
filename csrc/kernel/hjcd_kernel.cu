#include "kernel/hjcd_kernel.h"
#include "kernel/hjcd_settings.h"
#include "kernel/util.h"
#include "kernel/cuda_memory.h"
#include "kernel/device_utils.cuh"

#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <chrono>
#include <fstream>
#include <limits>
#include <type_traits>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include <mutex>

namespace {
// Stop flags/constants are shared. Protect native callers too, while Python releases its GIL.
std::recursive_mutex& solver_mutex() {
    static std::recursive_mutex mutex;
    return mutex;
}

void check_grid_initialization(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("GRiD CUDA initialization failed at ") +
            (operation ? operation : "unknown operation") + ": " + cudaGetErrorString(error));
}
}

template<typename T>
grid::robotModel<T>* cached_robot_model() {
    std::lock_guard<std::recursive_mutex> lock(solver_mutex());
    int device = 0;
    CUDA_OK(cudaGetDevice(&device));
    // Immutable generated models live as long as the CUDA context, shared by all solver stages.
    static std::unordered_map<int, grid::robotModel<T>*> models;
    auto& model = models[device];
    if (!model) {
        const char* operation = nullptr;
        const auto error = grid::init_robotModel_checked<T>(&model, &operation);
        check_grid_initialization(error, operation);
    }
    return model;
}

// Collision checking (grid_collision: URDF-driven spheres baked into grid.cuh)
#include <nlohmann/json.hpp>
#include "kernel/grid_env.cuh"   // requires grid.cuh (via hjcd_settings.h) already included above

enum : int {
    N = grid::NUM_JOINTS
};
extern "C" int grid_num_joints() { return N; }
extern "C" bool grid_has_collision() {
#if defined(HJCD_HAS_COLLISION)
    return true;
#else
    return false;
#endif
}

constexpr int FLANGE_IDX = N + 1;
constexpr int EE_IDX     = N;
constexpr int NX         = FLANGE_IDX + 1;

__constant__ double2 c_joint_limits[N];

void init_joint_limits_from_grid()
{
    std::lock_guard<std::recursive_mutex> lock(solver_mutex());
    int device = 0;
    CUDA_OK(cudaGetDevice(&device));
    static std::unordered_set<int> initialized;
    if (initialized.count(device)) return;
    hjcd::DeviceAllocations allocations;
    double* d_limits = nullptr;
    const char* operation = nullptr;
    const auto error = grid::init_joint_limits_checked<double>(&d_limits, &operation);
    check_grid_initialization(error, operation);
    allocations.adopt(d_limits);

    std::vector<double> h_limits(2 * N);
    CUDA_OK(cudaMemcpy(h_limits.data(), d_limits,
                       sizeof(double) * 2 * N, cudaMemcpyDeviceToHost));
    allocations.clear();

    std::vector<double2> packed(N);
    for (int j = 0; j < N; ++j) {
        double lo = h_limits[j];
        double hi = h_limits[j + N];

        if (!std::isfinite(lo)) lo = -PI;
        if (!std::isfinite(hi)) hi =  PI;
        if (lo > hi) std::swap(lo, hi);
        if (lo == hi) { lo -= 1e-9; hi += 1e-9; }

        packed[j] = make_double2(lo, hi);
    }

    CUDA_OK(cudaMemcpyToSymbol(c_joint_limits, packed.data(),
                               sizeof(double2) * N));
    initialized.insert(device);
}

template<typename T>
__device__ void perturb_joint_config(T* s_x, int global_problem, T sigma_frac = (T)0.05) {
    const uint32_t base = 911u;
    uint32_t s = make_seed(base, global_problem, 0, threadIdx.x);

#pragma unroll
    for (int j = 0; j < N; ++j) {
        uint32_t sj = wanghash(s ^ (uint32_t)j * 0x9E3779B9u);

        float low = (float)c_joint_limits[j].x;
        float hi = (float)c_joint_limits[j].y;
        float range = hi - low;

        float step = (float)sigma_frac * range * gauss01(sj);

        s_x[j] = (T)clamp_val<float>((float)s_x[j] + step, low, hi);
    }
}

// MATH HELPERS
// 4x4 col-major homogeneous transform -> unit wxyz quaternion. Thin alias over GLASS's
// Shepperd rot_to_quat with LDA=4 (reads the 3x3 rotation block of C in place — R3 of the
// 2026-07-30 GLASS request wave). GLASS canonicalizes the double cover to w >= 0 (benign:
// downstream comparisons either cover-fold against a reference or use fabs(w)).
template<typename T>
__device__ __forceinline__
void mat_to_quat(const T* __restrict__ C, T* __restrict__ q) {
    glass::thread::rot_to_quat<T, glass::block::QuatLayout::wxyz, /*LDA=*/4>(C, q);
}

// World-frame orientation residual w = log(q_goal ⊗ q_cur⁻¹), the rotation vector paired
// with the world-frame geometric Jacobian. Thin alias over GLASS's frame-tagged quat_error
// (R2 of the 2026-07-30 request wave). ⚠ OPERAND ORDER: GLASS pins WORLD as
// e = log(q ⊗ q_des⁻¹), so the drop-in is (q, q_des) = (q_goal, q_cur) — arguments SWAPPED
// relative to this helper's (q_cur, q_goal) order. Inputs must be unit (ours are: qee comes
// from rot_to_quat, q_goal from the normalized target).
template<typename T>
__device__ __forceinline__ void quat_err_rotvec(const T* q_cur, const T* q_goal, T* w_err3) {
    glass::thread::quat_error<T, glass::block::QuatLayout::wxyz, glass::block::ErrorFrame::WORLD>(
        q_goal, q_cur, w_err3);
}

// In-place unit-normalize a 3-vector (no-op for near-zero input).
template<typename T>
__device__ __forceinline__ void normalize_vec3(T* vec) {
    const T norm = glass::thread::nrm2<T, 3>(vec);
    if (norm > 1e-6) { vec[0] /= norm; vec[1] /= norm; vec[2] /= norm; }
}

// Pose errors of a standalone 16-cell column-major EE transform (offset 0). The coarse
// lane-parallel candidates keep each candidate's EE pose in a per-lane register buffer; the
// chain-indexed overloads below just select slot EE_IDX of a full frame array.
template<typename T>
__device__ __forceinline__ T compute_pos_err_at(const T* ee16, const T* target_pose) {
    const T dx = ee16[12] - target_pose[0];
    const T dy = ee16[13] - target_pose[1];
    const T dz = ee16[14] - target_pose[2];
    return sqrt(dx * dx + dy * dy + dz * dz);
}

// Scalar orientation error = geodesic angle. glass::block::quat_angle is frame-invariant and folds
// the double cover internally (no manual dot-sign flip needed).
template<typename T>
__device__ __forceinline__ T compute_ori_err_at(const T* ee16, const T* q_goal) {
    T qee[4];
    mat_to_quat(ee16, qee);
    return glass::block::quat_angle<T, glass::block::QuatLayout::wxyz>(qee, q_goal);
}

template<typename T>
__device__ __forceinline__ T compute_pos_err(const T* C, const T* target_pose) {
    return compute_pos_err_at(&C[EE_IDX * 16], target_pose);
}

template<typename T>
__device__ __forceinline__ T compute_ori_err(const T* C, const T* q_goal) {
    return compute_ori_err_at(&C[EE_IDX * 16], q_goal);
}

// Pack a 16-cell EE transform as [x, y, z, qw, qx, qy, qz] (rot_to_quat returns a unit quaternion).
template<typename T>
__device__ __forceinline__ void store_ee_pose7(const T* ee16, T* pose7) {
    T q[4];
    mat_to_quat(ee16, q);
    pose7[0] = ee16[12]; pose7[1] = ee16[13]; pose7[2] = ee16[14];
    pose7[3] = q[0]; pose7[4] = q[1]; pose7[5] = q[2]; pose7[6] = q[3];
}

// Greedy coordinate-descent step damping: 0.75..1.0 of the geometric angle, decaying with k.
template<typename T>
__device__ __forceinline__ T coarse_step(T theta, int k, int k_max) {
    const T delta = 0.75 + 0.25 * (1.0 - T(k) / T(k_max));
    return clamp_step_angle(theta * delta);
}

// SOLVE
template<typename T>
__device__ T solve_pos(const T* s_jointXforms, const T* pos, const T* target_pose_local, int joint, int k, int k_max) {
    T joint_pos[3] = {
        s_jointXforms[joint * 16 + 12],
        s_jointXforms[joint * 16 + 13],
        s_jointXforms[joint * 16 + 14]
    };

    T r[3] = {
        s_jointXforms[joint * 16 + 8],
        s_jointXforms[joint * 16 + 9],
        s_jointXforms[joint * 16 + 10]
    };
    normalize_vec3(r);

    T u[3] = {
        pos[0] - joint_pos[0],
        pos[1] - joint_pos[1],
        pos[2] - joint_pos[2]
    };

    T v[3] = {
        target_pose_local[0] - joint_pos[0],
        target_pose_local[1] - joint_pos[1],
        target_pose_local[2] - joint_pos[2]
    };

    T dot_u_r = u[0] * r[0] + u[1] * r[1] + u[2] * r[2];
    T dot_v_r = v[0] * r[0] + v[1] * r[1] + v[2] * r[2];
    T uproj[3] = { u[0] - dot_u_r * r[0],
                    u[1] - dot_u_r * r[1],
                    u[2] - dot_u_r * r[2] };
    T vproj[3] = { v[0] - dot_v_r * r[0],
                    v[1] - dot_v_r * r[1],
                    v[2] - dot_v_r * r[2] };
    normalize_vec3(uproj);
    normalize_vec3(vproj);

    T dotp = uproj[0] * vproj[0] + uproj[1] * vproj[1] + uproj[2] * vproj[2];
    dotp = glass::block::clamp_unit(dotp);
    T theta = acos(dotp);

    T cx = uproj[1] * vproj[2] - uproj[2] * vproj[1];
    T cy = uproj[2] * vproj[0] - uproj[0] * vproj[2];
    T cz = uproj[0] * vproj[1] - uproj[1] * vproj[0];

    T sign = r[0] * cx + r[1] * cy + r[2] * cz;
    if (sign < 0)
        theta = -theta;

    return coarse_step(theta, k, k_max);
}

template<typename T>
__device__ T solve_ori(const T* s_jointXforms, const T* q_t, int joint, int k, int k_max) {
    using QL = glass::block::QuatLayout;

    T r[3] = {
        s_jointXforms[joint * 16 + 8],
        s_jointXforms[joint * 16 + 9],
        s_jointXforms[joint * 16 + 10]
    };
    normalize_vec3(r);

    // q_err = q_t (x) conj(q_ee): the rotation taking the current EE orientation to the target.
    T q_ee[4], q_ee_inv[4], q_err[4];
    mat_to_quat(&s_jointXforms[EE_IDX * 16], q_ee);
    glass::thread::quat_conj<T, QL::wxyz>(q_ee, q_ee_inv);
    glass::thread::quat_mul<T, QL::wxyz>(q_t, q_ee_inv, q_err);
    glass::thread::quat_normalize<T, QL::wxyz>(q_err, q_err);

    // Unsigned angle from |w|; the axis sign is folded into `sign` below via the joint axis.
    T theta = 2.0f * acos(glass::block::clamp_unit(fabs(q_err[0])));
    T a[3] = { 1, 0, 0 };
    if (theta > 1e-3f) {
        const T sin_h = sin(theta / 2.0f);
        a[0] = q_err[1] / sin_h;
        a[1] = q_err[2] / sin_h;
        a[2] = q_err[3] / sin_h;
        normalize_vec3(a);
    }

    T sign = a[0] * r[0] + a[1] * r[1] + a[2] * r[2];
    if (sign < 0)
        theta = -theta;

    return coarse_step(theta, k, k_max);
}

// JACOBIAN TUNER
// Six-vector pose residual of EE transform Cn against target tp / unit q_goal:
// r = [tp - p_ee ; log(q_goal (x) q_ee^-1)], the world-frame rotation vector paired with the
// world-frame geometric Jacobian. (No cover fold needed: glass::quat_error returns the shortest path.)
template<typename T>
__device__ __forceinline__
void ee_residual6(const T* Cn, const T* tp, const T* q_goal, T* r6) {
    r6[0] = tp[0] - Cn[12];
    r6[1] = tp[1] - Cn[13];
    r6[2] = tp[2] - Cn[14];
    T qee[4]; mat_to_quat(Cn, qee);
    quat_err_rotvec(qee, q_goal, &r6[3]);
}

template<typename T>
__device__ __forceinline__ T half_sq_norm6(const T* r6) {
    return (T)0.5 * (r6[0]*r6[0] + r6[1]*r6[1] + r6[2]*r6[2] + r6[3]*r6[3] + r6[4]*r6[4] + r6[5]*r6[5]);
}

// Robust (Huber-like) per-row rescaling of an already row-normalised residual: rows whose
// magnitude exceeds the clip (cp for position, growing with the position error; co for
// orientation) are down-weighted by sqrt(clip/|r|). Returns the six sqrt-weights.
template<typename T>
__device__ __forceinline__
void robust_row_weights(const T* r_scaled, T pos_err_m, T* w6) {
    const T cp = fmax((T)1e-4, (T)5e-3 * (T)fmax((T)1, (T)1e3 * pos_err_m));
    const T co = (T)0.5;
#pragma unroll
    for (int k = 0; k < 6; ++k) {
        const T clip = (k < 3) ? cp : co;
        const T a = fabs(r_scaled[k]);
        const T w = (a <= clip) ? (T)1 : (clip / (a + (T)1e-30));
        w6[k] = sqrt(w);
    }
}

// Error-regime index 0..3 (coarse -> fine) shared by the LM trust-region / clip tables below.
template<typename T>
__device__ __forceinline__ int lm_regime(T pos_err_m, T ori_err_rad) {
    if (pos_err_m > (T)1e-2 || ori_err_rad > (T)0.6)  return 0;
    if (pos_err_m > (T)1e-3 || ori_err_rad > (T)0.25) return 1;
    if (pos_err_m > (T)2e-4 || ori_err_rad > (T)0.08) return 2;
    return 3;
}

template<typename T>
__device__ __forceinline__ T lm_step_clip(T pos_err_m) {
    return (pos_err_m > (T)1e-2) ? (T)0.30 :
           (pos_err_m > (T)1e-3) ? (T)0.15 :
           (pos_err_m > (T)2e-4) ? (T)0.08 : (T)0.03;
}

// Re-evaluate the row-scaled LM cost (and raw errors) of xcur in place: FK + residual.
template<typename T>
__device__ __forceinline__
void recompute_cost_scaled(T* xcur,
    T* s_jointX, T* s_XmatsHom,
    const T* row_s,
    const T* tp, const T* q_goal,
    T& cost_sq, T& pos_err_m, T& ori_err_rad)
{
    ee_fk_thread<T>(s_jointX, s_XmatsHom, xcur, EE_IDX, s_XmatsHom);
    T r6[6];
    ee_residual6(&s_jointX[EE_IDX * 16], tp, q_goal, r6);
    pos_err_m   = glass::thread::nrm2<T, 3>(&r6[0]);
    ori_err_rad = glass::thread::nrm2<T, 3>(&r6[3]);
#pragma unroll
    for (int k = 0; k < 6; ++k) r6[k] *= row_s[k];
    cost_sq = half_sq_norm6(r6);
}

template<typename T>
__device__ bool try_dogleg_step(
    T* s_x, const T* x_old,
    T* s_jointX, T* s_XmatsHom,
    const T* row_s,
    const T* tp, const T* q_goal,
    const T R,
    const T* dq_gn,
    const T* gvec, const T* diagA,
    T& cost_sq, T& pos_err_m, T& ori_err_rad,
    T& lambda, const T lambda_min, const T lambda_max,
    const double2* limits)
{
    T gtg = (T)0, gAg = (T)0;
#pragma unroll
    for (int i = 0; i < N; ++i) { T gi = gvec[i]; gtg += gi * gi; gAg += diagA[i] * gi * gi; }
    if (!(gtg > (T)0) || !(gAg > (T)0)) return false;

    T alpha = gtg / gAg;
    T gnorm = sqrt(gtg);
    if (alpha * gnorm > R) alpha = R / (gnorm + (T)1e-18);

    // Gradient
    T p_sd[N];
#pragma unroll
    for (int i = 0; i < N; ++i) p_sd[i] = -alpha * gvec[i];

    // Trial step (GN)
    T p_try[N], p_gn_norm = glass::thread::nrm2<T, N>(dq_gn);
    if (p_gn_norm <= R) {
#pragma unroll
        for (int i = 0; i < N; ++i) p_try[i] = dq_gn[i];
    }
    else {
        T a = 0, b = 0, c = 0;
#pragma unroll
        for (int i = 0; i < N; ++i) { T di = dq_gn[i] - p_sd[i]; a += di * di; b += (T)2 * (p_sd[i] * di); c += p_sd[i] * p_sd[i]; }
        T disc = b * b - (T)4 * a * (c - R * R);
        if (!(disc >= (T)0)) return false;
        T tau = (-b + sqrt(disc)) / ((T)2 * a);
        tau = fmin((T)1, fmax((T)0, tau));
#pragma unroll
        for (int i = 0; i < N; ++i) p_try[i] = p_sd[i] + tau * (dq_gn[i] - p_sd[i]);
    }

    T x_trial[N];
    clamp_into_limits(x_old, p_try, x_trial, limits);

    T c_new = cost_sq, p_new = pos_err_m, o_new = ori_err_rad;
    recompute_cost_scaled(x_trial, s_jointX, s_XmatsHom, row_s, tp, q_goal, c_new, p_new, o_new);

    const bool ok = (c_new + (T)1e-20 < cost_sq) && (p_new <= pos_err_m + (T)1e-12);
    if (ok) {
#pragma unroll
        for (int i = 0; i < N; ++i) s_x[i] = x_trial[i];
        cost_sq = c_new; pos_err_m = p_new; ori_err_rad = o_new;
        lambda = fmax(lambda * (T)0.5, lambda_min);
        return true;
    }
    else {
        lambda = fmin(lambda * (T)2.0, lambda_max);
        return false;
    }
}

template<typename T>
__device__ bool try_coord_linesearch(
    T* s_x, const T* x_old,
    T* s_jointX, T* s_XmatsHom,
    const T* row_s,
    const T* tp, const T* q_goal,
    const T* gvec,
    const T R, const T pos_err_m_hint,
    T& cost_sq, T& pos_err_m, T& ori_err_rad,
    T& lambda, const T lambda_min, const T lambda_max,
    const double2* limits)
{
    int i_star = 0; T gmax = fabs(gvec[0]);
#pragma unroll
    for (int i = 1; i < N; ++i) { T a = fabs(gvec[i]); if (a > gmax) { gmax = a; i_star = i; } }

    const T mag = lm_step_clip(pos_err_m_hint);
    T best_cost = cost_sq, best_pos = pos_err_m, best_ori = ori_err_rad;
    T x_trial[N]; bool accepted = false;

    for (int sgn = -1; sgn <= 1; sgn += 2) {
#pragma unroll
        for (int j = 0; j < N; ++j) x_trial[j] = x_old[j];
        const double2 L = limits[i_star];
        x_trial[i_star] = clamp_val<T>(x_old[i_star] + (T)sgn * mag, (T)L.x, (T)L.y);
        T c, p, o; recompute_cost_scaled(x_trial, s_jointX, s_XmatsHom, row_s, tp, q_goal, c, p, o);
        if ((p <= pos_err_m + (T)1e-12) && (c + (T)1e-20 < best_cost)) {
            best_cost = c; best_pos = p; best_ori = o; accepted = true;
#pragma unroll
            for (int j = 0; j < N; ++j) s_x[j] = x_trial[j];
        }
    }
    if (accepted) {
        cost_sq = best_cost; pos_err_m = best_pos; ori_err_rad = best_ori;
        lambda = fmax(lambda * (T)0.8, lambda_min);
        return true;
    }
    else {
        lambda = fmin(lambda * (T)2.0, lambda_max);
        return false;
    }
}


// Warp-cooperative NE build
template<typename T, int DIM>
__device__ inline void build_ne_and_solve_warp(
    const T* __restrict__ J,
    const T* __restrict__ r_scaled,
    T lambda,
    T* __restrict__ dq,
    T* __restrict__ diagA,
    T* __restrict__ gvec,
    T* __restrict__ A_sh,
    T* __restrict__ b_sh,
    int* __restrict__ s_fail_ptr
) {
    const unsigned mask = FULL_WARP_MASK;
    const int lane = threadIdx.x & 31;

    // Build A = J^T J and b = J^T r via GLASS (R4 adoption, composed rather than fused
    // gn_step so the un-shifted diag(A)/g taps below survive for the trust-region tuner;
    // same pieces, same order, zero duplicate work). Solve precision = T: fp64 by default,
    // fp32 when the refine knob is on, with no fp32<->fp64 casts in the hot path.
    // LAYOUT: our J buffer is stored J[k*DIM + r] (row k of 6, col r of DIM) — i.e. a
    // column-major DIM x 6 matrix B = J^T — so J^T J = B B^T is the TRANSPOSE=false syrk
    // and J^T r = B r the TRANSPOSE=false gemv, with ALL 32 lanes spreading the
    // accumulations (vs the former DIM-lane hand-rolled build). gemv's trailing
    // __syncwarp fences the syrk A-writes (same ordering contract as glass::warp::gn_step).
    glass::warp::syrk<T, DIM, 6, glass::block::FillMode::Full, /*TRANSPOSE=*/false>(
        (T)1, J, A_sh);
    glass::warp::gemv<T, DIM, 6, /*TRANSPOSE=*/false>(
        (T)1, J, r_scaled, (T)0, b_sh);

    // Save diag(A) and g = b for the downstream LM trust-region tuner (still consumed there).
    // The REG_DIAG posv applies the Levenberg lambda*diag(A) shift internally at factor time,
    // so we no longer mutate A's diagonal here; diagA holds the UN-shifted diagonal as before.
    if (lane < DIM) {
        diagA[lane] = A_sh[lane*DIM + lane];
        gvec[lane]  = b_sh[lane];
    }
    __syncwarp(mask);

    // One composed GLASS call folds: A += lambda*diag(A) (REG_DIAG Levenberg shift, == the old
    // `di + lambda*di` damping bit-for-bit) + Cholesky + forward/back solve + non-PD CHECK.
    // CHECK trips at the non-PD pivot (d<=0 || isnan), catching the finite-but-garbage
    // near-singular solves the former isfinite net let through. cholDecomp_InPlace writes
    // *s_fail_ptr from lane 0 (zeroed at entry); s_fail_ptr is a per-warp SHARED int
    // (LMWarpScratch::s_fail), so every lane reads it after the trailing __syncwarp.
    glass::warp::posv<T, DIM, /*NRHS=*/1, /*REGULARIZE=*/true, /*CHECK=*/true, /*REG_DIAG=*/true>(
        A_sh, b_sh, lambda, s_fail_ptr);  // A_sh <- L; b_sh <- x; *s_fail_ptr = 1 if non-PD pivot
    __syncwarp(mask);

    // Non-SPD fallback: dq=0 so LM grows lambda and retries (matches the old `ok ? x : 0`).
    const bool spd_ok = (*s_fail_ptr == 0);
    if (lane < DIM) dq[lane] = spd_ok ? (T)b_sh[lane] : (T)0;
    __syncwarp(mask);
}

// Per-warp scratch for the multi-warp LM refine: one independent LM candidate per warp,
// W warps per block. All of solve_lm_batched's former __shared__ arrays/scalars live here,
// placed W-wide in dynamic shared and indexed by warp_id, so each warp has a private copy
// and the iteration loop is fully warp-independent (every barrier __syncwarp, never
// __syncthreads). The constant cells of s_XmatsHom are q-independent (identical across
// candidates) and loaded once block-cooperatively in the prologue; ee_pose_inner_warp
// refreshes the q-dependent cells per-warp.
// ---------------------------------------------------------------------------------------------
// Collision-aware early stop + informed repair (2026-10). The cross-block stop flag used to be raised
// by the FIRST pose-accurate candidate, collision or not, so in cluttered scenes the whole batch
// could stop on a colliding candidate. In hard/both collision modes the stop now requires the
// candidate to be collision-free, decided WARP-LOCALLY from the joint transforms the solver already
// holds (no block barrier, no extra FK): each lane places spheres from the codegen sidecar tables
// and tests them against the environment, then the self-collision ranges are split across lanes.
// Open-world solves pass cc_stop = 0 and never enter this code.
#if defined(HJCD_HAS_COLLISION)
#include "hjcd_collision_tables.cuh"
using CCEnv = grid_collision::Environment<float>;
static_assert(hjcd_cc::NUM_SPHERES == grid_collision::NUM_COLLISION_SPHERES,
              "hjcd_collision_tables.cuh is stale: regenerate with scripts/codegen/generate_grid.py");
constexpr int CC_WARP_POS_FLOATS = 3 * hjcd_cc::NUM_SPHERES;   // per-warp scratch for sphere positions

// Warp-scoped verdict for the configuration whose joint world transforms are in s_jointX (any
// precision; column-major 4x4 per movable joint slot). w_pos = per-warp scratch of
// CC_WARP_POS_FLOATS floats. Entered by the full warp; every lane returns the same verdict.
template<typename T>
__device__ bool warp_config_free(const T* __restrict__ s_jointX, const CCEnv& env, float* w_pos)
{
    constexpr int NS = hjcd_cc::NUM_SPHERES;
    const int lane = threadIdx.x & 31;
    bool hit = false;
    for (int s = lane; s < NS; s += WARP_SIZE) {
        const T* X = &s_jointX[16 * hjcd_cc::sphere_anchor[s]];
        const float ox = hjcd_cc::sphere_offset[3 * s], oy = hjcd_cc::sphere_offset[3 * s + 1],
                    oz = hjcd_cc::sphere_offset[3 * s + 2];
        const float px = (float)(X[0] * ox + X[4] * oy + X[8]  * oz + X[12]);
        const float py = (float)(X[1] * ox + X[5] * oy + X[9]  * oz + X[13]);
        const float pz = (float)(X[2] * ox + X[6] * oy + X[10] * oz + X[14]);
        w_pos[3 * s] = px; w_pos[3 * s + 1] = py; w_pos[3 * s + 2] = pz;
        hit |= grid_collision::grid_cc_sphere_in_environment<float>(env, px, py, pz, hjcd_cc::sphere_radius[s]);
    }
    __syncwarp(FULL_WARP_MASK);
    for (int k = lane; k < grid_collision::NUM_COLLISION_SELF_CC_RANGES; k += WARP_SIZE) {
        const int i  = grid_collision::g_collision_self_cc_ranges[3 * k];
        const int j0 = grid_collision::g_collision_self_cc_ranges[3 * k + 1];
        const int j1 = grid_collision::g_collision_self_cc_ranges[3 * k + 2];
        const float ix = w_pos[3 * i], iy = w_pos[3 * i + 1], iz = w_pos[3 * i + 2], ir = hjcd_cc::sphere_radius[i];
        for (int j = j0; j <= j1 && !hit; ++j)
            hit |= grid_collision::grid_cc_sphere_sphere<float>(ix, iy, iz, ir, w_pos[3 * j], w_pos[3 * j + 1], w_pos[3 * j + 2],
                                                hjcd_cc::sphere_radius[j]) < 0.0f;
    }
    const bool any_hit = __any_sync(FULL_WARP_MASK, hit);
    __syncwarp(FULL_WARP_MASK);   // w_pos reads done before the caller reuses the scratch
    return !any_hit;
}
#else
struct CCEnv { int unused; };
constexpr int CC_WARP_POS_FLOATS = 0;
#endif

template<typename T>
struct LMWarpScratch {
    T s_x[N], x_old[N];
    T s_XmatsHom[grid::XHOM_T_COUNT], s_jointX[NX*16];
    T J[6*N], r_scaled[6], row_s[6], q_goal[4];
    T dq[N], diagA[N], gvec[N];
    T best_x_pos[N];
    T Ad_sh[N*N], rhsd_sh[N];
    T row_norm2[6];
    T pos_err_m, ori_err_rad, cost_sq, prev_cost, best_pos_seen;
    int s_break, stall, accepted;
    int s_fail;   // per-warp non-PD flag written by warp::posv CHECK (cholDecomp, lane 0)
    // collision-aware stop + repair round (hard/both modes only; unused open-world)
    int accurate, attempts, final_free, have_free;
    T q_acc[N];                                     // the accurate-but-colliding config of the repair round
    T best_free_x[N]; T best_free_pos;              // best collision-free config inside the fallback band
    float cc_pos[CC_WARP_POS_FLOATS > 0 ? CC_WARP_POS_FLOATS : 1];
};

template<typename T>
__device__ void solve_lm_batched(
    T* __restrict__ x,
    T* __restrict__ pose,
    const T* __restrict__ target_poses,
    T* __restrict__ pos_error,
    T* __restrict__ ori_error,
    const grid::robotModel<T>* d_robotModel,
    const T eps_pos,
    const T eps_ori,
    T lambda_init,
    const int k_max,
    const int B,
    int stop_on_first,
    CCEnv env,
    int cc_stop,              // 1 = collision-aware stop + repair (hard/both collision modes)
    int repair_attempts)
{
    // Multi-warp LM: one independent candidate per warp, W warps per block. All barriers
    // inside the iteration loop are warp-scoped (the only __syncthreads are in the one-time
    // constant-load prologue below, where the whole block genuinely participates).
    #define SYNC() __syncwarp(FULL_WARP_MASK)

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int warps_per_block = max(1, (int)(blockDim.x >> 5));
    const int gp  = blockIdx.x * warps_per_block + warp_id;
    const int tid = lane;   // all per-candidate work is now lane-indexed within the warp
    if (!x || !pose || !target_poses || !pos_error || !ori_error || !d_robotModel) return;

    // Per-warp scratch slice in dynamic shared (compiler computes per-warp offsets).
    extern __shared__ __align__(16) unsigned char s_lm_dyn_raw[];
    LMWarpScratch<T>* all_st = reinterpret_cast<LMWarpScratch<T>*>(s_lm_dyn_raw);
    LMWarpScratch<T>* st = &all_st[warp_id];

    // --- one-time block-cooperative constant-XmatsHom prologue ---
    // ee_pose_inner_warp refreshes the q-DEPENDENT cells of s_XmatsHom per-warp; only the
    // q-INDEPENDENT (constant) cells must be pre-present, and those are identical across all
    // candidates. Load once (any valid q -> only the constant cells are consumed) and
    // replicate into every warp slice. Block barriers are confined to this prologue.
    {
        __shared__ T s_xhom_tmpl[grid::XHOM_T_COUNT];
        __shared__ T s_q_tmpl[N];
        __shared__ T s_tmp_tmpl[NX*2];
        const int base0 = min(blockIdx.x * warps_per_block, B - 1);
        if (threadIdx.x < N) s_q_tmpl[threadIdx.x] = x[base0 * N + threadIdx.x];
        __syncthreads();
        grid::load_update_XmatsHom_helpers<T>(s_xhom_tmpl, /*s_topology_helpers=*/nullptr, s_q_tmpl, d_robotModel, s_tmp_tmpl);
        __syncthreads();
        for (int i = threadIdx.x; i < warps_per_block * grid::XHOM_T_COUNT; i += blockDim.x) {
            const int w = i / grid::XHOM_T_COUNT;
            const int c = i - w * grid::XHOM_T_COUNT;
            all_st[w].s_XmatsHom[c] = s_xhom_tmpl[c];
        }
        __syncthreads();
    }

    if (gp >= B) return;   // partial last block: excess warps exit cleanly (no block barrier below)

    // Per-warp aliases: the body below is unchanged (tid==lane, arrays/scalars are this warp's).
    T* s_x         = st->s_x;
    T* x_old       = st->x_old;
    T* s_XmatsHom  = st->s_XmatsHom;
    T* s_jointX    = st->s_jointX;
    T* J           = st->J;
    T* r_scaled    = st->r_scaled;
    T* row_s       = st->row_s;
    T* q_goal      = st->q_goal;
    T* dq          = st->dq;
    T* diagA       = st->diagA;
    T* gvec        = st->gvec;
    T* best_x_pos  = st->best_x_pos;
    T* Ad_sh       = st->Ad_sh;
    T* rhsd_sh     = st->rhsd_sh;
    T* row_norm2   = st->row_norm2;
    T& pos_err_m   = st->pos_err_m;
    T& ori_err_rad = st->ori_err_rad;
    T& cost_sq     = st->cost_sq;
    T& prev_cost   = st->prev_cost;
    T& best_pos_seen = st->best_pos_seen;
    int& s_break   = st->s_break;
    int& stall     = st->stall;
    int& accepted  = st->accepted;
    int& accurate  = st->accurate;
    int& attempts  = st->attempts;

    const T  lambda_min = (T)1e-12, lambda_max = (T)1e6;
    const int stall_lim = 5;

    const T* tp = &target_poses[gp*7];
    if (tid < N) s_x[tid] = x[gp*N + tid];
    if (tid == 0) {
        q_goal[0]=tp[3]; q_goal[1]=tp[4]; q_goal[2]=tp[5]; q_goal[3]=tp[6];
        T n = rsqrt(q_goal[0]*q_goal[0]+q_goal[1]*q_goal[1]+q_goal[2]*q_goal[2]+q_goal[3]*q_goal[3]);
        q_goal[0]*=n; q_goal[1]*=n; q_goal[2]*=n; q_goal[3]*=n;
        s_break=0; stall=0; prev_cost=(T)-1; cost_sq=(T)0; accurate=0; attempts=0;
        st->final_free = 0; st->have_free = 0; st->best_free_pos = (T)1e30;
    }
    SYNC();

    T lambda = lambda_init;

    // q-dependent cells refreshed per-warp by the FK (constants came from the prologue).
    ee_fk_warp<T>(s_jointX, s_XmatsHom, s_x, EE_IDX);
    SYNC();

    if (tid == 0) {
        pos_err_m   = compute_pos_err(s_jointX, tp);
        ori_err_rad = compute_ori_err(s_jointX, &tp[3]);
    }
    SYNC();

    if (tid == 0) best_pos_seen = pos_err_m;
    if (tid < N) best_x_pos[tid] = s_x[tid];
    // Already accurate on entry: done — unless the collision-aware stop finds it colliding, in which
    // case it enters the loop so the repair round below can act on it.
    if (tid == 0) accurate = (pos_err_m < eps_pos && ori_err_rad < eps_ori) ? 1 : 0;
    SYNC();
    if (accurate) {   // uniform (warp-shared)
#if defined(HJCD_HAS_COLLISION)
        if (cc_stop) {
            const bool free = warp_config_free<T>(s_jointX, env, st->cc_pos);
            if (tid == 0 && free) { s_break = 1; st->final_free = 1; }
        } else
#endif
        { if (tid == 0) s_break = 1; }
    }
    SYNC(); if (s_break) goto WRITE_OUT;
    // Finish the entry guard before lane 0 can update s_break from the global stop flag.
    SYNC();

    for (int it = 0; it < k_max; ++it) {
        if (stop_on_first && tid == 0 && ((it & 1) == 0)) {
            if (atomicAdd(&g_stop, 0)) s_break = 1;
        }
        SYNC(); if (s_break) break;

        // Position and orientation residual
        if (tid == 0) ee_residual6(&s_jointX[EE_IDX * 16], tp, q_goal, r_scaled);
        SYNC();

        // Build J and row-norms (row_norm2 is per-warp, aliased from the scratch struct)
        if (tid < 6) row_norm2[tid] = (T)0;
        SYNC();

        T p0 = (T)0, p1 = (T)0, p2 = (T)0, p3 = (T)0, p4 = (T)0, p5 = (T)0;

        if (tid < N) {
            const int i = tid;
            const T* Ci = &s_jointX[i*16];
            const T* Cn = &s_jointX[EE_IDX * 16];

            const T oi0=Ci[12], oi1=Ci[13], oi2=Ci[14];
            const T on0=Cn[12], on1=Cn[13], on2=Cn[14];
            const T zi0=Ci[8],  zi1=Ci[9],  zi2=Ci[10];
            const T r0=on0-oi0, r1=on1-oi1, r2=on2-oi2;

            const T j0 = zi1*r2 - zi2*r1;
            const T j1 = zi2*r0 - zi0*r2;
            const T j2 = zi0*r1 - zi1*r0;
            const T j3 = zi0;
            const T j4 = zi1;
            const T j5 = zi2;

            J[0*N+i]=j0;  J[1*N+i]=j1;  J[2*N+i]=j2;
            J[3*N+i]=j3;  J[4*N+i]=j4;  J[5*N+i]=j5;

            p0 = j0*j0; p1 = j1*j1; p2 = j2*j2; p3 = j3*j3; p4 = j4*j4; p5 = j5*j5;
        }

        // Warp row-norm reductions via glass::warp::reduce (returns the warp-wide sum
        // on every lane; equivalent to the previous __shfl_down_sync ladder, full warp).
        p0 = glass::warp::reduce(p0);
        p1 = glass::warp::reduce(p1);
        p2 = glass::warp::reduce(p2);
        p3 = glass::warp::reduce(p3);
        p4 = glass::warp::reduce(p4);
        p5 = glass::warp::reduce(p5);

        // Lane per warp accumulate into shared row sums
        if ((threadIdx.x & 31) == 0) {
            row_norm2[0] += p0; row_norm2[1] += p1; row_norm2[2] += p2;
            row_norm2[3] += p3; row_norm2[4] += p4; row_norm2[5] += p5;
        }
        SYNC();

        if (tid == 0) {
            #pragma unroll
            for (int k=0;k<6;++k)
                row_s[k] = (row_norm2[k] > (T)1e-18) ? rsqrt(row_norm2[k]) : (T)1;

            #pragma unroll
            for (int k=0;k<6;++k) r_scaled[k] *= row_s[k];

            // Robust row weights are folded into row_s so the same scaling applies to J.
            T w6[6];
            robust_row_weights(r_scaled, pos_err_m, w6);
            #pragma unroll
            for (int k=0;k<6;++k){ row_s[k]*=w6[k]; r_scaled[k]*=w6[k]; }

            T w_ori = (pos_err_m > (T)1e-3) ? (T)0.5 :
                      (pos_err_m > (T)2e-4) ? (T)1.0 : (T)1.5;
            const T s = sqrt(w_ori);
            row_s[3]*=s; row_s[4]*=s; row_s[5]*=s;
            r_scaled[3]*=s; r_scaled[4]*=s; r_scaled[5]*=s;

            cost_sq = half_sq_norm6(r_scaled);
        }
        SYNC();

        // Apply row_s to J (diagA / gvec are produced by the normal-equation build below)
        if (tid < N) {
            const int i = tid;
#pragma unroll
            for (int k=0;k<6;++k) J[k*N+i] *= row_s[k];
        }
        SYNC();

        // Build normal equations and solve (per-warp: each warp solves its own candidate)
        build_ne_and_solve_warp<T, N>(J, r_scaled, lambda, dq, diagA, gvec, Ad_sh, rhsd_sh, &st->s_fail);
        SYNC();

        if (tid == 0) {
            const T R_table[4] = { (T)0.38, (T)0.22, (T)0.12, (T)0.05 };
            const T R = R_table[lm_regime(pos_err_m, ori_err_rad)];

            const T nrm = glass::thread::nrm2<T, N>(dq);
            if (nrm > R) { T s = R/(nrm + (T)1e-18); for (int i=0;i<N;++i) dq[i]*=s; }

            const T clip = lm_step_clip(pos_err_m);
            for (int i=0;i<N;++i){
                dq[i]=clamp_val<T>(dq[i],-clip,clip);
                x_old[i]=s_x[i];
            }
        }
        SYNC();

        // accepted is per-warp (aliased from the scratch struct)
        if (tid == 0) accepted = 0;
        SYNC();

        T best_cost=(T)1e38, best_pos=pos_err_m, best_ori=ori_err_rad, best_a=(T)0;

        // Backtracking schedule: 1.0, 0.5, 0.25, 0.125
        for (int tries=0; tries<4; ++tries) {
            const T a = (tries==0)?(T)1.0 : (T)0.5 * (T)pow((T)0.5, tries-1);
            if (tid == 0) {
                for (int i=0;i<N;++i){
                    const double2 lim=c_joint_limits[i];
                    s_x[i] = clamp_val<T>(x_old[i] + a*dq[i], (T)lim.x, (T)lim.y);
                }
            }
            SYNC();

            ee_fk_warp<T>(s_jointX, s_XmatsHom, s_x, EE_IDX);
            SYNC();

            if (tid == 0) {
                const T* Cn=&s_jointX[EE_IDX * 16];
                const T pos_new = compute_pos_err_at(Cn, tp);
                const T ori_new = compute_ori_err_at(Cn, &tp[3]);

                // Trial cost: row-scaled residual with a fresh robust re-weighting at the trial's
                // own position error (on top of the weights already folded into row_s).
                T rr[6], w6[6];
                ee_residual6(Cn, tp, q_goal, rr);
#pragma unroll
                for (int k=0;k<6;++k) rr[k] *= row_s[k];
                robust_row_weights(rr, pos_new, w6);
#pragma unroll
                for (int k=0;k<6;++k) rr[k] *= w6[k];
                const T trial = half_sq_norm6(rr);

                const bool nonincreasing_pos = (pos_new <= pos_err_m + (T)1e-12);
                const bool improves_cost     = (trial + (T)1e-20 < best_cost);
                if (improves_cost && nonincreasing_pos) { 
                    best_cost=trial; best_pos=pos_new; best_ori=ori_new; best_a=a; accepted=1; 
                }
            }
            SYNC();
            if (accepted) break;
        }

        if (tid == 0) {
            if (accepted) {
                T ared = cost_sq - best_cost, pred=(T)0;
                for (int i=0;i<N;++i){ const T ad=best_a*dq[i]; pred += (T)0.5 * (lambda*diagA[i]*ad*ad + ad*gvec[i]); }
                pred = fmax((T)1e-20, pred);
                const T rho = ared / pred;

                if      (rho > (T)0.90) lambda=fmax(lambda*(T)0.3, lambda_min);
                else if (rho > (T)0.50) lambda=fmax(lambda*(T)0.5, lambda_min);
                else if (rho < (T)0.25) lambda=fmin(lambda*(T)3.0, lambda_max);

                cost_sq=best_cost; pos_err_m=best_pos; ori_err_rad=best_ori;

                if (pos_err_m + (T)1e-20 < best_pos_seen) { best_pos_seen=pos_err_m; for (int i=0;i<N;++i) best_x_pos[i]=s_x[i]; }
                if (prev_cost > (T)0 && (prev_cost - cost_sq)/prev_cost < (T)1e-9) ++stall; else stall=0;
                prev_cost=cost_sq;
            } else {
                // Dogleg + linesearch
                const T R_table[4] = { (T)0.45, (T)0.28, (T)0.10, (T)0.04 };
                const T R = R_table[lm_regime(pos_err_m, ori_err_rad)];

                bool took = try_dogleg_step<T>(s_x, x_old, s_jointX, s_XmatsHom,
                                               row_s, tp, q_goal, R,
                                               dq, gvec, diagA,
                                               cost_sq, pos_err_m, ori_err_rad,
                                               lambda, lambda_min, lambda_max,
                                               c_joint_limits);
                if (!took) {
                    took = try_coord_linesearch<T>(s_x, x_old, s_jointX, s_XmatsHom,
                                                   row_s, tp, q_goal, gvec,
                                                   R, pos_err_m,
                                                   cost_sq, pos_err_m, ori_err_rad,
                                                   lambda, lambda_min, lambda_max,
                                                   c_joint_limits);
                }
                if (!took) ++stall;
            }
            if (stall >= stall_lim) {
                for (int i=0;i<N;++i){
                    const double2 L=c_joint_limits[i];
                    const T span=(T)(L.y-L.x);
                    uint32_t u = 0x9E3779B9u ^ (uint32_t)(i*0xC2B2AE35u);
                    T sgn = (T)((int)(u&1)?1:-1);
                    T kick = (T)0.005 * span * sgn;
                    s_x[i] = clamp_val<T>(s_x[i] + kick, (T)L.x, (T)L.y);
                }
                ee_fk_thread<T>(s_jointX, s_XmatsHom, s_x, EE_IDX, s_XmatsHom);
                pos_err_m   = compute_pos_err(s_jointX,tp);
                ori_err_rad = compute_ori_err(s_jointX,&tp[3]);
                stall = 0;
            }

            accurate = (pos_err_m < eps_pos && ori_err_rad < eps_ori) ? 1 : 0;
            if (accurate && !cc_stop) { atomicCAS(&g_stop, 0, 1); s_break = 1; }
        }
        SYNC();
#if defined(HJCD_HAS_COLLISION)
        if (cc_stop && !accurate) {   // uniform: track the best collision-free config in the fallback band
            if (pos_err_m < HJCDSettings<T>::cc_fallback_pos && ori_err_rad < HJCDSettings<T>::cc_fallback_ori
                && pos_err_m < st->best_free_pos) {
                const bool free = warp_config_free<T>(s_jointX, env, st->cc_pos);
                if (tid == 0 && free) {
                    st->have_free = 1; st->best_free_pos = pos_err_m;
                    for (int i = 0; i < N; ++i) st->best_free_x[i] = s_x[i];
                }
            }
        }
        if (cc_stop && accurate) {   // uniform across the warp (shared)
            const bool free = warp_config_free<T>(s_jointX, env, st->cc_pos);
            if (tid == 0) {
                if (free) {
                    atomicCAS(&g_stop, 0, 1); s_break = 1; st->final_free = 1;
                } else if (attempts < repair_attempts) {
                    // Informed repair round: keep the accurate-but-colliding configuration, kick it with a
                    // deterministic joint-space perturbation that grows per attempt, and let the LM re-project
                    // it onto the pose; the verdict above re-runs when it is accurate again.
                    if (attempts == 0) for (int i = 0; i < N; ++i) st->q_acc[i] = s_x[i];
                    ++attempts;
                    const T amp = (T)0.04 * (T)attempts;
                    for (int i = 0; i < N; ++i) {
                        uint32_t seed = make_seed(0x5EEDu, gp, attempts, i);
                        const double2 L = c_joint_limits[i];
                        const T kick = ((T)2 * (T)u01(seed) - (T)1) * amp * (T)(L.y - L.x);
                        s_x[i] = clamp_val<T>(st->q_acc[i] + kick, (T)L.x, (T)L.y);
                    }
                    ee_fk_thread<T>(s_jointX, s_XmatsHom, s_x, EE_IDX, s_XmatsHom);
                    pos_err_m   = compute_pos_err(s_jointX, tp);
                    ori_err_rad = compute_ori_err(s_jointX, &tp[3]);
                    lambda = lambda_init; stall = 0; prev_cost = (T)-1;
                    accurate = 2;                              // -> every lane restarts its iteration budget
                } else {
                    // Out of attempts: hand back the accurate (colliding) configuration; the post-solve
                    // hard filter drops it, so a failed repair costs nothing in correctness.
                    if (attempts > 0) {
                        for (int i = 0; i < N; ++i) s_x[i] = st->q_acc[i];
                        ee_fk_thread<T>(s_jointX, s_XmatsHom, s_x, EE_IDX, s_XmatsHom);
                        pos_err_m   = compute_pos_err(s_jointX, tp);
                        ori_err_rad = compute_ori_err(s_jointX, &tp[3]);
                    }
                    s_break = 1;
                }
            }
            SYNC();
            if (accurate == 2) it = -1;   // uniform: `accurate` is warp-shared
        }
#endif
        if (s_break) break;

        ee_fk_warp<T>(s_jointX, s_XmatsHom, s_x, EE_IDX);
        SYNC();

        if (tid == 0) {
            pos_err_m   = compute_pos_err(s_jointX, tp);
            ori_err_rad = compute_ori_err(s_jointX, &tp[3]);
        }
        SYNC();
    }

WRITE_OUT:

    // Drift guard against noisy acceptances
    {
        const T MAX_DRIFT = (T)2e-4;
        if (pos_err_m > best_pos_seen + MAX_DRIFT) {
            if (tid < N) s_x[tid] = best_x_pos[tid];
            SYNC();
            ee_fk_warp<T>(s_jointX, s_XmatsHom, s_x, EE_IDX);
            SYNC();
        }
    }
    // Even when no restore was needed, every lane must finish reading the shared
    // restore guard before lane 0 overwrites pos_err_m for the final output.
    SYNC();
#if defined(HJCD_HAS_COLLISION)
    // Collision-aware fallback: no collision-free converged configuration -> hand back the best
    // collision-free one inside the tolerance band instead of an exact-but-colliding (filtered) one.
    if (cc_stop && !st->final_free && st->have_free) {   // uniform (warp-shared)
        if (tid < N) s_x[tid] = st->best_free_x[tid];
        SYNC();
        ee_fk_warp<T>(s_jointX, s_XmatsHom, s_x, EE_IDX);
        SYNC();
    }
#endif

    if (tid == 0) {
        pos_err_m   = compute_pos_err(s_jointX, tp);
        ori_err_rad = compute_ori_err(s_jointX, &tp[3]);

        store_ee_pose7(&s_jointX[EE_IDX * 16], &pose[gp*7]);
        pos_error[gp] = pos_err_m * (T)1000.0;
        ori_error[gp] = ori_err_rad;
        for (int i=0;i<N;++i) x[gp*N+i] = s_x[i];
    }

    #undef SYNC
}

// COARSE SEARCH
template<typename T>
__global__ void coarse_search(
    T* __restrict__ x,
    T* __restrict__ pose,
    const T* __restrict__ targetsB,
    T* __restrict__ pos_errors,
    T* __restrict__ ori_errors,
    const grid::robotModel<T>* d_robotModel,
    bool stop_on_first,
    CCEnv env,
    int cc_stop,              // 1 = only a collision-free accurate candidate raises the stop flag
    unsigned char* __restrict__ coarse_free   // cc_stop: per-block verdict of the returned candidate
) {
    const int gp   = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps_per_block = max(1, (int)(blockDim.x >> 5));

    if (!x || !pose || !targetsB || !pos_errors || !ori_errors || !d_robotModel) return;

    // Two per-warp scratch blocks: l_tmp (anchor-p locals) and l_C1 (anchor-p world chain).
    // The anchor FK (cand_p) is computed ONCE per anchor; every candidate j then recomputes
    // only the suffix from j into a PER-LANE register buffer (no shared per-candidate frame
    // array needed), so candidates parallelize across the warp's lanes.
    extern __shared__ __align__(16) unsigned char s_dyn_raw[];
    T* s_dyn = reinterpret_cast<T*>(s_dyn_raw);
    const size_t per_warp_elems = (size_t)(2 * NX * 16);
    T* warp_base = s_dyn + (size_t)warp * per_warp_elems;
    T* l_tmp = warp_base;
    T* l_C1  = warp_base + (size_t)(NX * 16);

    __shared__ int  s_stop;
    __shared__ int  s_allow_ori;
    __shared__ int  s_last_joint_o, s_last_joint_p;
    __shared__ int  s_accurate;

    __shared__ T s_x[N];
    __shared__ T s_pose[7];
    __shared__ T s_glob_pos_err, s_glob_ori_err;

    __shared__ T s_pos_theta1[N], s_ori_theta1[N];
    __shared__ T s_pos_err[N],    s_ori_err[N];

    __shared__ T s_XmatsHom[grid::XHOM_T_COUNT];
    __shared__ T s_jointXforms[NX*16];
    __shared__ T s_temp[NX*2];

    const T* target_pose_local = &targetsB[gp * 7];
    const T q_t[4] = { target_pose_local[3], target_pose_local[4],
                       target_pose_local[5], target_pose_local[6] };

    if (tid == 0) { s_last_joint_o = -1; s_last_joint_p = -1; }
    __syncthreads();

    // Random initial config in limits
    if (tid < N) {
        uint32_t st = make_seed(1337u, gp, 0, tid);
        float r = u01(st);
        const double2 L = c_joint_limits[tid];
        s_x[tid] = (T)(L.x + r * (L.y - L.x));
        s_pos_theta1[tid] = (T)0;
        s_ori_theta1[tid] = (T)0;
        s_pos_err[tid]    = (T)1e9;
        s_ori_err[tid]    = (T)1e9;
    }
    __syncthreads();

    grid::load_update_XmatsHom_helpers<T>(s_XmatsHom, /*s_topology_helpers=*/nullptr, s_x, d_robotModel, s_temp);
    __syncthreads();

    if ((threadIdx.x >> 5) == 0) { // warp 0
        ee_fk_warp<T>(s_jointXforms, s_XmatsHom, s_x, EE_IDX);
    }
    __syncthreads();

    if (tid == 0) {
        s_glob_pos_err = compute_pos_err(s_jointXforms, target_pose_local);
        s_glob_ori_err = compute_ori_err(s_jointXforms, q_t);
        store_ee_pose7(&s_jointXforms[EE_IDX * 16], s_pose);
    }
    __syncthreads();

    // Poll once here; later iterations reuse the flag published at the loop's end.
    // Rewriting it at the top could race another warp's previous stop decision.
    if (tid == 0) s_stop = stop_on_first ? read_stop() : 0;
    __syncthreads();
    for (int k = 0; k < HJCDSettings<T>::k_max; ++k) {
        if (stop_on_first && s_stop) break;

        if ((threadIdx.x >> 5) == 0) { // warp 0
            ee_fk_warp<T>(s_jointXforms, s_XmatsHom, s_x, EE_IDX);
        }
        __syncthreads();

        if (tid == 0) {
            store_ee_pose7(&s_jointXforms[EE_IDX * 16], s_pose);
            const T pos_gate = (T)10e-4;
            s_allow_ori = (s_glob_pos_err < pos_gate) ? 1 : 0;
        }
        __syncthreads();

        // Compute per-joint theta1 for pos & ori
        for (int idx = warp; idx < 2 * N; idx += warps_per_block) {
            const int phase = idx / N;
            const int p     = idx % N;
            if (lane == 0) {
                if (phase == 0) {
                    s_pos_theta1[p] = solve_pos<T>(s_jointXforms, s_pose, target_pose_local, p, k, HJCDSettings<T>::k_max);
                } else {
                    s_ori_theta1[p] = s_allow_ori ? solve_ori<T>(s_jointXforms, q_t, p, k, HJCDSettings<T>::k_max) : (T)0;
                }
            }
        }
        __syncthreads();

        // Evaluate greedy pairwise (p,j) with two scratch buffers (l_tmp, l_C)
        for (int idx = warp; idx < 2 * N; idx += warps_per_block) {
            const int phase = idx / N;
            const int p     = idx % N;
            const bool pos_phase = (phase == 0);

            T best_err_lane = pos_phase ? s_glob_pos_err : s_glob_ori_err;
            int best_j_lane = -1;

            // cand_p = s_x with the anchor perturbation on joint p — shared by every candidate j.
            // C1 (the FK of cand_p) is computed ONCE on lane 0 into per-warp l_C1 (world chain)
            // + l_tmp (locals), then published to the whole warp; the N candidates then run in
            // PARALLEL across the warp's lanes (the supported serial models have N <= 32),
            // each recomputing only the suffix from its joint j into a per-lane EE buffer.
            const T delta1 = pos_phase ? s_pos_theta1[p] : s_ori_theta1[p];
            if (lane == 0) {
                T cand_p[N];
                #pragma unroll
                for (int m = 0; m < N; ++m) cand_p[m] = s_x[m];
                cand_p[p] = clamp_val<T>(cand_p[p] + delta1,
                                         (T)c_joint_limits[p].x, (T)c_joint_limits[p].y);
                #pragma unroll
                for (int m = 0; m < NX * 16; ++m) l_tmp[m] = s_XmatsHom[m];
                ee_fk_thread<T>(l_C1, l_tmp, cand_p, EE_IDX, s_XmatsHom);
            }
            __syncwarp(FULL_WARP_MASK);   // publish l_C1 / l_tmp (lane 0 -> all lanes)

            const int ee = EE_IDX * 16;
            const T pos1[3] = { l_C1[ee + 12], l_C1[ee + 13], l_C1[ee + 14] };
            const T cand_pp = clamp_val<T>(s_x[p] + delta1,
                                           (T)c_joint_limits[p].x, (T)c_joint_limits[p].y);

            for (int j = lane; j < N; j += WARP_SIZE) {
                // theta2 from C1 (all lanes read the shared anchor FK read-only)
                T theta2 = (T)0;
                if (pos_phase) {
                    theta2 = solve_pos<T>(l_C1, pos1, target_pose_local, j, k, HJCDSettings<T>::k_max);
                } else {
                    theta2 = s_allow_ori ? solve_ori<T>(l_C1, q_t, j, k, HJCDSettings<T>::k_max) : (T)0;
                }
                const T candp_j = (j == p) ? cand_pp : s_x[j];   // cand_p[j]
                const T aj = clamp_val<T>(candp_j + theta2,
                                          (T)c_joint_limits[j].x, (T)c_joint_limits[j].y);

                // C2 = cand_p with only joint j perturbed -> reuse C1's chain/locals, recompute
                // only the suffix from j into a per-lane EE transform. Bit-identical to a full FK.
                T ee16[16];
                ee_fk_suffix_thread<T>(ee16, l_C1, l_tmp, s_XmatsHom, j, aj);
                const T err = pos_phase ? compute_pos_err_at(ee16, target_pose_local)
                                        : compute_ori_err_at(ee16, q_t);

                if (err < best_err_lane) { best_err_lane = err; best_j_lane = j; }
            }

            // warp min-reduce over candidates via GLASS's keyed register-pair argmin
            // (lower-index tie-break == the serial "first strict-improvement wins" selection;
            // no-candidate lanes pass the UINT32_MAX empty sentinel == the old -1). Winner
            // (index + key) is broadcast to all lanes.
            {
                const uint32_t idx_in = (best_j_lane < 0) ? UINT32_MAX : (uint32_t)best_j_lane;
                T win_err;
                const uint32_t win_j = glass::warp::argmin_pair(best_err_lane, idx_in, win_err);
                best_j_lane   = (win_j == UINT32_MAX) ? -1 : (int)win_j;
                best_err_lane = (win_j == UINT32_MAX)
                                    ? (pos_phase ? s_glob_pos_err : s_glob_ori_err) : win_err;
            }

            if (lane == 0) {
                if (pos_phase) s_pos_err[p] = best_err_lane;
                else           s_ori_err[p] = best_err_lane;
            }
            // The shuffle reduction exchanges registers, not a shared-memory fence.
            // Finish every lane's anchor reads before lane 0 reuses l_C1/l_tmp.
            __syncwarp(FULL_WARP_MASK);
        }
        __syncthreads();

        // Choose best position and orientation joints
        if (tid == 0) {
            int best_pos_joint = -1, best_ori_joint = -1;
            T best_pos_imp = (T)0,  best_ori_imp = (T)0;

            for (int jj = 0; jj < N; ++jj) {
                if (jj == s_last_joint_o) continue;
                const T imp_p = s_glob_pos_err - s_pos_err[jj];
                if (imp_p > best_pos_imp && imp_p > (T)1e-5) {
                    best_pos_imp = imp_p; best_pos_joint = jj;
                }
            }
            for (int jj = 0; jj < N; ++jj) {
                if (jj == s_last_joint_p) continue;
                const T imp_o = s_glob_ori_err - s_ori_err[jj];
                if (imp_o > best_ori_imp && imp_o > (T)1e-5) {
                    best_ori_imp = imp_o; best_ori_joint = jj;
                }
            }

            s_last_joint_o = best_ori_joint;
            s_last_joint_p = best_pos_joint;

            if (best_ori_joint != -1 && best_ori_joint != best_pos_joint) {
                const T d = s_ori_theta1[best_ori_joint];
                s_x[best_ori_joint] = clamp_val<T>(
                    s_x[best_ori_joint] + d,
                    (T)c_joint_limits[best_ori_joint].x,
                    (T)c_joint_limits[best_ori_joint].y);
            }
            if (best_pos_joint != -1) {
                const T d = s_pos_theta1[best_pos_joint];
                s_x[best_pos_joint] = clamp_val<T>(
                    s_x[best_pos_joint] + d,
                    (T)c_joint_limits[best_pos_joint].x,
                    (T)c_joint_limits[best_pos_joint].y);
            }

            if (best_ori_joint == -1 && best_pos_joint == -1) {
                perturb_joint_config<T>(s_x, gp);
            }
        }
        __syncthreads();

        // Update global err and pose, early-exit
        if ((threadIdx.x >> 5) == 0) { // warp 0
            ee_fk_warp<T>(s_jointXforms, s_XmatsHom, s_x, EE_IDX);
        }
        __syncthreads();

        if (tid == 0) {
            s_glob_pos_err = compute_pos_err(s_jointXforms, target_pose_local);
            s_glob_ori_err = compute_ori_err(s_jointXforms, q_t);
            store_ee_pose7(&s_jointXforms[EE_IDX * 16], s_pose);

            for (int jj = 0; jj < N; ++jj) {
                s_pos_err[jj] = s_glob_pos_err;
                s_ori_err[jj] = s_glob_ori_err;
            }

            s_accurate = (s_glob_pos_err < HJCDSettings<T>::epsilon && s_glob_ori_err < HJCDSettings<T>::nu) ? 1 : 0;
            if (stop_on_first && s_accurate && !cc_stop)
                atomicCAS(&g_stop, 0, 1);
        }
        __syncthreads();
#if defined(HJCD_HAS_COLLISION)
        if (cc_stop) {   // uniform: collision-aware stop. warp 0's dead per-warp scratch holds the spheres.
            if (stop_on_first && s_accurate) {
                if (warp == 0) {
                    const bool free = warp_config_free<T>(s_jointXforms, env, reinterpret_cast<float*>(warp_base));
                    if (lane == 0 && free) atomicCAS(&g_stop, 0, 1);
                }
                __syncthreads();
            }
        }
#endif
        if (tid == 0) s_stop = read_stop();
        __syncthreads();
        if (s_stop) break;
        // An accurate but colliding candidate is this block's result either way: hand it to the LM
        // (whose repair round can fix it) rather than wandering off with further descent steps.
        if (cc_stop && s_accurate) break;

        if (tid < N) x[gp * N + tid] = s_x[tid];
    }

    if (tid < N) x[gp * N + tid] = s_x[tid];
    if (tid < 7) pose[gp * 7 + tid] = s_pose[tid];
    if (tid == 0) {
        pos_errors[gp] = s_glob_pos_err * (T)1000.0;
        ori_errors[gp] = s_glob_ori_err;
    }
#if defined(HJCD_HAS_COLLISION)
    // Collision-aware ranking: the top-K selection that seeds the LM penalises colliding coarse
    // candidates (a 20 mm-accurate colliding seed is often repairable, so a penalty, not exclusion).
    // s_jointXforms holds the FK of the returned s_x (refreshed at the end of every iteration).
    if (cc_stop) {   // uniform
        if (warp == 0) {
            const bool free = warp_config_free<T>(s_jointXforms, env, reinterpret_cast<float*>(warp_base));
            if (lane == 0) coarse_free[gp] = free ? 1 : 0;
        }
    }
#endif
}


template<typename T>
__global__ void lm_tuner(
    T* __restrict__ x,
    T* __restrict__ pose,
    const T* __restrict__ targetsB,
    T* __restrict__ pos_errors,
    T* __restrict__ ori_errors,
    const grid::robotModel<T>* d_robotModel,
    T eps_pos_m,
    T eps_ori_rad,
    T lambda_init,
    int k_max,
    int B,
    int stop_on_first,
    CCEnv env,
    int cc_stop,
    int repair_attempts
) {
    // B = total candidates (Krep). grid = ceil(B / warps_per_block); each warp does one candidate.
    solve_lm_batched<T>(
        x,
        pose,
        targetsB,
        pos_errors,
        ori_errors,
        d_robotModel,
        eps_pos_m,
        eps_ori_rad,
        lambda_init,
        k_max,
        B,
        stop_on_first,
        env,
        cc_stop,
        repair_attempts
    );
}


template <typename T>
__global__ void gather_rows_kernel(const T* __restrict__ xsrc,
    const int* __restrict__ idx,
    T* __restrict__ xdst,
    int rows) {
    int r = blockIdx.x;
    if (r >= rows) return;
    int src_row = idx[r];

    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        xdst[r * N + j] = xsrc[src_row * N + j];
    }
}

// EE pose [x,y,z,qw,qx,qy,qz] of each of B joint configurations (one block per config).
template <typename T>
__global__ void forward_kinematics_kernel(
    const T* __restrict__ q,
    T* __restrict__ ee_pose7,
    const grid::robotModel<T>* __restrict__ RM,
    const int B)
{
    const int b = blockIdx.x;
    if (!q || !ee_pose7 || !RM || b >= B) return;

    __shared__ T s_q[N];
    __shared__ T s_XmatsHom[grid::XHOM_T_COUNT];
    __shared__ T s_jointX[NX * 16];
    __shared__ T s_tmp[NX * 2];

    for (int j = threadIdx.x; j < N; j += blockDim.x)
        s_q[j] = q[(size_t)b * N + j];
    __syncthreads();

    grid::load_update_XmatsHom_helpers<T>(s_XmatsHom, /*s_topology_helpers=*/nullptr, s_q, RM, s_tmp);
    __syncthreads();

    if (threadIdx.x == 0) {
        ee_fk_thread<T>(s_jointX, s_XmatsHom, s_q, EE_IDX, s_XmatsHom);
        store_ee_pose7(&s_jointX[EE_IDX * 16], &ee_pose7[(size_t)b * 7]);
    }
}

// SAMPLE CONFIG
__device__ __constant__ int c_halton_bases[32] =
    {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,
     59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131};

template <typename T>
__device__ inline T radical_inverse(uint32_t n, int b) {
    T inv = (T)1.0 / (T)b;
    T f   = inv;
    T x   = (T)0.0;
    while (n) {
        uint32_t d = n % (uint32_t)b;
        x += (T)d * f;
        n /= (uint32_t)b;
        f *= inv;
    }
    return x; 
}

template <typename T>
__global__ void sample_q_halton_kernel(T* __restrict__ d_q,
                                       int num_configs,
                                       uint64_t seed,
                                       int offset = 1,
                                       int leap   = 1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_configs) return;

    uint32_t n = (uint32_t)(offset + i * leap);

    uint32_t hseed = (uint32_t)(seed ^ 0x9E3779B97f4a7c15ull);

    #pragma unroll
    for (int j = 0; j < N; ++j) {
        const int base = c_halton_bases[j];
        T u = radical_inverse<T>(n, base); 

        uint32_t sh = wanghash(hseed + (uint32_t)j * 0x9E3779B9u);
        T shift = (T)((sh & 0xFFFFFFu) / (double)0x1000000u); 
        u = u + shift;
        u = u - floor(u);

        double2 lim = c_joint_limits[j];
        T lo = (T)lim.x;
        T hi = (T)lim.y;
        d_q[(size_t)i * N + j] = lo + u * (hi - lo);
    }
}

template<typename T>
T* sample_ik_config_halton(hjcd::DeviceAllocations& allocations,
                           const grid::robotModel<T>* d_robotModel,
                           int num_configs,
                           uint64_t seed,
                           int offset = 1,
                           int leap   = 1) {
    if (num_configs <= 0 || !d_robotModel) return nullptr;

    T* d_q = nullptr;
    allocations.allocate(d_q, sizeof(T) * (size_t)num_configs * N);

    const int tpb = 256;
    const int gpb = (num_configs + tpb - 1) / tpb;

    sample_q_halton_kernel<T><<<gpb, tpb>>>(d_q, num_configs, seed, offset, leap);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    return d_q;
}

template<typename T>
std::vector<std::array<T,7>>
sample_random_target_poses(const grid::robotModel<T>* d_robotModel,
                           int num_configs, uint64_t seed) {
    std::lock_guard<std::recursive_mutex> lock(solver_mutex());
    if (num_configs <= 0 || num_configs > std::numeric_limits<int>::max() - 255)
        throw std::invalid_argument("num_configs must be positive and fit CUDA launch indexing");
    init_joint_limits_from_grid();
    if (!d_robotModel) d_robotModel = cached_robot_model<T>();
    hjcd::DeviceAllocations allocations;
    std::vector<std::array<T,7>> out;

    T* d_q = sample_ik_config_halton<T>(allocations, d_robotModel, num_configs, seed, /*offset=*/1, /*leap=*/1);
    if (!d_q) return out;

    T* d_pose7 = nullptr;
    allocations.allocate(d_pose7, sizeof(T) * 7 * (size_t)num_configs);

    const int threads = 32;
    const int blocks  = num_configs;

    forward_kinematics_kernel<T><<<blocks, threads>>>(d_q, d_pose7, d_robotModel, num_configs);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<T> h_pose7((size_t)num_configs * 7);
    CUDA_OK(cudaMemcpy(h_pose7.data(), d_pose7,
               sizeof(T) * 7 * (size_t)num_configs, cudaMemcpyDeviceToHost));

    out.resize(num_configs);
    for (int i = 0; i < num_configs; ++i)
        for (int k = 0; k < 7; ++k)
            out[i][k] = h_pose7[(size_t)i * 7 + k];

    allocations.clear();
    return out;
}

// Candidate ranking score (lower is better): position error in mm plus a heavy penalty on
// orientation error beyond a small target. Shared by the device coarse top-K ranking and the
// host final selection so both stages rank candidates identically.
template<typename T>
__host__ __device__ __forceinline__ T rank_score(T pos_err_mm, T ori_err_rad) {
    constexpr T ORI_TARGET_RAD = (T)1.1e-4;
    constexpr T ORI_OUTLIER_W  = (T)1e4;
    const T ori_excess = ori_err_rad - ORI_TARGET_RAD;
    return pos_err_mm + ORI_OUTLIER_W * (ori_excess > (T)0 ? ori_excess : (T)0);
}

template<typename T>
__global__ void build_scores_kernel(const T* __restrict__ pos_err_mm,
    const T* __restrict__ ori_err_rad,
    const unsigned char* __restrict__ coarse_free,   // nullptr = open world; else 1 = collision-free
    T penalty,                                       // added to colliding candidates' rank score
    T* __restrict__ scores,
    int B)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    T sc = rank_score(pos_err_mm[i], ori_err_rad[i]);
    if (coarse_free && !coarse_free[i]) sc += penalty;
    scores[i] = sc;
}

template<typename T>
__global__ void replicate_rows_kernel(const T* __restrict__ src,
    T* __restrict__ dst,
    int K, int C, int rep)
{
    int r = blockIdx.x;
    if (r >= K) return;
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        T v = src[r * C + j];
        for (int t = 0; t < rep; ++t) {
            dst[(r * rep + t) * C + j] = v;
        }
    }
}

// out[r*7 + k] = target7[k] for r < R, one thread per output element (flat over R*7).
template<typename T>
__global__ void replicate_target7_kernel(const T* __restrict__ target7,
    T* __restrict__ out,
    int R)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)R * 7) out[i] = target7[i % 7];
}

template<typename T>
__global__ void perturb_rows_kernel(T* __restrict__ X,
    int R,
    T sigma_frac,
    uint64_t seed,
    int groupSize,
    bool skip_first_in_group)
{
    int r = blockIdx.x;
    if (r >= R) return;
    const bool skip = skip_first_in_group && (groupSize > 0) && ((r % groupSize) == 0);
    if (skip) return;

    uint32_t s = (uint32_t)(seed ^ (uint64_t)r * 0x9E3779B97F4A7C15ull);
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        uint32_t sj = wanghash(s ^ (uint32_t)j * 0xC2B2AE35u);
        const double2 L = c_joint_limits[j];
        const float step = (float)sigma_frac * (float)(L.y - L.x) * gauss01(sj);
        X[r * N + j] = clamp_val<T>(X[r * N + j] + (T)step, (T)L.x, (T)L.y);
    }
}

template <typename Dst, typename Src>
__global__ void cast_array(const Src* __restrict__ in,
                           Dst* __restrict__ out,
                           size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (Dst)in[i];
}

#if defined(HJCD_HAS_COLLISION)
// Threads per block for the collision kernels.
static constexpr int CC_TPB = 128;
#endif

// Soft environment-collision penetration cost (mm) for a batch of candidate configs, scored via
// grid_collision AFTER optimization (never on the hot solver path). One block per config,
// thread-count-invariant. Places the URDF-driven sphere model at q via the batched extractor and
// reduces the per-sphere signed clearance:
//   cost_mm[i] = 1000 * sum_sphere max(0, -d_sphere)          (d = nearest signed clearance, m)
// Base-link spheres are already dropped at codegen (anchor < 0).
// Requires dynamic smem = grid::MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<float>() for the FK
// extractor (TIER_SHARED => nullptr workspace).
//
// Both collision kernels reference the grid_collision namespace, which grid.cuh only carries when it
// was generated with --collision (sentinel HJCD_HAS_COLLISION). A no-collision header (e.g. the
// DoF-scaling regens) compiles fine: the kernels are omitted and the runtime collision path is off.
#if defined(HJCD_HAS_COLLISION)
__global__ void score_environment_costs(
    const double* __restrict__ q_in,                 // K x N
    int K,
    float* __restrict__ cost_mm,                     // K floats
    const grid::robotModel<float>* d_robotModel,
    grid_collision::Environment<float> env)          // device pointers, by value
{
    namespace gc = grid_collision;
    constexpr int NS = gc::NUM_COLLISION_SPHERES;

    const int i   = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    if (i >= K) return;

    __shared__ float s_q[N];
    __shared__ float s_pos[3 * NS];
    __shared__ float s_r[NS];
    __shared__ float s_dist[NS];
    __shared__ float s_normal[3 * NS];
    __shared__ float red[CC_TPB];

    for (int j = tid; j < N; j += blockDim.x) s_q[j] = (float)q_in[(size_t)i * N + j];
    __syncthreads();

    gc::collision_distance<float>(s_dist, s_normal, s_q, d_robotModel, env, s_pos, s_r, nullptr);

    float local = 0.0f;
    for (int s = tid; s < NS; s += blockDim.x) {
        const float pen = -s_dist[s];              // >0 => penetrating (empty env => -1e30 => skip)
        if (pen > 0.0f) local += 1000.0f * pen;
    }
    red[tid] = local;
    __syncthreads();
    glass::block::reduce<float>((uint32_t)blockDim.x, red);   // red[0] = block sum
    if (tid == 0) cost_mm[i] = red[0];
}

// Hard collision constraint: per candidate config, one block, write valid[i] =
// grid_collision::config_free (SELF + environment; note the soft path is env-only). The boolean is
// thread-invariant, so lane 0 publishes it. Same dynamic-smem contract as score_environment_costs.
// Selected by collision_mode = hard | both (see generate_ik_solutions).
__global__ void mark_collisions(
    const double* __restrict__ q_in,                 // K x N
    int K,
    unsigned char* __restrict__ valid,               // K bytes (1 = collision-free)
    const grid::robotModel<float>* d_robotModel,
    grid_collision::Environment<float> env)          // device pointers, by value
{
    namespace gc = grid_collision;
    constexpr int NS = gc::NUM_COLLISION_SPHERES;

    const int i   = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    if (i >= K) return;

    __shared__ float s_q[N];
    __shared__ float s_pos[3 * NS];
    __shared__ float s_r[NS];

    for (int j = tid; j < N; j += blockDim.x) s_q[j] = (float)q_in[(size_t)i * N + j];
    __syncthreads();

    const bool ok = gc::config_free<float>(s_q, d_robotModel, env, s_pos, s_r, nullptr);
    if (tid == 0) valid[i] = ok ? 1 : 0;
}
#endif  // HJCD_HAS_COLLISION

const double ENV_COLLISION_COST_W = 1.5;   // soft-mode weight of the penetration cost in the rank
const double COARSE_COLLIDING_RANK_PENALTY = 50.0;   // rank-score units (~mm) added to colliding coarse seeds
const float  CC_SPHERE_MARGIN_MM = 0.0f;   // env-collision margin (mm) for the coll-free tally

namespace {

// Clear the cross-block early-stop flag before each solver stage.
void reset_stop_flag() {
    const int zero = 0;
    CUDA_OK(cudaMemcpyToSymbol(g_stop, &zero, sizeof(int)));
}

// Upload the 7-vector target in precision T and replicate it R times (one row per candidate).
template<typename T>
T* upload_replicated_target(hjcd::DeviceAllocations& allocations, const double* target_pose, int R) {
    T h_target7[7];
    for (int i = 0; i < 7; ++i) h_target7[i] = static_cast<T>(target_pose[i]);
    T* d_target7 = nullptr;
    T* d_targets = nullptr;
    allocations.allocate(d_target7, sizeof(T) * 7);
    allocations.allocate(d_targets, sizeof(T) * 7 * (size_t)R);
    CUDA_OK(cudaMemcpy(d_target7, h_target7, sizeof(T) * 7, cudaMemcpyHostToDevice));
    const int tpb = 256;
    replicate_target7_kernel<T><<<(int)(((size_t)R * 7 + tpb - 1) / tpb), tpb>>>(d_target7, d_targets, R);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    return d_targets;
}

// Convert a device array to another scalar type (no-op alias when the types match).
template<typename Dst, typename Src>
Dst* device_cast(hjcd::DeviceAllocations& allocations, Src* src, size_t n) {
    if constexpr (std::is_same_v<Dst, Src>) {
        return src;
    } else {
        Dst* dst = nullptr;
        allocations.allocate(dst, sizeof(Dst) * n);
        const int tpb = 256;
        cast_array<Dst, Src><<<(int)((n + tpb - 1) / tpb), tpb>>>(src, dst, n);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
        return dst;
    }
}

}  // namespace

// RT = LM-refine compute precision (the user-facing speed/accuracy knob). RT=double is the
// full-fp64 default; RT=float runs FK/Jacobian/residual/line-search in fp32 (~2.4x cheaper FK,
// cf. coarse_search), including the normal-equations Cholesky in build_ne_and_solve_warp.
// (Default RT=double is declared in the header; not repeated here.)
template<typename T, typename RT>
Result<T> generate_ik_solutions(
    T* target_pose,
    int b_size,
    int num_solutions,
    bool collision_free,
    const char* problems_json_text,
    const char* problem_set_name,
    int problem_idx,
    bool write_stats,
    int collision_mode
)
{
    std::lock_guard<std::recursive_mutex> lock(solver_mutex());
    if (!target_pose) throw std::invalid_argument("target_pose must not be null");
    if (b_size <= 0 || b_size > (std::numeric_limits<int>::max() - 255) / (16 * N))
        throw std::invalid_argument("batch_size must be positive and fit CUDA indexing");
    if (num_solutions <= 0)
        throw std::invalid_argument("num_solutions must be positive");
    if (collision_mode < 0 || collision_mode > 2)
        throw std::invalid_argument("invalid collision mode (0 = soft, 1 = hard, 2 = both)");
    if (problem_idx < 0) throw std::invalid_argument("problem_idx must be non-negative");
    const auto normalized_target = normalized_target_pose(target_pose);
    double target_pose64[7];
    for (int i = 0; i < 7; ++i) target_pose64[i] = static_cast<double>(normalized_target[i]);
    init_joint_limits_from_grid();
    int device = 0;
    CUDA_OK(cudaGetDevice(&device));
    hjcd::DeviceAllocations allocations;

    using std::chrono::high_resolution_clock;
    auto t0 = high_resolution_clock::now();
    CUDA_OK(cudaDeviceSynchronize());

    Result<T> result{};

    // Collision environment (grid_collision). Cache exact problem contents and the fp32 robot model
    // per CUDA device. Collision requests fail explicitly when their scene is unusable; they never
    // degrade silently to an open-world solve. The path exists only in --collision-generated headers.
    const bool stop_on_first = true;   // coarse blocks stop once any block hits the coarse tolerance

#if defined(HJCD_HAS_COLLISION)
    struct CachedCollisionEnvironment {
        hjcd_env::DeviceEnv device_env;
        hjcd_env::ProblemDocument document;
        std::string problem_set;
        int problem_idx = -1;
        bool ready = false;
    };
    static std::unordered_map<int, CachedCollisionEnvironment> g_cc_env_by_device;
    grid_collision::Environment<float> cc_env{nullptr, 0, nullptr, 0, nullptr, 0};
    const grid::robotModel<float>* d_robotModel_cc = nullptr;

    if (collision_free) {
        if (!problems_json_text || !problem_set_name)
            throw std::invalid_argument(
                "collision-free solving requires problem JSON and a problem-set name");

        int cc_device = 0;
        CUDA_OK(cudaGetDevice(&cc_device));
        auto& cached = g_cc_env_by_device[cc_device];
        const bool changed = cached.document.update(problems_json_text);
        if (changed || !cached.ready || cached.problem_set != problem_set_name ||
            cached.problem_idx != problem_idx) {
            // Invalidate BEFORE selection/validation: a new document may fail here.
            // Retained allocations remain owned and are released on replacement.
            cached.ready = false;
            const auto& data = cached.document.select(problem_set_name, problem_idx);
            if (data.contains("valid") && !bool(data["valid"]))
                throw std::runtime_error("collision problem is marked invalid");

            const hjcd_env::HostEnv host_env = hjcd_env::problem_dict_to_env(data);
            std::string selected_set(problem_set_name);
            hjcd_env::free_env(cached.device_env);
            cached.device_env = hjcd_env::upload_env(host_env);
            cached.problem_set = std::move(selected_set);
            cached.problem_idx = problem_idx;
            cached.ready = true;
        }

        cc_env = cached.device_env.env;
        d_robotModel_cc = cached_robot_model<float>();
    }
#else
    if (collision_free)
        throw std::runtime_error(
            "collision-free solving requires grid.cuh generated with --collision");
#endif  // HJCD_HAS_COLLISION

    const bool do_cc = collision_free;

    // Collision policy: "hard" (1, default) filters colliding candidates outright; "soft" (0) is a
    // penetration cost that only biases selection; "both" (2) = soft cost + hard filter. Strict
    // filtering is what makes collision_free=True truthful.
    const bool use_soft = do_cc && (collision_mode == 0 || collision_mode == 2);
    const bool use_hard = do_cc && (collision_mode == 1 || collision_mode == 2);

    // Collision-aware early stop + LM repair round (hard/both modes): the stop flag is raised only by
    // a collision-free accurate candidate, and an accurate-but-colliding LM candidate gets
    // HJCD_REPAIR_ATTEMPTS kicked re-projections before it is handed to the post-solve filter.
    // HJCD_CC_STOP (diagnostic/A-B knob): bit 0 = coarse stage, bit 1 = LM stage; default 3 (both).
    CCEnv cc_kernel_env{};
    int cc_stop_mask = 3;
    if (const char* e = std::getenv("HJCD_CC_STOP")) { int v = std::atoi(e); if (v >= 0 && v <= 3) cc_stop_mask = v; }
    const int cc_stop = use_hard ? 1 : 0;
    const int cc_stop_coarse = (cc_stop && (cc_stop_mask & 1)) ? 1 : 0;
    const int cc_stop_lm     = (cc_stop && (cc_stop_mask & 2)) ? 1 : 0;
    int repair_attempts = 4;
    if (const char* e = std::getenv("HJCD_REPAIR_ATTEMPTS")) { int v = std::atoi(e); if (v >= 0 && v <= 64) repair_attempts = v; }
#if defined(HJCD_HAS_COLLISION)
    if (use_hard) cc_kernel_env = cc_env;
#endif

    // Coarse phase precision
    using TC = float;

    const int    B            = b_size;
    const size_t num_elems_x  = (size_t)B * N;
    const size_t num_elems_p7 = (size_t)B * 7;

    const auto* d_robotModel_f = cached_robot_model<TC>();

    TC *d_x_c=nullptr, *d_pose_c=nullptr, *d_pos_mm_c=nullptr, *d_ori_r_c=nullptr;

    allocations.allocate(d_x_c, sizeof(TC) * num_elems_x);
    allocations.allocate(d_pose_c, sizeof(TC) * num_elems_p7);
    allocations.allocate(d_pos_mm_c, sizeof(TC) * B);
    allocations.allocate(d_ori_r_c, sizeof(TC) * B);

    // init errors to +inf
    {
        thrust::device_ptr<TC> p(d_pos_mm_c), o(d_ori_r_c);
        thrust::fill(p, p + B, std::numeric_limits<TC>::infinity());
        thrust::fill(o, o + B, std::numeric_limits<TC>::infinity());
    }

    // float coarse targets, one row per candidate block
    TC* d_targets_coarse_c = upload_replicated_target<TC>(allocations, target_pose64, B);
    reset_stop_flag();
    unsigned char* d_coarse_free = nullptr;          // cc_stop only: per-block collision verdict
    if (cc_stop_coarse) allocations.allocate(d_coarse_free, (size_t)B);

    // COARSE SEARCH
    {
        // threads-per-block request ~ 2N warps
        int TPB_req = std::min((int)(2 * N * WARP_SIZE), 256);
        int maxThreadsPerBlock = 0;
        CUDA_OK(cudaDeviceGetAttribute(&maxThreadsPerBlock,
                                       cudaDevAttrMaxThreadsPerBlock, device));
        TPB_req = std::min(TPB_req, maxThreadsPerBlock);
        TPB_req = (TPB_req + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
        TPB_req = std::max(TPB_req, WARP_SIZE);

        // Per-warp scratch: the two anchor FK buffers, or (collision-aware stop) warp 0's sphere positions.
        const size_t perWarpBytes = std::max((size_t)(2 * NX * 16) * sizeof(TC),
                                             cc_stop_coarse ? (size_t)CC_WARP_POS_FLOATS * sizeof(float) : (size_t)0);

        cudaFuncAttributes attr{};
        CUDA_OK(cudaFuncGetAttributes(&attr, (const void*)coarse_search<TC>));
        const size_t staticShmem = (size_t)attr.sharedSizeBytes;

        int maxOptIn=0, maxDefault=0;
        CUDA_OK(cudaDeviceGetAttribute(&maxOptIn,
                                       cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        CUDA_OK(cudaDeviceGetAttribute(&maxDefault,
                                       cudaDevAttrMaxSharedMemoryPerBlock, device));
        size_t maxSharedAvail = (size_t)std::max(maxOptIn, maxDefault);

        size_t roomForDyn = (maxSharedAvail > staticShmem)
                            ? (maxSharedAvail - staticShmem) : 0;
        int maxWarpsBySmem = (perWarpBytes > 0)
                             ? (int)(roomForDyn / perWarpBytes) : 1;
        maxWarpsBySmem = std::max(1, maxWarpsBySmem);

        int reqWarps = TPB_req / WARP_SIZE;
        int warpsPerBlock = std::min(reqWarps, maxWarpsBySmem);
        warpsPerBlock = std::min(warpsPerBlock, 4);
        warpsPerBlock = std::max(1, warpsPerBlock);

        int TPB = warpsPerBlock * WARP_SIZE;
        size_t scratchBytes = (size_t)warpsPerBlock * perWarpBytes;

        int ask = (int)std::min(maxSharedAvail, staticShmem + scratchBytes);
        CUDA_OK(cudaFuncSetAttribute((const void*)coarse_search<TC>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     ask));

        for (;;) {
            coarse_search<TC><<<B, TPB, scratchBytes>>>(
                d_x_c, d_pose_c, d_targets_coarse_c,
                d_pos_mm_c, d_ori_r_c, d_robotModel_f,
                stop_on_first, cc_kernel_env, cc_stop_coarse, d_coarse_free
            );
            cudaError_t e = cudaGetLastError();
            if (e == cudaSuccess) break;
            if (e != cudaErrorLaunchOutOfResources) CUDA_OK(e);
            if (warpsPerBlock > 1) {
                warpsPerBlock >>= 1;
                TPB = warpsPerBlock * WARP_SIZE;
                scratchBytes = (size_t)warpsPerBlock * perWarpBytes;
                continue;
            }
            CUDA_OK(e);
            break;
        }
        CUDA_OK(cudaDeviceSynchronize());
    }

    std::vector<TC> h_pos_mm_coarse_f(B), h_ori_rad_coarse_f(B);
    std::vector<TC> h_pose_coarse_f(num_elems_p7), h_x_coarse_f(num_elems_x);
    CUDA_OK(cudaMemcpy(h_pos_mm_coarse_f.data(), d_pos_mm_c,
                       sizeof(TC) * B, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(h_ori_rad_coarse_f.data(), d_ori_r_c,
                       sizeof(TC) * B, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(h_pose_coarse_f.data(), d_pose_c,
                       sizeof(TC) * num_elems_p7, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(h_x_coarse_f.data(), d_x_c,
                       sizeof(TC) * num_elems_x, cudaMemcpyDeviceToHost));

    const auto sch = schedule_for_B(B);
    const auto top_k_req = static_cast<long long>(sch.top_k) * std::max(1, num_solutions / 2);
    const int repeats = sch.repeats;
    const double sigma_frac = sch.sigma_frac;

    // score: pos + ori error (float)
    TC* d_scores_c = nullptr;
    allocations.allocate(d_scores_c, sizeof(TC) * B);
    {
        const int tpb = 256, gpb = (B + tpb - 1) / tpb;
        build_scores_kernel<TC><<<gpb, tpb>>>(
            d_pos_mm_c, d_ori_r_c, d_coarse_free, (TC)COARSE_COLLIDING_RANK_PENALTY, d_scores_c, B);
        CUDA_OK(cudaGetLastError());
    }

    // sort configs and gather top K
    thrust::device_vector<int> d_idx(B);
    thrust::sequence(d_idx.begin(), d_idx.end(), 0);

    {
        thrust::device_ptr<TC> s_ptr(d_scores_c);
        thrust::sort_by_key(s_ptr, s_ptr + B, d_idx.begin());
    }

    // K in [1, B]
    const int K = static_cast<int>(std::clamp(top_k_req, 1LL, static_cast<long long>(B)));

    thrust::device_vector<int> d_top_idx(K);
    thrust::copy(d_idx.begin(), d_idx.begin() + K, d_top_idx.begin());

    TC* d_x_top_c = nullptr;
    allocations.allocate(d_x_top_c, sizeof(TC) * (size_t)K * N);
    {
        const int blocks = K, tpb = 128;
        gather_rows_kernel<TC><<<blocks, tpb>>>(
            d_x_c,
            thrust::raw_pointer_cast(d_top_idx.data()),
            d_x_top_c,
            K
        );
        CUDA_OK(cudaGetLastError());
    }

    const int Krep = K * repeats;
    TC* d_x_rep_c = nullptr;
    allocations.allocate(d_x_rep_c, sizeof(TC) * (size_t)Krep * N);
    {
        const int blocks = K, tpb = 128;
        replicate_rows_kernel<TC><<<blocks, tpb>>>(
            d_x_top_c, d_x_rep_c, K, N, repeats);
        CUDA_OK(cudaGetLastError());
    }
    {
        const int blocks = Krep, tpb = 128;
        perturb_rows_kernel<TC><<<blocks, tpb>>>(
            d_x_rep_c, Krep, (TC)sigma_frac, 0xC0FFEEull, repeats, /*skip_first_in_group=*/true);
        CUDA_OK(cudaGetLastError());
    }

    CUDA_OK(cudaDeviceSynchronize());

    // JACOBIAN LM TUNER — runs in the refine precision RT (double by default; float = the
    // fp32 speed knob). The Cholesky solve follows RT as well.
    // (Array names keep the "64" suffix for continuity; their element type is RT.)
    RT *dpose64=nullptr, *dposmm64=nullptr, *dori64=nullptr;
    const size_t KrepN = (size_t)Krep * N;
    const size_t Krep7 = (size_t)Krep * 7;

    allocations.allocate(dpose64, sizeof(RT) * Krep7);
    allocations.allocate(dposmm64, sizeof(RT) * Krep);
    allocations.allocate(dori64, sizeof(RT) * Krep);

    RT* dx64 = device_cast<RT>(allocations, d_x_rep_c, KrepN);          // coarse float -> RT
    RT* dtgt64 = upload_replicated_target<RT>(allocations, target_pose64, Krep);

    const auto* d_robotModel_rt = cached_robot_model<RT>();
    reset_stop_flag();

    {
        const int max_iters = HJCDSettings<RT>::lm_max_iters;
        const int stop_on_first_lm = (num_solutions > 1) ? 0 : 1;

        // Multi-warp LM: W independent candidates per block (one per warp). grid=ceil(Krep/W),
        // block=32*W, dynamic shared = W * sizeof(LMWarpScratch<RT>) (opt-in >48KB).
        // DEFAULT W=1 (fastest). MEASURED on the RTX 5090 (correct binary, 2026-06-18): W>1 is strictly
        // SLOWER here — up to 41% (ns=1) / 63% (ns=4) at W=8 — because bigger blocks (32*W threads,
        // W*~4KB smem) cut resident blocks/SM and the workload is far below GPU saturation, so there's no
        // occupancy gain to offset it. W>1 is retained ONLY as an opt-in (env HJCD_LM_WARPS) for low-SM
        // devices (e.g. Jetson) where few SMs saturate at low Krep — UNTESTED there. Clamp by Krep and by
        // device max opt-in shared memory.
        int W = 1;
        if (const char* e = std::getenv("HJCD_LM_WARPS")) { int v = std::atoi(e); if (v >= 1 && v <= 32) W = v; }
        if (Krep > 0 && W > Krep) W = Krep;
        int lm_dev = 0;
        CUDA_OK(cudaGetDevice(&lm_dev));
        int smem_optin = 48 * 1024;
        CUDA_OK(cudaDeviceGetAttribute(&smem_optin,
                                       cudaDevAttrMaxSharedMemoryPerBlockOptin, lm_dev));
        cudaFuncAttributes lm_attr{};
        CUDA_OK(cudaFuncGetAttributes(&lm_attr, (const void*)lm_tuner<RT>));
        int max_regs_per_block = 0;
        CUDA_OK(cudaDeviceGetAttribute(&max_regs_per_block,
                                       cudaDevAttrMaxRegistersPerBlock, lm_dev));
        auto lm_resources_fit = [&]() {
            const long long threads = (long long)WARP_SIZE * W;
            const long long registers = threads * lm_attr.numRegs;
            return threads <= lm_attr.maxThreadsPerBlock
                && (size_t)W * sizeof(LMWarpScratch<RT>) <= (size_t)smem_optin
                && (lm_attr.numRegs <= 0 || registers <= max_regs_per_block);
        };
        while (W > 1 && !lm_resources_fit()) W >>= 1;
        if (!lm_resources_fit())
            throw std::runtime_error("LM kernel does not fit device per-block resource limits");

        // Convergence / early-stop tolerance (see HJCDSettings); the env knobs let perf sweeps try
        // a precision-appropriate looser tolerance for fp32 refinement.
        RT eps_pos = HJCDSettings<RT>::lm_eps_pos, eps_ori = HJCDSettings<RT>::lm_eps_ori;
        if (const char* e = std::getenv("HJCD_LM_EPS_POS")) { double v = std::atof(e); if (v > 0) eps_pos = (RT)v; }
        if (const char* e = std::getenv("HJCD_LM_EPS_ORI")) { double v = std::atof(e); if (v > 0) eps_ori = (RT)v; }

        // Register pressure can make a requested warp count unlaunchable even when threads and
        // dynamic shared memory fit (especially in fp64). Downshift on
        // cudaErrorLaunchOutOfResources instead of continuing with uninitialized result buffers.
        for (;;) {
            const int TPB_lm = WARP_SIZE * W;
            const int grid_lm = (Krep + W - 1) / W;
            const size_t lm_smem = (size_t)W * sizeof(LMWarpScratch<RT>);
            if (lm_smem > (size_t)48 * 1024) {
                CUDA_OK(cudaFuncSetAttribute((const void*)lm_tuner<RT>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)lm_smem));
            }
            lm_tuner<RT><<<grid_lm, TPB_lm, lm_smem>>>(
                dx64, dpose64, dtgt64, dposmm64, dori64, d_robotModel_rt,
                eps_pos, eps_ori, HJCDSettings<RT>::lambda_init, max_iters, Krep, stop_on_first_lm,
                cc_kernel_env, cc_stop_lm, repair_attempts
            );
            cudaError_t launch_err = cudaGetLastError();
            if (launch_err == cudaSuccess) break;
            if (launch_err == cudaErrorLaunchOutOfResources && W > 1) {
                W >>= 1;
                continue;
            }
            CUDA_OK(launch_err);
        }
        CUDA_OK(cudaDeviceSynchronize());
    }

    // Host collision buffers are needed only for the requested check, not open-world solves.
    std::vector<float> h_env_cost_refined(use_soft ? Krep : 0);
    std::vector<float> h_env_cost_coarse(use_soft ? B : 0);
    std::vector<unsigned char> h_valid_refined(use_hard ? Krep : 0);
    std::vector<unsigned char> h_valid_coarse(use_hard ? B : 0);

#if defined(HJCD_HAS_COLLISION)
    if (do_cc) {
        // Both collision kernels read double q: the coarse pool (float) is cast; the refined pool
        // is reused in place when RT == double.
        double* dq_coarse = device_cast<double>(allocations, d_x_c, num_elems_x);
        double* dq_ref    = device_cast<double>(allocations, dx64, KrepN);
        // Dynamic smem for the multi_target FK extractor (shared by both collision kernels).
        const size_t cc_smem = grid::MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<float>();

        // Run one per-config collision kernel over the refined pool and the coarse pool, and read
        // both result arrays back to the host.
        auto score_pools = [&](auto kernel, auto& h_refined, auto& h_coarse) {
            using Out = typename std::decay_t<decltype(h_refined)>::value_type;
            Out* d_refined = nullptr;
            Out* d_coarse = nullptr;
            allocations.allocate(d_refined, sizeof(Out) * (size_t)Krep);
            allocations.allocate(d_coarse, sizeof(Out) * (size_t)B);
            CUDA_OK(cudaFuncSetAttribute((const void*)kernel,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cc_smem));
            kernel<<<Krep, CC_TPB, cc_smem>>>(dq_ref, Krep, d_refined, d_robotModel_cc, cc_env);
            kernel<<<B, CC_TPB, cc_smem>>>(dq_coarse, B, d_coarse, d_robotModel_cc, cc_env);
            CUDA_OK(cudaGetLastError());
            CUDA_OK(cudaDeviceSynchronize());
            CUDA_OK(cudaMemcpy(h_refined.data(), d_refined, sizeof(Out) * (size_t)Krep, cudaMemcpyDeviceToHost));
            CUDA_OK(cudaMemcpy(h_coarse.data(), d_coarse, sizeof(Out) * (size_t)B, cudaMemcpyDeviceToHost));
        };
        if (use_soft) score_pools(score_environment_costs, h_env_cost_refined, h_env_cost_coarse);
        if (use_hard) score_pools(mark_collisions, h_valid_refined, h_valid_coarse);
    }
#endif  // HJCD_HAS_COLLISION

    // Read the RT device results back into double host buffers (downstream stays fp64).
    std::vector<double> h_posmm64(Krep), h_orir64(Krep);
    std::vector<double> h_pose64(Krep7), h_x64(KrepN);
    {
        auto readback = [&](std::vector<double>& host, const RT* device, size_t n) {
            if constexpr (std::is_same_v<RT, double>) {
                CUDA_OK(cudaMemcpy(host.data(), device, sizeof(double) * n, cudaMemcpyDeviceToHost));
            } else {
                std::vector<RT> tmp(n);
                CUDA_OK(cudaMemcpy(tmp.data(), device, sizeof(RT) * n, cudaMemcpyDeviceToHost));
                std::copy(tmp.begin(), tmp.end(), host.begin());
            }
        };
        readback(h_posmm64, dposmm64, Krep);
        readback(h_orir64,  dori64,   Krep);
        readback(h_pose64,  dpose64,  Krep7);
        readback(h_x64,     dx64,     KrepN);
    }

    // GET SOLUTIONS. Candidates are addressed by a signed index: idx >= 0 is refined row idx,
    // idx < 0 is coarse row (-1 - idx). The refined pool is preferred; the coarse pool only tops
    // up when fewer than S_target distinct, valid refined candidates exist.
    const int S_target = std::min(num_solutions, Krep + B);
    auto coarse_row = [](int idx) { return -1 - idx; };
    auto pos_mm   = [&](int idx)->double { return idx >= 0 ? h_posmm64[idx] : (double)h_pos_mm_coarse_f[coarse_row(idx)]; };
    auto ori_rad  = [&](int idx)->double { return idx >= 0 ? h_orir64[idx]  : (double)h_ori_rad_coarse_f[coarse_row(idx)]; };
    auto env_cost = [&](int idx)->float  { return idx >= 0 ? h_env_cost_refined[idx] : h_env_cost_coarse[coarse_row(idx)]; };
    auto is_valid = [&](int idx)->bool   { return idx >= 0 ? bool(h_valid_refined[idx]) : bool(h_valid_coarse[coarse_row(idx)]); };
    auto joint_value = [&](int idx, int joint)->double {
        return idx >= 0 ? h_x64[(size_t)idx * N + joint]
                        : h_x_coarse_f[(size_t)coarse_row(idx) * N + joint];
    };
    auto score = [&](int idx)->double {
        double s = rank_score(pos_mm(idx), ori_rad(idx));
        if (use_soft) s += ENV_COLLISION_COST_W * (double)env_cost(idx);
        return std::isfinite(s) ? s : std::numeric_limits<double>::infinity();
    };
    const double DUP_TOL = 1e-7;
    auto is_dup = [&](int ia, int ib)->bool {
        for (int j = 0; j < N; ++j)
            if (std::fabs(joint_value(ia, j) - joint_value(ib, j)) > DUP_TOL)
                return false;
        return true;
    };

    std::vector<int> chosen;
    chosen.reserve(S_target);
    // Append the best-scoring, valid, non-duplicate candidates of one pool until S_target is reached.
    auto select_from = [&](std::vector<int> pool) {
        std::sort(pool.begin(), pool.end(), [&](int a, int b){ return score(a) < score(b); });
        for (int idx : pool) {
            if ((int)chosen.size() == S_target) break;
            if (!std::isfinite(score(idx)) || (use_hard && !is_valid(idx))) continue;
            if (std::any_of(chosen.begin(), chosen.end(),
                            [&](int previous) { return is_dup(idx, previous); })) continue;
            chosen.push_back(idx);
        }
    };
    {
        std::vector<int> refined(Krep);
        std::iota(refined.begin(), refined.end(), 0);
        select_from(std::move(refined));
    }
    if ((int)chosen.size() < S_target) {
        std::vector<int> coarse(B);
        for (int c = 0; c < B; ++c) coarse[c] = -1 - c;
        select_from(std::move(coarse));
    }

    if (write_stats) {
        // Diagnostic work stays off the normal solve path. Unmeasured values are -1,
        // not zero or "collision-free". In soft-only mode clearance is environment-only.
        constexpr double POS_THR_MM = 5.0, ORI_THR_RAD = 1e-3;
        auto accurate = [&](double pos, double ori) {
            return pos < POS_THR_MM && ori < ORI_THR_RAD;
        };
        auto collision_clear = [&](int idx) {
            if (!do_cc) return false;
            if (use_hard) return is_valid(idx);
            return use_soft && env_cost(idx) <= CC_SPHERE_MARGIN_MM;
        };

        int n_ik_good_ref = 0, n_coll_free_ref = 0, n_feasible_ref = 0, n_ik_lost = 0;
        float env_cost_min = std::numeric_limits<float>::infinity(), env_cost_max = 0;
        double env_cost_mean = 0;
        for (int i = 0; i < Krep; ++i) {
            const bool ik_good = accurate(h_posmm64[i], h_orir64[i]);
            const bool coll_free = collision_clear(i);
            n_ik_good_ref += ik_good;
            n_coll_free_ref += coll_free;
            n_feasible_ref += ik_good && coll_free;
            n_ik_lost += ik_good && !coll_free;
            if (use_soft) {
                const float cost = h_env_cost_refined[i];
                env_cost_min = std::min(env_cost_min, cost);
                env_cost_max = std::max(env_cost_max, cost);
                env_cost_mean += cost;
            }
        }
        if (use_soft) env_cost_mean /= Krep;
        const int n_cc_in_refined = Krep - n_coll_free_ref;
        int n_cc_in_coarse = 0;
        if (do_cc)
            for (int c = 0; c < B; ++c) n_cc_in_coarse += !collision_clear(-1 - c);

        int n_out_ik = 0, n_out_cf = 0, n_out_feasible = 0;
        for (int idx : chosen) {
            const bool ik_good = accurate(pos_mm(idx), ori_rad(idx));
            const bool coll_free = collision_clear(idx);
            n_out_ik += ik_good;
            n_out_cf += coll_free;
            n_out_feasible += ik_good && coll_free;
        }
        constexpr const char* CSV_PATH = "ik_stats.csv";
        std::ofstream csv(CSV_PATH, std::ios::app | std::ios::ate);
        if (!csv) throw std::runtime_error("cannot open ik_stats.csv for append");
        {
            if (csv.tellp() == std::streampos(0)) {
                csv << "b_size,krep"
                       ",n_ik_accurate,n_coll_free_refined,n_feasible,n_ik_lost"
                       ",n_coll_in_refined,n_coll_in_coarse"
                       ",env_cost_min_mm,env_cost_max_mm,env_cost_mean_mm"
                       ",n_returned,n_returned_ik_accurate,n_returned_coll_free"
                       ",pct_returned_coll_free,n_returned_feasible\n";
            }

            const double pct_cf = chosen.empty() ? 0.0
                                                  : 100.0 * n_out_cf / (double)chosen.size();
            csv << B           << ',' << Krep
                << ',' << n_ik_good_ref
                << ',' << (do_cc ? n_coll_free_ref : -1)
                << ',' << (do_cc ? n_feasible_ref : -1)
                << ',' << (do_cc ? n_ik_lost : -1)
                << ',' << (do_cc ? n_cc_in_refined : -1)
                << ',' << (do_cc ? n_cc_in_coarse  : -1)
                << ',' << (use_soft ? env_cost_min  : -1.f)
                << ',' << (use_soft ? env_cost_max  : -1.f)
                << ',' << (use_soft ? env_cost_mean : -1.0)
                << ',' << (int)chosen.size()
                << ',' << n_out_ik
                << ',' << (do_cc ? n_out_cf : -1)
                << ',' << (do_cc ? pct_cf : -1.0)
                << ',' << (do_cc ? n_out_feasible : -1)
                << '\n';
            csv.close();
            if (!csv) throw std::runtime_error("cannot write ik_stats.csv");
        }
    }

    // PACK OUTPUTS
    const int S = (int)chosen.size();
    result.pos_errors   = new T[S];
    result.ori_errors   = new T[S];
    result.pose         = new T[7 * S];
    result.joint_config = new T[N * S];
    result.count = S;

    for (int r = 0; r < S; ++r) {
        const int idx = chosen[r];
        result.pos_errors[r] = (T)pos_mm(idx);
        result.ori_errors[r] = (T)ori_rad(idx);
        for (int k = 0; k < 7; ++k)
            result.pose[r * 7 + k] = idx >= 0 ? (T)h_pose64[(size_t)idx * 7 + k]
                                              : (T)h_pose_coarse_f[(size_t)coarse_row(idx) * 7 + k];
        for (int j = 0; j < N; ++j)
            result.joint_config[(size_t)r * N + j] = (T)joint_value(idx, j);
    }

    allocations.clear();

    auto t1 = high_resolution_clock::now();
    result.elapsed_time =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    return result;
}

template Result<double> generate_ik_solutions<double>(   // RT=double (full fp64, default)
    double* target_pose,
    int b_size,
    int num_solutions,
    bool collision_free,
    const char* problems_json_text,
    const char* problem_set_name,
    int problem_idx,
    bool write_stats,
    int collision_mode
);

template Result<double> generate_ik_solutions<double, float>(   // RT=float (fp32 refine knob)
    double* target_pose,
    int b_size,
    int num_solutions,
    bool collision_free,
    const char* problems_json_text,
    const char* problem_set_name,
    int problem_idx,
    bool write_stats,
    int collision_mode
);

template Result<float> generate_ik_solutions<float>(
    float* target_pose,
    int b_size,
    int num_solutions,
    bool collision_free,
    const char* problems_json_text,
    const char* problem_set_name,
    int problem_idx,
    bool write_stats,
    int collision_mode
);

template std::vector<std::array<double, 7>> sample_random_target_poses(
    const grid::robotModel<double>* d_robotModel,
    int num_configs,
    uint64_t seed
);

template std::vector<std::array<float, 7>> sample_random_target_poses(
    const grid::robotModel<float>* d_robotModel,
    int num_configs,
    uint64_t seed
);
