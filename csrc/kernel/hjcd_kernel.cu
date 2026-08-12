#include "kernel/hjcd_kernel.h"
#include "kernel/hjcd_settings.h"
#include "kernel/util.h"
#include "kernel/device_utils.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/copy.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
#include <vector>
#include <chrono>
#include <fstream>
#include <limits>
#include <type_traits>
#include <cstdlib>

// Collision checking (grid_collision: URDF-driven spheres baked into grid.cuh)
#include <nlohmann/json.hpp>
#include "kernel/grid_env.cuh"   // requires grid.cuh (via hjcd_settings.h) already included above

// Self-collision sidecar, HARD mode (Checkpoint 3D/3E). Pulled into the SOLVER TU only for the two
// __noinline__ hot-path entry points (sidecar_hard_trial / sidecar_hard_restore). The 177-register
// batched full checker stays where it was, in src/collision_sidecar.cu, and is never inlined here.
#include "collision_sidecar_hard.cuh"

enum : int {
    N = grid::NUM_JOINTS
};
extern "C" int grid_num_joints() { return N; }

// The sidecar collision model is G1-specific (29 joints, 40 links, hashed URDF). A build whose
// grid.cuh is a DIFFERENT robot keeps hard mode compiled but permanently unavailable -- silently
// checking one robot's geometry against another's kinematics is the failure mode this guards.
static constexpr bool HJCD_HARD_AVAILABLE = (N == g1sc::N_JOINTS);
extern "C" int hjcd_hard_available() { return HJCD_HARD_AVAILABLE ? 1 : 0; }

// SEPARABLE_COMPILATION is OFF, so this TU has its OWN copies of the sidecar's `g_sdf` / `g_cverts`
// device pointer symbols, distinct from the ones src/collision_sidecar.cu writes at upload time.
// Both must point at the SAME device allocations or the hot path would dereference null. The host
// binds them here once, from the pointers the sidecar TU already owns.
extern "C" void hjcd_hard_bind_model(const void* const* sdf_ptrs, int n_sdf, const void* cverts) {
    for (int i = 0; i < n_sdf; ++i)
        CUDA_OK(cudaMemcpyToSymbol(g1sc::g_sdf, &sdf_ptrs[i], sizeof(void*),
                                   (size_t)i * sizeof(void*)));
    CUDA_OK(cudaMemcpyToSymbol(g1sc::g_cverts, &cverts, sizeof(void*)));
}

constexpr int FLANGE_IDX = N + 1;
constexpr int EE_IDX     = N;
constexpr int NX         = FLANGE_IDX + 1;

__constant__ double2 c_joint_limits[N];
// fp32 MIRROR of the joint limits. Without it every fp32 kernel would load a double2 from constant
// memory and cvt.f64.f32 it on the hot path -- FP64-pipe work on a GPU where FP64 runs at 1/64 rate,
// and, at the two perturbation sites, an actual double subtraction ((T)(L.y - L.x)) before the cast.
// The mirror keeps the fp32 path free of FP64 entirely. Both copies are written from the SAME
// validated doubles, so the fp64 path is bit-for-bit unchanged.
__constant__ float2  c_joint_limits_f[N];

// Joint limits in the kernel's own compute type. Specialised, never converted.
template<typename T> __device__ __forceinline__ void joint_limit(int j, T* lo, T* hi);
template<> __device__ __forceinline__ void joint_limit<double>(int j, double* lo, double* hi) {
    const double2 L = c_joint_limits[j];   *lo = L.x; *hi = L.y;
}
template<> __device__ __forceinline__ void joint_limit<float>(int j, float* lo, float* hi) {
    const float2 L = c_joint_limits_f[j];  *lo = L.x; *hi = L.y;
}

// GRiD HELPER FUNCTIONS
namespace grid {
  template<typename T>
  T* init_joint_limits();
}


void init_joint_limits_from_grid()
{
    double* d_limits = grid::init_joint_limits<double>();

    std::vector<double> h_limits(2 * N);
    CUDA_OK(cudaMemcpy(h_limits.data(), d_limits,
                       sizeof(double) * 2 * N, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaFree(d_limits));

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

    // fp32 mirror, narrowed from the SAME validated doubles.
    std::vector<float2> packed_f(N);
    for (int j = 0; j < N; ++j)
        packed_f[j] = make_float2((float)packed[j].x, (float)packed[j].y);
    CUDA_OK(cudaMemcpyToSymbol(c_joint_limits_f, packed_f.data(),
                               sizeof(float2) * N));
}

template<typename T>
__device__ void sample_joint_config(T* s_x, int local_problem, int global_problem) {
    // 5D.14c (semantic_problem_rng_v2). This helper has ZERO callers today; normalized anyway so
    // no compiled stochastic path is left on the legacy contract. `global_problem` is the semantic
    // candidate index of the enclosing 1-D launch (one block per candidate), NOT launch geometry.
    const int offset = local_problem * N;

#pragma unroll
    for (int j = 0; j < N; ++j) {
        uint32_t sj = semantic_rng((uint32_t)global_problem, RNG_SUB_INITIAL_JOINT_SAMPLE,
                                   (uint32_t)local_problem, 0u, (uint32_t)j, 0u);

        float r = u01(sj);
        float low = (float)c_joint_limits[j].x;
        float hi = (float)c_joint_limits[j].y;

        float v = fmaf(r, (hi - low), low);
        s_x[offset + j] = (T)v;
    }
}

template<typename T>
__device__ void perturb_joint_config(T* s_x, int global_problem, T sigma_frac = (T)0.05) {
    // 5D.14c: identity is (semantic candidate, joint). `global_problem` == blockIdx.x by the
    // enclosing kernel's one-block-per-candidate construction, which is a SEMANTIC id, not grid
    // geometry; there is no outer P dimension on this legacy path, so no slot dependence exists.
#pragma unroll
    for (int j = 0; j < N; ++j) {
        uint32_t sj = semantic_rng((uint32_t)global_problem, RNG_SUB_PO_CCD_STALL_PERTURBATION,
                                   0u, 0u, (uint32_t)j, 0u);

        float low = (float)c_joint_limits[j].x;
        float hi = (float)c_joint_limits[j].y;
        float range = hi - low;

        float step = (float)sigma_frac * range * gauss01(sj);

        float v = (float)s_x[j] + step;
        v = fminf(hi, fmaxf(low, v));
        s_x[j] = (T)v;
    }
}

// MATH HELPERS
template<typename T>
__device__ __forceinline__
void mat_to_quat(const T* __restrict__ C, T* __restrict__ q) {
    const T m00 = C[0], m01 = C[4], m02 = C[8];
    const T m10 = C[1], m11 = C[5], m12 = C[9];
    const T m20 = C[2], m21 = C[6], m22 = C[10];

    const T trace = m00 + m11 + m22;
    const T eps = (T)1e-20;

    if (trace > (T)0) {
        T r = sqrt(fmax((T)1 + trace, eps));
        T s = (T)0.5 / r;
        q[0] = (T)0.5 * r;
        q[1] = (m21 - m12) * s;
        q[2] = (m02 - m20) * s;
        q[3] = (m10 - m01) * s;
    }
    else if (m00 >= m11 && m00 >= m22) {
        T r = sqrt(fmax((T)1 + m00 - m11 - m22, eps));
        T s = (T)0.5 / r;
        q[1] = (T)0.5 * r;
        q[0] = (m21 - m12) * s;
        q[2] = (m01 + m10) * s;
        q[3] = (m02 + m20) * s;
    }
    else if (m11 >= m22) {
        T r = sqrt(fmax((T)1 - m00 + m11 - m22, eps));
        T s = (T)0.5 / r;
        q[2] = (T)0.5 * r;
        q[0] = (m02 - m20) * s;
        q[1] = (m01 + m10) * s;
        q[3] = (m12 + m21) * s;
    }
    else {
        T r = sqrt(fmax((T)1 - m00 - m11 + m22, eps));
        T s = (T)0.5 / r;
        q[3] = (T)0.5 * r;
        q[0] = (m10 - m01) * s;
        q[1] = (m02 + m20) * s;
        q[2] = (m12 + m21) * s;
    }

    T n = rsqrt(fmax(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3], eps));
    q[0] *= n; q[1] *= n; q[2] *= n; q[3] *= n;
}

template<typename T>
__device__ void multiply_quat(const T* r, const T* s, T* t) {
    t[0] = r[0] * s[0] - r[1] * s[1] - r[2] * s[2] - r[3] * s[3];
    t[1] = r[0] * s[1] + r[1] * s[0] - r[2] * s[3] + r[3] * s[2];
    t[2] = r[0] * s[2] + r[1] * s[3] + r[2] * s[0] - r[3] * s[1];
    t[3] = r[0] * s[3] - r[1] * s[2] + r[2] * s[1] + r[3] * s[0];
}

template<typename T>
__device__ void normalize_quat(T* quat) {
    T norm = sqrt(quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);
    if (norm > 1e-6f) {
        quat[0] /= norm;
        quat[1] /= norm;
        quat[2] /= norm;
        quat[3] /= norm;
    }
}

template<typename T>
__device__ __forceinline__ void quat_conj(const T* q, T* qc) {
    qc[0] = q[0]; qc[1] = -q[1]; qc[2] = -q[2]; qc[3] = -q[3];
}
template<typename T>
__device__ __forceinline__ void quat_mul(const T* a, const T* b, T* o) {
    o[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    o[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    o[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
    o[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

template<typename T>
__device__ __forceinline__ void quat_err_rotvec(const T* q_cur, const T* q_goal, T* w_err3) {
    T qc[4], qe[4];
    quat_conj(q_cur, qc);
    quat_mul(q_goal, qc, qe);
    T n = rsqrt(qe[0] * qe[0] + qe[1] * qe[1] + qe[2] * qe[2] + qe[3] * qe[3]);
    qe[0] *= n; qe[1] *= n; qe[2] *= n; qe[3] *= n;
    T vnorm = sqrt(qe[1] * qe[1] + qe[2] * qe[2] + qe[3] * qe[3]);
    T cw = fabs(qe[0]);
    T theta = (vnorm > (T)1e-12) ? (T)2 * atan2(vnorm, cw) : (T)0;
    if (theta < (T)1e-12) { w_err3[0] = w_err3[1] = w_err3[2] = (T)0; return; }
    T s = theta / vnorm;
    w_err3[0] = s * qe[1]; w_err3[1] = s * qe[2]; w_err3[2] = s * qe[3];
}

template<typename T>
__device__ void normalize_vec3(T* vec) {
    T norm = sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
    if (norm > 1e-6) {
        vec[0] /= norm;
        vec[1] /= norm;
        vec[2] /= norm;
    }
}

template<typename T>
__device__ T compute_ori_err(const T* CjX, const T* q_goal) {
    T qee[4];
    mat_to_quat(&CjX[EE_IDX*16], qee);
    if (qee[0]*q_goal[0]+qee[1]*q_goal[1]+qee[2]*q_goal[2]+qee[3]*q_goal[3] < (T)0) {
        qee[0]=-qee[0]; qee[1]=-qee[1]; qee[2]=-qee[2]; qee[3]=-qee[3];
    }
    T wv[3]; quat_err_rotvec(qee, q_goal, wv);
    return sqrt(wv[0]*wv[0] + wv[1]*wv[1] + wv[2]*wv[2]);
}

template<typename T>
__device__ T compute_pos_err(const T* C, const T* target_pose) {
    const T dx = C[EE_IDX * 16 + 12] - target_pose[0];
    const T dy = C[EE_IDX * 16 + 13] - target_pose[1];
    const T dz = C[EE_IDX * 16 + 14] - target_pose[2];
    return sqrt(dx * dx + dy * dy + dz * dz);
}

// Variants taking a standalone 16-cell EE transform (offset 0) rather than a full chain
// indexed at EE_IDX — used by the lane-parallel coarse candidates, which keep each
// candidate's EE pose in a per-lane register buffer instead of a shared frame array.
template<typename T>
__device__ __forceinline__ T compute_pos_err_at(const T* ee16, const T* target_pose) {
    const T dx = ee16[12] - target_pose[0];
    const T dy = ee16[13] - target_pose[1];
    const T dz = ee16[14] - target_pose[2];
    return sqrt(dx * dx + dy * dy + dz * dz);
}

template<typename T>
__device__ __forceinline__ T compute_ori_err_at(const T* ee16, const T* q_goal) {
    T qee[4];
    mat_to_quat(ee16, qee);
    if (qee[0]*q_goal[0]+qee[1]*q_goal[1]+qee[2]*q_goal[2]+qee[3]*q_goal[3] < (T)0) {
        qee[0]=-qee[0]; qee[1]=-qee[1]; qee[2]=-qee[2]; qee[3]=-qee[3];
    }
    T wv[3]; quat_err_rotvec(qee, q_goal, wv);
    return sqrt(wv[0]*wv[0] + wv[1]*wv[1] + wv[2]*wv[2]);
}

// SOLVE
template<typename T>
__device__ T solve_pos(const T* s_jointXforms, const T* pos, const T* target_pose_local, int joint, int k, int k_max, T delta_min = 0.35, T delta_max = 0.75) {
    T joint_pos[3] = {
        s_jointXforms[joint * 16 + 12],
        s_jointXforms[joint * 16 + 13],
        s_jointXforms[joint * 16 + 14]
    };

    T r[3];
    joint_world_axis<T>(&s_jointXforms[joint * 16], joint, r);
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
    dotp = clamp_dot(dotp);
    T theta = acos(dotp);

    T cx = uproj[1] * vproj[2] - uproj[2] * vproj[1];
    T cy = uproj[2] * vproj[0] - uproj[0] * vproj[2];
    T cz = uproj[0] * vproj[1] - uproj[1] * vproj[0];

    T sign = r[0] * cx + r[1] * cy + r[2] * cz;
    if (sign < 0)
        theta = -theta;

    T delta = 0.75 + 0.25 * (1.0 - T(k) / T(k_max));
    T step = theta * delta;
    step = clamp_step_angle(step);
    return step;
}

template<typename T>
__device__ T solve_ori(const T* s_jointXforms, const T* q_t, int joint, int k, int k_max) {

    T r[3];
    joint_world_axis<T>(&s_jointXforms[joint * 16], joint, r);
    normalize_vec3(r);

    T q_ee[4];
    mat_to_quat(&s_jointXforms[EE_IDX * 16], q_ee);
    normalize_quat(q_ee);

    T q_ee_inv[4] = { q_ee[0], -q_ee[1], -q_ee[2], -q_ee[3] };
    T q_err[4];
    multiply_quat(q_t, q_ee_inv, q_err);
    normalize_quat(q_err);

    T theta = 2.0f * acos(clamp_dot(fabs(q_err[0])));
    T sin_h = sin(theta / 2.0f);
    T a[3] = { 1, 0, 0 };

    if (theta > 1e-3f) {
        a[0] = q_err[1] / sin_h;
        a[1] = q_err[2] / sin_h;
        a[2] = q_err[3] / sin_h;

        T norm = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
        if (norm > 1e-6f) {
            a[0] /= norm;
            a[1] /= norm;
            a[2] /= norm;
        }
    }

    T sign = a[0] * r[0] + a[1] * r[1] + a[2] * r[2];
    if (sign < 0)
        theta = -theta;

    T delta = 0.75 + 0.25 * (1.0 - T(k) / T(k_max));
    T step = theta * delta;
    step = clamp_step_angle(step);
    return step;
}

// JACOBIAN TUNER
__device__ __forceinline__ void upper_index_to_rc(int idx, int DIM, int& r, int& c) {
    int acc = 0;
    for (int rr = 0; rr < DIM; ++rr) {
        int rowCount = DIM - rr;
        if (idx < acc + rowCount) { r = rr; c = rr + (idx - acc); return; }
        acc += rowCount;
    }
    r = c = 0;
}

template<typename T>
__device__ __forceinline__ T safe_normN(const T* v, int n) {
    T s = (T)0; for (int i = 0; i < n; ++i) s += v[i] * v[i]; return sqrt(s);
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
    bool stop_on_first
) {
    const int gp   = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps_per_block = max(1, (int)(blockDim.x >> 5));

    if (!x || !pose || !targetsB || !pos_errors || !ori_errors || !d_robotModel) return;
    if (warps_per_block == 0) return;

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

    __shared__ T s_x[N];
    __shared__ T s_pose[7];
    __shared__ T s_glob_pos_err, s_glob_ori_err;

    __shared__ T s_pos_theta1[N], s_ori_theta1[N];
    __shared__ T s_pos_err[N],    s_ori_err[N];

    __shared__ T s_XmatsHom[grid::XHOM_T_COUNT];
    __shared__ T s_jointXforms[NX*16];
    __shared__ T s_temp[NX*2];
    __shared__ int s_topology_helpers[hjcd::TOPO];

    const T* target_pose_local = &targetsB[gp * 7];
    const T q_t[4] = { target_pose_local[3], target_pose_local[4],
                       target_pose_local[5], target_pose_local[6] };

    if (tid == 0) { s_last_joint_o = -1; s_last_joint_p = -1; }
    __syncthreads();

    // Random initial config in limits
    if (tid < N) {
        // 5D.14c: (semantic candidate gp, joint tid). One block per candidate => gp is semantic.
        uint32_t st = semantic_rng((uint32_t)gp, RNG_SUB_PER_THREAD_INITIAL_CONFIG,
                                   0u, 0u, (uint32_t)tid, 0u);
        float r = u01(st);
        const double2 L = c_joint_limits[tid];
        s_x[tid] = (T)(L.x + r * (L.y - L.x));
        s_pos_theta1[tid] = (T)0;
        s_ori_theta1[tid] = (T)0;
        s_pos_err[tid]    = (T)1e9;
        s_ori_err[tid]    = (T)1e9;
    }
    __syncthreads();

    grid::load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_x, d_robotModel, s_temp);
    __syncthreads();

    if ((threadIdx.x >> 5) == 0) { // warp 0
        ee_fk_warp<T>(s_jointXforms, s_XmatsHom, s_x, EE_IDX);
    }
    __syncthreads();

    if (tid == 0) {
        s_glob_pos_err = compute_pos_err(s_jointXforms, target_pose_local);
        s_glob_ori_err = compute_ori_err(s_jointXforms, q_t);

        T q_ee[4];
        mat_to_quat(&s_jointXforms[EE_IDX * 16], q_ee);
        normalize_quat(q_ee);
        s_pose[0] = s_jointXforms[EE_IDX * 16 + 12];
        s_pose[1] = s_jointXforms[EE_IDX * 16 + 13];
        s_pose[2] = s_jointXforms[EE_IDX * 16 + 14];
        s_pose[3] = q_ee[0]; s_pose[4] = q_ee[1]; s_pose[5] = q_ee[2]; s_pose[6] = q_ee[3];
    }
    __syncthreads();

    for (int k = 0; k < HJCDSettings<T>::k_max; ++k) {
        if (stop_on_first && tid == 0) s_stop = read_stop();
        __syncthreads();
        if (stop_on_first && s_stop) break;

        if ((threadIdx.x >> 5) == 0) { // warp 0
            ee_fk_warp<T>(s_jointXforms, s_XmatsHom, s_x, EE_IDX);
        }
        __syncthreads();

        if (tid == 0) {
            T q_ee[4];
            mat_to_quat(&s_jointXforms[EE_IDX * 16], q_ee);
            normalize_quat(q_ee);
            s_pose[0] = s_jointXforms[EE_IDX * 16 + 12];
            s_pose[1] = s_jointXforms[EE_IDX * 16 + 13];
            s_pose[2] = s_jointXforms[EE_IDX * 16 + 14];
            s_pose[3] = q_ee[0]; s_pose[4] = q_ee[1]; s_pose[5] = q_ee[2]; s_pose[6] = q_ee[3];
        }
        __syncthreads();

        if (tid == 0) {
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
            // PARALLEL across the warp's lanes (strided, so N > 32 / humanoids are supported),
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

            // warp min-reduce over candidates; tie-break to the LOWEST j to match the serial
            // "first strict-improvement wins" selection exactly.
            #pragma unroll
            for (int off = WARP_SIZE >> 1; off > 0; off >>= 1) {
                const T   o_err = __shfl_down_sync(FULL_WARP_MASK, best_err_lane, off);
                const int o_j   = __shfl_down_sync(FULL_WARP_MASK, best_j_lane,   off);
                if (o_err < best_err_lane ||
                    (o_err == best_err_lane && o_j >= 0 && (best_j_lane < 0 || o_j < best_j_lane))) {
                    best_err_lane = o_err; best_j_lane = o_j;
                }
            }
            best_err_lane = __shfl_sync(FULL_WARP_MASK, best_err_lane, 0);
            best_j_lane   = __shfl_sync(FULL_WARP_MASK, best_j_lane,   0);

            if (lane == 0) {
                if (pos_phase) s_pos_err[p] = best_err_lane;
                else           s_ori_err[p] = best_err_lane;
            }
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

            T q_ee[4];
            mat_to_quat(&s_jointXforms[EE_IDX * 16], q_ee);
            normalize_quat(q_ee);
            s_pose[0] = s_jointXforms[EE_IDX * 16 + 12];
            s_pose[1] = s_jointXforms[EE_IDX * 16 + 13];
            s_pose[2] = s_jointXforms[EE_IDX * 16 + 14];
            s_pose[3] = q_ee[0]; s_pose[4] = q_ee[1]; s_pose[5] = q_ee[2]; s_pose[6] = q_ee[3];

            for (int jj = 0; jj < N; ++jj) {
                s_pos_err[jj] = s_glob_pos_err;
                s_ori_err[jj] = s_glob_ori_err;
            }

            if (stop_on_first && s_glob_pos_err < HJCDSettings<T>::epsilon && s_glob_ori_err < HJCDSettings<T>::nu) {
                int old = atomicCAS(&g_stop, 0, 1);
                if (old == 0) { __threadfence(); g_winner = gp; }
            }
        }
        __syncthreads();

        if (tid == 0) s_stop = read_stop();
        __syncthreads();
        if (s_stop) break;

        if (tid < N) x[gp * N + tid] = s_x[tid];
    }

    if (tid < N) x[gp * N + tid] = s_x[tid];
    if (tid < 7) pose[gp * 7 + tid] = s_pose[tid];
    if (tid == 0) {
        pos_errors[gp] = s_glob_pos_err * (T)1000.0;
        ori_errors[gp] = s_glob_ori_err;
    }
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

template <typename T>
__global__ void forward_kinematics_kernel(
    const T* __restrict__ q,
    T* __restrict__ ee_pose7,
    T* __restrict__ all_link_T,
    const grid::robotModel<T>* __restrict__ RM,
    const int B)
{
    const int b = blockIdx.x;
    if (!q || !RM || b >= B) return;

    __shared__ T s_q[N];
    __shared__ T s_XmatsHom[grid::XHOM_T_COUNT];
    __shared__ T s_jointX[NX * 16];
    __shared__ T s_tmp[NX * 2];
    __shared__ int s_topology_helpers[hjcd::TOPO];

    for (int j = threadIdx.x; j < N; j += blockDim.x)
        s_q[j] = q[b * N + j];
    __syncthreads();

    grid::load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, RM, s_tmp);
    __syncthreads();

    if (threadIdx.x == 0) {
        ee_fk_thread<T>(s_jointX, s_XmatsHom, s_q, EE_IDX, s_XmatsHom);

        if (ee_pose7) {
            const T* Cee = &s_jointX[EE_IDX * 16];
            T qee[4];
            mat_to_quat(Cee, qee);
            ee_pose7[b * 7 + 0] = Cee[12];
            ee_pose7[b * 7 + 1] = Cee[13];
            ee_pose7[b * 7 + 2] = Cee[14];
            ee_pose7[b * 7 + 3] = qee[0];
            ee_pose7[b * 7 + 4] = qee[1];
            ee_pose7[b * 7 + 5] = qee[2];
            ee_pose7[b * 7 + 6] = qee[3];
        }

        if (all_link_T) {
            T* out = &all_link_T[b * (NX * 16)];
#pragma unroll
            for (int i = 0; i < NX * 16; ++i) out[i] = s_jointX[i];
        }
    }
}

// World transforms of every movable joint, for a batch of configs. Slots 0..N-1 are GRiD's
// cumulative world transform of joint j (column-major 4x4); slot EE_IDX is the appended
// single-target grasptarget frame (only meaningful on a build whose FLANGE_JID/GRASP_FIXED_IDX
// match the robot — i.e. the single-EE serial-chain robots, not G1). Test/introspection entry
// point: it is what tests/test_joint_axis.py checks the FK and the joint-axis metadata against.
template<typename T>
std::vector<T> compute_link_transforms(const T* h_q, int B, const grid::robotModel<T>* d_robotModel)
{
    std::vector<T> out((size_t)B * NX * 16, (T)0);
    if (!h_q || !d_robotModel || B <= 0) return out;

    T *d_q = nullptr, *d_T = nullptr;
    CUDA_OK(cudaMalloc(&d_q, sizeof(T) * (size_t)B * N));
    CUDA_OK(cudaMalloc(&d_T, sizeof(T) * (size_t)B * NX * 16));
    CUDA_OK(cudaMemcpy(d_q, h_q, sizeof(T) * (size_t)B * N, cudaMemcpyHostToDevice));

    forward_kinematics_kernel<T><<<B, 32>>>(d_q, nullptr, d_T, d_robotModel, B);
    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());

    CUDA_OK(cudaMemcpy(out.data(), d_T, sizeof(T) * (size_t)B * NX * 16, cudaMemcpyDeviceToHost));
    cudaFree(d_q);
    cudaFree(d_T);
    return out;
}

extern "C" int grid_num_frames() { return NX; }

// ---------------------------------------------------------------------------
// Multi-target frame FK (Phase 1B): ONE full GRiD FK, then compose every target frame from it.
// Nothing here touches the solver; it is the kinematic layer the later phases build on, and the
// entry point the target-FK tests validate against the numpy URDF oracle.
// ---------------------------------------------------------------------------
template<typename T>
__global__ void target_fk_kernel(
    const T* __restrict__ q,
    T* __restrict__ out_target_T,      // B x NT x 16, column-major 4x4 per target
    const grid::robotModel<T>* __restrict__ RM,
    const int B)
{
    const int b = blockIdx.x;
    if (!q || !RM || !out_target_T || b >= B) return;

    __shared__ T s_q[N];
    __shared__ T s_XmatsHom[grid::XHOM_T_COUNT];
    __shared__ T s_jointX[N * 16];              // world transform of every movable joint
    __shared__ T s_target_X[hjcd::NT * 16];     // the composed target frames (warp-shared, not per-lane)
    __shared__ T s_tmp[NX * 2];
    __shared__ int s_topology_helpers[hjcd::TOPO];

    for (int j = threadIdx.x; j < N; j += blockDim.x) s_q[j] = q[(size_t)b * N + j];
    __syncthreads();

    grid::load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, RM, s_tmp);
    __syncthreads();

    grid::ee_pose_inner_warp<T>(s_jointX, s_XmatsHom, s_q, N - 1);
    __syncwarp(FULL_WARP_MASK);

    compose_target_frames_warp<T>(s_target_X, s_jointX);

    for (int i = threadIdx.x; i < hjcd::NT * 16; i += blockDim.x)
        out_target_T[(size_t)b * hjcd::NT * 16 + i] = s_target_X[i];
}

template<typename T>
std::vector<T> compute_target_transforms(const T* h_q, int B, const grid::robotModel<T>* d_robotModel)
{
    std::vector<T> out((size_t)B * hjcd::NT * 16, (T)0);
    if (!h_q || !d_robotModel || B <= 0) return out;

    T *d_q = nullptr, *d_T = nullptr;
    CUDA_OK(cudaMalloc(&d_q, sizeof(T) * (size_t)B * N));
    CUDA_OK(cudaMalloc(&d_T, sizeof(T) * out.size()));
    CUDA_OK(cudaMemcpy(d_q, h_q, sizeof(T) * (size_t)B * N, cudaMemcpyHostToDevice));

    target_fk_kernel<T><<<B, 32>>>(d_q, d_T, d_robotModel, B);
    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());

    CUDA_OK(cudaMemcpy(out.data(), d_T, sizeof(T) * out.size(), cudaMemcpyDeviceToHost));
    cudaFree(d_q);
    cudaFree(d_T);
    return out;
}

// Read the generated metadata back out of DEVICE code. The arrays are __device__ constexpr, so a
// host-side copy would be a different object and could silently drift from what the kernels use;
// this dumps exactly what the hot path sees. Test-only, off every solver path.
__global__ void dump_target_metadata_kernel(int* __restrict__ anchor,
                                            unsigned int* __restrict__ tgt_anc_mask,
                                            unsigned int* __restrict__ joint_tgt_mask,
                                            double* __restrict__ tool)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int k = 0; k < hjcd::NT; ++k) {
        anchor[k] = hjcd_gen::TARGET_ANCHOR_JID[k];
        tgt_anc_mask[k] = hjcd_gen::TARGET_ANCESTOR_MASK[k];
        for (int i = 0; i < 16; ++i) tool[16 * k + i] = hjcd_gen::TARGET_TOOL_XFORM[16 * k + i];
    }
    for (int j = 0; j < N; ++j) joint_tgt_mask[j] = hjcd_gen::JOINT_TARGET_MASK[j];
}

TargetMetadata read_target_metadata()
{
    TargetMetadata m;
    m.num_targets = hjcd::NT;
    m.num_joints = N;
    m.anchor_jid.resize(hjcd::NT);
    m.target_ancestor_mask.resize(hjcd::NT);
    m.joint_target_mask.resize(N);
    m.tool_xform.resize((size_t)hjcd::NT * 16);

    int* d_a = nullptr; unsigned int *d_tam = nullptr, *d_jtm = nullptr; double* d_tool = nullptr;
    CUDA_OK(cudaMalloc(&d_a, sizeof(int) * hjcd::NT));
    CUDA_OK(cudaMalloc(&d_tam, sizeof(unsigned int) * hjcd::NT));
    CUDA_OK(cudaMalloc(&d_jtm, sizeof(unsigned int) * N));
    CUDA_OK(cudaMalloc(&d_tool, sizeof(double) * hjcd::NT * 16));

    dump_target_metadata_kernel<<<1, 1>>>(d_a, d_tam, d_jtm, d_tool);
    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());

    CUDA_OK(cudaMemcpy(m.anchor_jid.data(), d_a, sizeof(int) * hjcd::NT, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(m.target_ancestor_mask.data(), d_tam, sizeof(unsigned int) * hjcd::NT,
                       cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(m.joint_target_mask.data(), d_jtm, sizeof(unsigned int) * N,
                       cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(m.tool_xform.data(), d_tool, sizeof(double) * hjcd::NT * 16,
                       cudaMemcpyDeviceToHost));
    cudaFree(d_a); cudaFree(d_tam); cudaFree(d_jtm); cudaFree(d_tool);
    return m;
}

extern "C" int grid_num_targets() { return hjcd::NT; }

// The joint limits the solver actually clamps to -- i.e. c_joint_limits, including the +-PI fallback
// applied to non-finite URDF limits. Exposed so callers (and tests) can sample the exact feasible
// set the LM enforces; sampling outside it silently produces infeasible targets.
std::vector<double> get_joint_limits()
{
    init_joint_limits_from_grid();
    std::vector<double2> packed(N);
    CUDA_OK(cudaMemcpyFromSymbol(packed.data(), c_joint_limits, sizeof(double2) * N));
    std::vector<double> out(2 * N);
    for (int j = 0; j < N; ++j) { out[2*j] = packed[j].x; out[2*j + 1] = packed[j].y; }
    return out;
}

// ---------------------------------------------------------------------------
// Multi-target residual layer (Phase 2). Small, reusable __device__ helpers plus a standalone
// diagnostic kernel. Nothing here is wired into lm_tuner / coarse_search yet -- those kernels are
// already at the 255-register cap on G1 and are Phase 3/5 work.
//
// CONVENTIONS (fixed here; Phase 3's Jacobian must match):
//   quaternions        WXYZ, unit (the host normalizes; device does not renormalize the target)
//   position residual  e_p = p* - p                                    (world frame, metres)
//   orientation resid. e_R = Log(R* R^T)  == rotvec(q* (x) q^-1)       (WORLD frame, radians)
// e_R is the SPATIAL (world) error, matching the existing quat_err_rotvec and pairing with a
// world-frame angular Jacobian Jw = axis_world. A body-frame Log(R^T R*) would NOT pair with it.
// Both residuals are UNWEIGHTED physical quantities; weights enter only in the cost below.
// ---------------------------------------------------------------------------

// Unweighted residual of one target frame. X is its world 4x4 (column-major).
template<typename T>
__device__ __forceinline__
void target_residual(const T* __restrict__ X, const T* __restrict__ tgt_p,
                     const T* __restrict__ tgt_q, T* __restrict__ e_p, T* __restrict__ e_R)
{
    e_p[0] = tgt_p[0] - X[12];
    e_p[1] = tgt_p[1] - X[13];
    e_p[2] = tgt_p[2] - X[14];

    T qee[4];
    mat_to_quat(X, qee);
    // Double cover: q and -q are the same rotation. Flip the CURRENT quat into the target's
    // hemisphere so quat_err_rotvec's fabs(qe.w) picks the short way round (it assumes this).
    if (qee[0]*tgt_q[0] + qee[1]*tgt_q[1] + qee[2]*tgt_q[2] + qee[3]*tgt_q[3] < (T)0) {
        qee[0] = -qee[0]; qee[1] = -qee[1]; qee[2] = -qee[2]; qee[3] = -qee[3];
    }
    quat_err_rotvec(qee, tgt_q, e_R);
}

// Weighted cost of one target from its UNWEIGHTED residual norms:
//     c_k = w_p * |e_p|^2 + w_R * |e_R|^2
// Residuals are never pre-scaled by sqrt(w) -- weights are applied exactly once, here.
template<typename T>
__device__ __forceinline__
T target_cost(T pnorm, T onorm, T wp, T wo) {
    return wp * pnorm * pnorm + wo * onorm * onorm;
}

// Normalization denominator epsilon (reporting only; never divides anything in an optimizer).
template<typename T> __device__ __forceinline__ T cost_norm_eps() { return (T)1e-12; }

// Standalone diagnostic kernel: one block (one warp) per problem. Full FK -> target frames ->
// per-target unweighted residuals -> weighted costs -> per-target and all-target success.
// Inactive targets write exactly zero everywhere (never NaN).
template<typename T>
__global__ void target_residual_kernel(
    const T* __restrict__ q,                 // B x N
    const T* __restrict__ tgt_p,             // B x NT x 3
    const T* __restrict__ tgt_q,             // B x NT x 4   (wxyz, unit)
    const unsigned int* __restrict__ active, // B            (bit k = target k active)
    const T* __restrict__ w_pos,             // B x NT
    const T* __restrict__ w_ori,             // B x NT
    const T* __restrict__ eps_pos,           // NT
    const T* __restrict__ eps_ori,           // NT
    T* __restrict__ out_e_pos,               // B x NT x 3
    T* __restrict__ out_e_ori,               // B x NT x 3
    T* __restrict__ out_pnorm,               // B x NT
    T* __restrict__ out_onorm,               // B x NT
    T* __restrict__ out_cost,                // B x NT
    T* __restrict__ out_cost_raw,            // B
    T* __restrict__ out_cost_norm,           // B
    unsigned char* __restrict__ out_succ,    // B x NT
    unsigned char* __restrict__ out_succ_all,// B
    const grid::robotModel<T>* __restrict__ RM,
    const int B)
{
    const int b = blockIdx.x;
    if (b >= B) return;

    __shared__ T s_q[N];
    __shared__ T s_XmatsHom[grid::XHOM_T_COUNT];
    __shared__ T s_jointX[N * 16];
    __shared__ T s_target_X[hjcd::NT * 16];
    __shared__ T s_tmp[NX * 2];
    __shared__ int s_topology_helpers[hjcd::TOPO];
    __shared__ T s_cost[hjcd::NT];
    __shared__ unsigned char s_ok[hjcd::NT];

    for (int j = threadIdx.x; j < N; j += blockDim.x) s_q[j] = q[(size_t)b * N + j];
    __syncthreads();

    grid::load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topology_helpers, s_q, RM, s_tmp);
    __syncthreads();
    grid::ee_pose_inner_warp<T>(s_jointX, s_XmatsHom, s_q, N - 1);
    __syncwarp(FULL_WARP_MASK);
    compose_target_frames_warp<T>(s_target_X, s_jointX);

    const unsigned int mask = active[b];
    const int lane = threadIdx.x & 31;

    // One lane per target (NT <= MAX_TARGETS = 4): 6 scalars of residual live per lane, no
    // per-target arrays. Lanes >= NT idle.
    if (lane < hjcd::NT) {
        const int k = lane;
        const size_t o3 = ((size_t)b * hjcd::NT + k) * 3;
        const size_t o1 = (size_t)b * hjcd::NT + k;
        const bool on = (mask >> k) & 1u;

        T e_p[3] = {(T)0, (T)0, (T)0};
        T e_R[3] = {(T)0, (T)0, (T)0};
        T pn = (T)0, on_ = (T)0, c = (T)0;
        unsigned char ok = 0;

        if (on) {
            target_residual<T>(&s_target_X[16 * k], &tgt_p[o3], &tgt_q[((size_t)b * hjcd::NT + k) * 4],
                               e_p, e_R);
            pn  = sqrt(e_p[0]*e_p[0] + e_p[1]*e_p[1] + e_p[2]*e_p[2]);
            on_ = sqrt(e_R[0]*e_R[0] + e_R[1]*e_R[1] + e_R[2]*e_R[2]);
            const T wp = w_pos[o1], wo = w_ori[o1];
            c = target_cost<T>(pn, on_, wp, wo);
            // A zero-weight channel is "don't care", not "must be zero".
            const bool pos_ok = (wp == (T)0) || (pn  <= eps_pos[k]);
            const bool ori_ok = (wo == (T)0) || (on_ <= eps_ori[k]);
            ok = (pos_ok && ori_ok) ? 1 : 0;
        }
        // Inactive => exactly zero residual/norm/cost, success 0 (not evaluated).
        out_e_pos[o3 + 0] = e_p[0]; out_e_pos[o3 + 1] = e_p[1]; out_e_pos[o3 + 2] = e_p[2];
        out_e_ori[o3 + 0] = e_R[0]; out_e_ori[o3 + 1] = e_R[1]; out_e_ori[o3 + 2] = e_R[2];
        out_pnorm[o1] = pn;
        out_onorm[o1] = on_;
        out_cost[o1]  = c;
        out_succ[o1]  = ok;
        s_cost[k] = c;
        s_ok[k] = ok;
    }
    __syncwarp(FULL_WARP_MASK);

    if (lane == 0) {
        T raw = (T)0, wsum = (T)0;
        unsigned char all_ok = (mask != 0u) ? 1 : 0;   // an empty mask never "succeeds"
        #pragma unroll
        for (int k = 0; k < hjcd::NT; ++k) {
            if (!((mask >> k) & 1u)) continue;
            raw  += s_cost[k];
            wsum += w_pos[(size_t)b * hjcd::NT + k] + w_ori[(size_t)b * hjcd::NT + k];
            if (!s_ok[k]) all_ok = 0;
        }
        out_cost_raw[b]  = raw;
        // Reporting only. Constant per problem, so it cannot change the minimizer -- the optimizer
        // (Phase 3) will consume out_cost_raw and never divide by this.
        out_cost_norm[b] = raw / (wsum + cost_norm_eps<T>());
        out_succ_all[b]  = all_ok;
    }
}

// ---------------------------------------------------------------------------
// Phase 3: direct normal-equation accumulation.
//
//   e(q + dq) ~= e(q) - J dq        =>       (J^T W J + lambda D) dq = J^T W e
//
// A and b are accumulated TARGET BY TARGET; the stacked 6K x N Jacobian is never formed:
//   A_ij += w_p,k Jv_k,i . Jv_k,j  +  w_R,k Jw_k,i . Jw_k,j
//   b_i  += w_p,k Jv_k,i . e_p,k   +  w_R,k Jw_k,i . e_R,k
//
// Lane i owns joint i's column and holds its SIX Jacobian scalars in registers; other lanes'
// columns arrive by __shfl_sync. Only one target's Jacobian is live at a time -- there is no
// [K][6][N] array anywhere, and no 6*N scratch buffer.
//
// TARGET_ANCESTOR_MASK is a CORRECTNESS constraint, not an optimization: a joint that is not an
// ancestor of target k has an exactly-zero column for k. On a branched robot, filling it with
// axis x (p_target - p_joint) would invent motion that cannot happen.
// ---------------------------------------------------------------------------
// Row-norm preconditioner  s_{k,r} = 1 / ||J_{k,r}||  (r = 0..5, the six rows of target k's
// Jacobian), i.e. the DERIVED default scaling recovered from the old single-target LM.
//
// The old LM scaled BOTH residual and Jacobian rows by s: r~ = S e, J~ = S J, then formed
// A = J~^T J~ = J^T S^2 J and b = J~^T r~ = J^T S^2 e. So it was already solving J^T W J dq = J^T W e
// with W = S^2 -- weights applied exactly once, exactly the Phase-3 formulation. The scaling is
// therefore not an extra heuristic to be dropped; it IS the old default weighting, and dropping it
// is what broke Panda: without it A = J^T J is badly scaled (position rows carry metres-scale lever
// arms, orientation rows carry unit axes), the gain ratio runs far above 0.9 every iteration, lambda
// collapses to ~1e-11, and the rank-deficient Panda system (6 task rows, 7 joints) then yields a huge
// null-space step that the trust region scales away. Measured mean ||Jv_row||^2 = 0.39 vs
// ||Jw_row||^2 = 2.33, i.e. s_p^2/s_R^2 ~ 6.
//
// User weights multiply on top: W_{k,r} = w_{p|R,k} * s_{k,r}^2. The scaling is frozen for the whole
// iteration (computed once, reused by the trial evaluations) so that accept/reject compares like with
// like -- exactly as the old LM's frozen row_s did.
template<typename T>
__device__ __forceinline__
void compute_row_scales_warp(
    T* __restrict__ s_scale,                 // NT*6 out
    const T* __restrict__ s_jointX,
    const T* __restrict__ s_target_X,
    unsigned int active)
{
    const unsigned m = FULL_WARP_MASK;
    const int lane = threadIdx.x & 31;

    #pragma unroll
    for (int k = 0; k < hjcd::NT; ++k) {
        if (!((active >> k) & 1u)) {
            if (lane < 6) s_scale[6 * k + lane] = (T)1;
            continue;
        }
        const unsigned int amask = hjcd_gen::TARGET_ANCESTOR_MASK[k];
        const T* __restrict__ Xt = &s_target_X[16 * k];
        const T px = Xt[12], py = Xt[13], pz = Xt[14];

        T c[6] = {(T)0,(T)0,(T)0,(T)0,(T)0,(T)0};
        if ((lane < N) && ((amask >> lane) & 1u)) {
            const T* __restrict__ Ci = &s_jointX[16 * lane];
            T ax[3];
            joint_world_axis<T>(Ci, lane, ax);
            const bool prism = grid::HAS_PRISMATIC && grid::JOINT_IS_PRISMATIC[lane];
            const T rx = px - Ci[12], ry = py - Ci[13], rz = pz - Ci[14];
            c[0] = prism ? ax[0] : (ax[1]*rz - ax[2]*ry);
            c[1] = prism ? ax[1] : (ax[2]*rx - ax[0]*rz);
            c[2] = prism ? ax[2] : (ax[0]*ry - ax[1]*rx);
            c[3] = prism ? (T)0 : ax[0];
            c[4] = prism ? (T)0 : ax[1];
            c[5] = prism ? (T)0 : ax[2];
        }
        #pragma unroll
        for (int r = 0; r < 6; ++r) {
            T v = c[r] * c[r];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(m, v, off);
            v = __shfl_sync(m, v, 0);                       // ||J_{k,r}||^2 on every lane
            if (lane == r) s_scale[6 * k + r] = (v > (T)1e-18) ? rsqrt(v) : (T)1;
        }
        __syncwarp(m);
    }
    __syncwarp(m);
}

// Weighted cost using the frozen row scaling:  c_k = sum_r W_{k,r} * e_{k,r}^2,
// W_{k,r} = (r < 3 ? w_p,k : w_R,k) * s_{k,r}^2.
template<typename T>
__device__ __forceinline__
void weighted_cost_warp(const T* __restrict__ s_e_pos, const T* __restrict__ s_e_ori,
                        const T* __restrict__ s_wp, const T* __restrict__ s_wo,
                        const T* __restrict__ s_scale, unsigned int active,
                        T* __restrict__ out_cost)
{
    const int lane = threadIdx.x & 31;
    if (lane == 0) {
        T c = (T)0;
        #pragma unroll
        for (int k = 0; k < hjcd::NT; ++k) {
            if (!((active >> k) & 1u)) continue;
            #pragma unroll
            for (int r = 0; r < 3; ++r) {
                const T sp = s_scale[6*k + r], so = s_scale[6*k + 3 + r];
                c += s_wp[k] * sp * sp * s_e_pos[3*k + r] * s_e_pos[3*k + r]
                   + s_wo[k] * so * so * s_e_ori[3*k + r] * s_e_ori[3*k + r];
            }
        }
        *out_cost = c;
    }
    __syncwarp(FULL_WARP_MASK);
}

template<typename T>
__device__ __forceinline__
void accumulate_normal_equations_warp(
    T* __restrict__ A,                       // N*N shared, row-major (lane i owns row i)
    T* __restrict__ b,                       // N   shared
    const T* __restrict__ s_jointX,          // N*16 world joint transforms
    const T* __restrict__ s_target_X,        // NT*16 target frames
    const T* __restrict__ s_e_pos,           // NT*3 unweighted position residuals
    const T* __restrict__ s_e_ori,           // NT*3 unweighted orientation residuals
    const T* __restrict__ s_wp,              // NT
    const T* __restrict__ s_wo,              // NT
    const T* __restrict__ s_scale,           // NT*6 row preconditioner (nullptr => all ones)
    unsigned int active)
{
    const unsigned m = FULL_WARP_MASK;
    const int lane = threadIdx.x & 31;

    for (int idx = lane; idx < N * N; idx += WARP_SIZE) A[idx] = (T)0;
    if (lane < N) b[lane] = (T)0;
    __syncwarp(m);

    #pragma unroll
    for (int k = 0; k < hjcd::NT; ++k) {
        if (!((active >> k) & 1u)) continue;                  // inactive target: contributes nothing
        const T wp = s_wp[k], wo = s_wo[k];
        if (wp == (T)0 && wo == (T)0) continue;               // fully unweighted: contributes nothing

        // Per-ROW weights: W_{k,r} = w * s_{k,r}^2.  s == nullptr => unit scaling (the raw J^T W J
        // reference used by normal_equations(), which the CPU stacked-Jacobian test compares against).
        T Wp[3], Wo[3];
        #pragma unroll
        for (int r = 0; r < 3; ++r) {
            const T sp = s_scale ? s_scale[6*k + r]     : (T)1;
            const T so = s_scale ? s_scale[6*k + 3 + r] : (T)1;
            Wp[r] = wp * sp * sp;
            Wo[r] = wo * so * so;
        }

        const unsigned int amask = hjcd_gen::TARGET_ANCESTOR_MASK[k];
        const T* __restrict__ Xt = &s_target_X[16 * k];
        const T px = Xt[12], py = Xt[13], pz = Xt[14];

        // This lane's column of J_k. Non-ancestor => exactly zero (not "small").
        T jv0 = (T)0, jv1 = (T)0, jv2 = (T)0, jw0 = (T)0, jw1 = (T)0, jw2 = (T)0;
        const bool mine = (lane < N) && ((amask >> lane) & 1u);
        if (mine) {
            const T* __restrict__ Ci = &s_jointX[16 * lane];
            T ax[3];
            joint_world_axis<T>(Ci, lane, ax);
            const bool prism = grid::HAS_PRISMATIC && grid::JOINT_IS_PRISMATIC[lane];
            const T rx = px - Ci[12], ry = py - Ci[13], rz = pz - Ci[14];
            jv0 = prism ? ax[0] : (ax[1] * rz - ax[2] * ry);
            jv1 = prism ? ax[1] : (ax[2] * rx - ax[0] * rz);
            jv2 = prism ? ax[2] : (ax[0] * ry - ax[1] * rx);
            jw0 = prism ? (T)0 : ax[0];
            jw1 = prism ? (T)0 : ax[1];
            jw2 = prism ? (T)0 : ax[2];

            b[lane] += Wp[0]*jv0*s_e_pos[3*k+0] + Wp[1]*jv1*s_e_pos[3*k+1] + Wp[2]*jv2*s_e_pos[3*k+2]
                     + Wo[0]*jw0*s_e_ori[3*k+0] + Wo[1]*jw1*s_e_ori[3*k+1] + Wo[2]*jw2*s_e_ori[3*k+2];
        }

        // Row i of A: broadcast lane j's column to every lane, then lane i accumulates A[i][j].
        // ALL lanes must reach the shuffles (they are warp-collective), so the mask guard is on the
        // accumulate, not on the loop.
        for (int j = 0; j < N; ++j) {
            const T ov0 = __shfl_sync(m, jv0, j);
            const T ov1 = __shfl_sync(m, jv1, j);
            const T ov2 = __shfl_sync(m, jv2, j);
            const T ow0 = __shfl_sync(m, jw0, j);
            const T ow1 = __shfl_sync(m, jw1, j);
            const T ow2 = __shfl_sync(m, jw2, j);
            if (mine)
                A[lane * N + j] += Wp[0]*jv0*ov0 + Wp[1]*jv1*ov1 + Wp[2]*jv2*ov2
                                 + Wo[0]*jw0*ow0 + Wo[1]*jw1*ow1 + Wo[2]*jw2*ow2;
        }
        __syncwarp(m);
    }
    __syncwarp(m);
}

// Per-target residuals of the CURRENT config into shared, plus the raw weighted cost.
// Reuses the Phase-2 helpers verbatim: residuals stay UNWEIGHTED; weights enter only in the cost.
template<typename T>
__device__ __forceinline__
void eval_targets_warp(
    const T* __restrict__ s_target_X,
    const T* __restrict__ s_tgt_p, const T* __restrict__ s_tgt_q,
    const T* __restrict__ s_wp, const T* __restrict__ s_wo,
    unsigned int active,
    T* __restrict__ s_e_pos, T* __restrict__ s_e_ori,
    T* __restrict__ s_pn, T* __restrict__ s_on,
    T* __restrict__ out_cost)
{
    const unsigned m = FULL_WARP_MASK;
    const int lane = threadIdx.x & 31;

    if (lane < hjcd::NT) {
        const int k = lane;
        T e_p[3] = {(T)0,(T)0,(T)0}, e_R[3] = {(T)0,(T)0,(T)0};
        T pn = (T)0, on = (T)0;
        if ((active >> k) & 1u) {
            target_residual<T>(&s_target_X[16*k], &s_tgt_p[3*k], &s_tgt_q[4*k], e_p, e_R);
            pn = sqrt(e_p[0]*e_p[0] + e_p[1]*e_p[1] + e_p[2]*e_p[2]);
            on = sqrt(e_R[0]*e_R[0] + e_R[1]*e_R[1] + e_R[2]*e_R[2]);
        }
        s_e_pos[3*k+0]=e_p[0]; s_e_pos[3*k+1]=e_p[1]; s_e_pos[3*k+2]=e_p[2];
        s_e_ori[3*k+0]=e_R[0]; s_e_ori[3*k+1]=e_R[1]; s_e_ori[3*k+2]=e_R[2];
        s_pn[k] = pn;
        s_on[k] = on;
    }
    __syncwarp(m);

    if (lane == 0) {
        T c = (T)0;
        #pragma unroll
        for (int k = 0; k < hjcd::NT; ++k)
            if ((active >> k) & 1u)
                c += target_cost<T>(s_pn[k], s_on[k], s_wp[k], s_wo[k]);
        *out_cost = c;                       // RAW weighted cost: what acceptance and rho use.
    }
    __syncwarp(m);
}

// Per-target cost under the frozen row preconditioner:
//     c_k = sum_r  w_p,k * s_{k,r}^2   * e_p,k[r]^2
//         + sum_r  w_R,k * s_{k,3+r}^2 * e_R,k[r]^2
// s_scale == nullptr collapses this to the plain w_p|e_p|^2 + w_R|e_R|^2 of Phase 2 (which is what
// the residual/incremental diagnostic paths still use).
template<typename T>
__device__ __forceinline__
T scaled_target_cost(const T* e_p, const T* e_R, T wp, T wo, const T* s_scale, int k) {
    if (!s_scale) {
        const T pn2 = e_p[0]*e_p[0] + e_p[1]*e_p[1] + e_p[2]*e_p[2];
        const T on2 = e_R[0]*e_R[0] + e_R[1]*e_R[1] + e_R[2]*e_R[2];
        return wp * pn2 + wo * on2;
    }
    T c = (T)0;
    #pragma unroll
    for (int r = 0; r < 3; ++r) {
        const T sp = s_scale[6*k + r], so = s_scale[6*k + 3 + r];
        c += wp * sp * sp * e_p[r] * e_p[r] + wo * so * so * e_R[r] * e_R[r];
    }
    return c;
}

// ---------------------------------------------------------------------------
// Phase 4: incremental target state.
//
// Recompute residual + cost for the AFFECTED targets only, and fold the total cost incrementally:
//     C_new = C_old - sum_{k in A_j} c_k^old + sum_{k in A_j} c_k^new
// Unaffected targets are never read, never written, never rescored -- their cached state stays
// bitwise identical.  `tmask` must already be (JOINT_TARGET_MASK[j] & active_target_mask).
// ---------------------------------------------------------------------------
template<typename T>
__device__ __forceinline__
void eval_targets_masked_warp(
    const T* __restrict__ s_target_X,
    const T* __restrict__ s_tgt_p, const T* __restrict__ s_tgt_q,
    const T* __restrict__ s_wp, const T* __restrict__ s_wo,
    const T* __restrict__ s_scale,     // frozen row preconditioner; nullptr => unit weights
    unsigned int tmask,
    T* __restrict__ s_e_pos, T* __restrict__ s_e_ori,
    T* __restrict__ s_pn, T* __restrict__ s_on, T* __restrict__ s_ck,
    T* __restrict__ io_total)
{
    const int lane = threadIdx.x & 31;

    if (lane == 0) {                       // drop the OLD contributions of the affected targets
        T sub = (T)0;
        #pragma unroll
        for (int k = 0; k < hjcd::NT; ++k)
            if ((tmask >> k) & 1u) sub += s_ck[k];
        *io_total -= sub;
    }
    __syncwarp(FULL_WARP_MASK);

    if (lane < hjcd::NT && ((tmask >> lane) & 1u)) {
        const int k = lane;
        T e_p[3], e_R[3];
        target_residual<T>(&s_target_X[16*k], &s_tgt_p[3*k], &s_tgt_q[4*k], e_p, e_R);
        const T pn = sqrt(e_p[0]*e_p[0] + e_p[1]*e_p[1] + e_p[2]*e_p[2]);
        const T on = sqrt(e_R[0]*e_R[0] + e_R[1]*e_R[1] + e_R[2]*e_R[2]);
        s_e_pos[3*k+0]=e_p[0]; s_e_pos[3*k+1]=e_p[1]; s_e_pos[3*k+2]=e_p[2];
        s_e_ori[3*k+0]=e_R[0]; s_e_ori[3*k+1]=e_R[1]; s_e_ori[3*k+2]=e_R[2];
        s_pn[k] = pn;
        s_on[k] = on;
        s_ck[k] = scaled_target_cost<T>(e_p, e_R, s_wp[k], s_wo[k], s_scale, k);
    }
    __syncwarp(FULL_WARP_MASK);

    if (lane == 0) {                       // add the NEW contributions
        T add = (T)0;
        #pragma unroll
        for (int k = 0; k < hjcd::NT; ++k)
            if ((tmask >> k) & 1u) add += s_ck[k];
        *io_total += add;
    }
    __syncwarp(FULL_WARP_MASK);
}

// Full per-target eval that also fills the per-target cost cache (the incremental path's baseline).
template<typename T>
__device__ __forceinline__
void eval_targets_full_warp(
    const T* __restrict__ s_target_X,
    const T* __restrict__ s_tgt_p, const T* __restrict__ s_tgt_q,
    const T* __restrict__ s_wp, const T* __restrict__ s_wo,
    const T* __restrict__ s_scale,     // frozen row preconditioner; nullptr => unit weights
    unsigned int active,
    T* __restrict__ s_e_pos, T* __restrict__ s_e_ori,
    T* __restrict__ s_pn, T* __restrict__ s_on, T* __restrict__ s_ck,
    T* __restrict__ out_total)
{
    const int lane = threadIdx.x & 31;
    if (lane < hjcd::NT) {
        const int k = lane;
        T e_p[3] = {(T)0,(T)0,(T)0}, e_R[3] = {(T)0,(T)0,(T)0};
        T pn = (T)0, on = (T)0, ck = (T)0;
        if ((active >> k) & 1u) {
            target_residual<T>(&s_target_X[16*k], &s_tgt_p[3*k], &s_tgt_q[4*k], e_p, e_R);
            pn = sqrt(e_p[0]*e_p[0] + e_p[1]*e_p[1] + e_p[2]*e_p[2]);
            on = sqrt(e_R[0]*e_R[0] + e_R[1]*e_R[1] + e_R[2]*e_R[2]);
            ck = scaled_target_cost<T>(e_p, e_R, s_wp[k], s_wo[k], s_scale, k);
        }
        s_e_pos[3*k+0]=e_p[0]; s_e_pos[3*k+1]=e_p[1]; s_e_pos[3*k+2]=e_p[2];
        s_e_ori[3*k+0]=e_R[0]; s_e_ori[3*k+1]=e_R[1]; s_e_ori[3*k+2]=e_R[2];
        s_pn[k]=pn; s_on[k]=on; s_ck[k]=ck;
    }
    __syncwarp(FULL_WARP_MASK);
    if (lane == 0) {
        T c = (T)0;
        #pragma unroll
        for (int k = 0; k < hjcd::NT; ++k)
            if ((active >> k) & 1u) c += s_ck[k];
        *out_total = c;
    }
    __syncwarp(FULL_WARP_MASK);
}

// Worst active per-target errors -> convergence. Maximum, never mean: a solved left hand must not
// hide an unsolved right foot. A zero-weight channel is "don't care" and is excluded.
template<typename T>
__device__ __forceinline__
void worst_active_errors(const T* s_pn, const T* s_on, const T* s_wp, const T* s_wo,
                         unsigned int active, T* out_maxp, T* out_maxo)
{
    T mp = (T)0, mo = (T)0;
    #pragma unroll
    for (int k = 0; k < hjcd::NT; ++k) {
        if (!((active >> k) & 1u)) continue;
        if (s_wp[k] > (T)0 && s_pn[k] > mp) mp = s_pn[k];
        if (s_wo[k] > (T)0 && s_on[k] > mo) mo = s_on[k];
    }
    *out_maxp = mp;
    *out_maxo = mo;
}

template<typename T>
__device__ __forceinline__
bool all_active_converged(const T* s_pn, const T* s_on, const T* s_wp, const T* s_wo,
                          unsigned int active, T eps_p, T eps_o)
{
    if (active == 0u) return false;
    #pragma unroll
    for (int k = 0; k < hjcd::NT; ++k) {
        if (!((active >> k) & 1u)) continue;
        if (s_wp[k] > (T)0 && !(s_pn[k] <= eps_p)) return false;
        if (s_wo[k] > (T)0 && !(s_on[k] <= eps_o)) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Phase 3: multi-target LM refine. ONE LM for every K -- the single-target public path routes
// through this with K=1 and mask bit 0 set; there is no second LM implementation.
//
// Per iteration: full GRiD FK -> compose active target frames -> Phase-2 residuals -> all-target
// convergence check -> direct A/b accumulation -> lambda*diag(A) damping + one N x N warp-Cholesky
// -> backtracking trial -> accept/reject + lambda update.
//
// Damping is UNCHANGED from the single-target LM: the same lambda*diag(A) Levenberg shift through
// glass::warp::posv<REG_DIAG>, the same backtracking schedule (1, 1/2, 1/4, 1/8), the same gain-
// ratio thresholds (0.9/0.5/0.25) and factors (0.3/0.5/3.0), the same stall limit and kick.
// What changed is only WHAT is being minimized: the raw weighted multi-target cost.
// ---------------------------------------------------------------------------
// One LM trace row. Column 0 is an EXPLICIT validity flag -- row validity is never inferred from
// cost, lambda or the iteration index (a converged row legitimately has cost == 0, and it == 0 is a
// real first row). Every public LM diagnostic is derived from these rows.
//   0 valid   1 it   2 lm_trials(cum)   3 accepted(this iter)   4 accepted(cum)
//   5 cost    6 max_pos_err   7 max_ori_err   8 lambda   9 line_searches(cum)
static constexpr int TRACE_COLS = 11;   // ... + col 10 = E_phys (tolerance-normalised PHYSICAL
                                        // error, the only cross-iteration-comparable metric here)

// E_phys(q) = sum_k [ |e_p,k|^2 / eps_p^2  +  |e_R,k|^2 / eps_R^2 ]  over ACTIVE targets.
//
// The row-scaled cost CANNOT be compared across iterations: the row scales s_{k,r} = 1/||J_{k,r}||
// are re-frozen every iteration, so C^(t) and C^(t-1) are expressed in different units and a
// "relative improvement" between them is not a real quantity. E_phys is built from the physical
// residual norms and fixed tolerances, so it is stable across iterations. It is ANALYSIS/STOPPING
// only -- nothing in the optimizer ever reads it.
template<typename T>
__device__ __forceinline__ T e_phys(const T* s_pn, const T* s_on, unsigned int active,
                                    T eps_p, T eps_o) {
    T e = (T)0;
    #pragma unroll
    for (int k = 0; k < hjcd::NT; ++k) {
        if (!((active >> k) & 1u)) continue;
        const T rp = s_pn[k] / eps_p, ro = s_on[k] / eps_o;
        e += rp * rp + ro * ro;
    }
    return e;
}

// Controls for the alternating base update (Architecture B). Passed BY VALUE to the kernel: it is
// 14 scalars, and a pointer would cost a global read per candidate for data every candidate shares.
// enabled == 0 makes the whole feature a branch that is never taken, which is how the fixed-base
// path stays bit-identical.
template<typename T>
struct BaseUpdateCfg {
    int enabled = 0;
    int interval = 1;             // take a base step every `interval` LM iterations
    T damping = (T)1e-3;          // lambda in H + lambda*D   (see base_update_warp)
    // D = diag(s_p^-2 I3, s_R^-2 I3): the metric lambda is measured in. MUST match
    // hjcdik/base_update.py's damping_matrix() -- the host reference is the oracle for this
    // solve, and it is only an oracle while both solve the same system.
    T damping_scale_p = (T)1;     // s_p, metres  (> 0)
    T damping_scale_R = (T)1;     // s_R, radians (> 0)
    T step_scale = (T)1;          // alpha
    T max_translation = (T)0.05;  // m,  clipped independently of rotation
    T max_rotation = (T)0.10;     // rad
    T lo[3] = {(T)-1e30, (T)-1e30, (T)-1e30};   // base position bounds; +-1e30 == unbounded
    T hi[3] = {(T)1e30, (T)1e30, (T)1e30};
};

template<typename T>
struct LMScratch {
    T s_x[N], x_old[N], best_x[N];
    T s_XmatsHom[grid::XHOM_T_COUNT];
    T s_jointX[N * 16];
    T s_target_X[hjcd::NT * 16];
    T s_tgt_p[hjcd::NT * 3], s_tgt_q[hjcd::NT * 4];   // in the CANDIDATE'S BASE FRAME
    // Floating base, per candidate (hjcd::FLOATING_BASE_DOF). s_tgt_p/s_tgt_q above are stored
    // in THIS base's frame -- the only way the base enters the solver. The FK, the coordinate
    // machinery and the cost never see it. Identity for a fixed-base solve, which then reduces
    // to exactly the previous bytes.
    T s_base_p[3], s_base_q[4];
    // best_x tracks the best JOINTS seen; with a moving base the best BASE must travel with them
    // or the returned joints would be paired with whatever base happened to be current at the end.
    T best_base_p[3], best_base_q[4];
    T bak_base_p[3], bak_base_q[4];    // exact rollback of a rejected base step
    T s_Hb[hjcd::FLOATING_BASE_DOF * hjcd::FLOATING_BASE_DOF];   // 6x6 normal matrix, col-major
    T s_bb[hjcd::FLOATING_BASE_DOF];                             // 6 rhs
    // eval_targets_full_warp writes s_ck[k] and *out_total UNCONDITIONALLY -- it has no null
    // guards -- so the base re-score must hand it real storage. Passing nullptr was a device-side
    // null write and surfaced as "unspecified launch failure" at the next sync.
    T s_ck_base[hjcd::NT], s_total_base;
    int s_base_fail;                   // posv non-PD flag for the 6x6 solve
    int base_att, base_acc, base_numfail;   // diagnostics (attempted / accepted / gave up)
    T s_wp[hjcd::NT], s_wo[hjcd::NT];
    T s_e_pos[hjcd::NT * 3], s_e_ori[hjcd::NT * 3];
    T s_pn[hjcd::NT], s_on[hjcd::NT];
    T s_scale[hjcd::NT * 6];   // frozen row preconditioner for this iteration
    T A[N * N], b[N], g[N], dq[N], diagA[N];   // g = J^T W e, saved before posv overwrites b
    T cost, trial_cost, best_cost, prev_cost;
    T best_ephys;          // cross-iteration best-state metric (STABLE, physical)
    int take_best;         // lane-0 decision, broadcast to the warp
    T max_pn, max_on;
    unsigned int active;
    // Diagnostics. SEMANTICS (Phase 3C):
    //   lm_iterations  outer LM linearizations = Jacobian rebuilds = normal-equation assemblies.
    //   lm_trials      damped linear systems SOLVED, including ones whose step was rejected. One
    //                  posv per outer iteration, so lm_trials == lm_iterations here; it is tracked
    //                  separately because the backtracking line search inside an iteration evaluates
    //                  several step LENGTHS without re-solving, and Phase 5 may re-solve per trial.
    //   line_searches  cost evaluations inside the backtracking loop (step lengths tried).
    //   accepted_steps / rejected_steps  outer iterations whose step was taken / not taken.
    // A problem that is already converged on entry does ZERO outer iterations and reports all zeros.
    int s_break, stall, accepted, s_fail, kicked;
    int stag_n;            // consecutive negligible-E_phys-improvement iterations
    T   prev_ephys, ephys;
};

// FK -> target frames -> residuals -> raw weighted cost, for the config currently in st->s_x.
// FK -> target frames -> UNWEIGHTED residuals. The cost is NOT computed here: it depends on the
// row preconditioner, which is frozen for the whole iteration, so the caller applies it.
template<typename T>
__device__ __forceinline__
void lm_refresh(LMScratch<T>* st, T* out_cost) {
    T dummy;
    grid::ee_pose_inner_warp<T>(st->s_jointX, st->s_XmatsHom, st->s_x, N - 1);
    __syncwarp(FULL_WARP_MASK);
    compose_target_frames_warp<T>(st->s_target_X, st->s_jointX);
    eval_targets_warp<T>(st->s_target_X, st->s_tgt_p, st->s_tgt_q, st->s_wp, st->s_wo,
                         st->active, st->s_e_pos, st->s_e_ori, st->s_pn, st->s_on, &dummy);
    weighted_cost_warp<T>(st->s_e_pos, st->s_e_ori, st->s_wp, st->s_wo, st->s_scale,
                          st->active, out_cost);
}

// exp of a rotation vector, as a WXYZ quaternion: q = (cos(th/2), sin(th/2) * w/th).
// The quaternion form of the SO(3) exp, so no rotation matrix is ever built.
template<typename T>
__device__ __forceinline__
void quat_from_rotvec(const T* __restrict__ w, T* __restrict__ q) {
    const T th2 = w[0]*w[0] + w[1]*w[1] + w[2]*w[2];
    const T th = sqrt(th2);
    if (th < (T)1e-12) {          // first order: exp(w) ~ (1, w/2), then normalized below
        q[0] = (T)1; q[1] = (T)0.5*w[0]; q[2] = (T)0.5*w[1]; q[3] = (T)0.5*w[2];
    } else {
        const T s = sin((T)0.5*th) / th;
        q[0] = cos((T)0.5*th); q[1] = w[0]*s; q[2] = w[1]*s; q[3] = w[2]*s;
    }
    const T n = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    #pragma unroll
    for (int i = 0; i < 4; ++i) q[i] /= n;
}

// Refresh this candidate's base-frame target copy from the immutable WORLD targets, then re-score.
// NOTE: no FK. The joints did not move, and the FK is expressed in the base frame, so s_jointX and
// s_target_X are still exactly valid -- only the TARGETS moved. That is what makes a base step
// cheap next to a joint step.
template<typename T>
__device__ __forceinline__
void base_retarget_and_eval_warp(LMScratch<T>* st, const T* __restrict__ tgt_p,
                                 const T* __restrict__ tgt_q, size_t pid, int K,
                                 T eps_pos, T eps_ori) {
    const int lane = threadIdx.x & 31;
    if (lane < K) {
        world_target_to_base<T>(&tgt_p[(pid*K + lane)*3], &tgt_q[(pid*K + lane)*4],
                                st->s_base_p, st->s_base_q,
                                &st->s_tgt_p[3*lane], &st->s_tgt_q[4*lane]);
    }
    __syncwarp(FULL_WARP_MASK);
    eval_targets_full_warp<T>(st->s_target_X, st->s_tgt_p, st->s_tgt_q, st->s_wp, st->s_wo,
                              /*s_scale=*/nullptr, st->active, st->s_e_pos, st->s_e_ori,
                              st->s_pn, st->s_on, st->s_ck_base, &st->s_total_base);
    __syncwarp(FULL_WARP_MASK);
}

// ONE alternating base step: damped Gauss-Newton on the 6 base DOF, joints held.
//
//   x_k     = R_b fk_k(q) + p_b                      world contact
//   r_k     = x*_k - x_k = R_b e_base,k              (e_base is what the sweep already computed)
//   J_b,k   = [ I3 , -[x_k - p_b]x ] = [ I3, -[R_b fk_k]x ]      3x6, world frame
//   H dxi   = b ,  H_lambda = H + lambda*D ,  D = diag(s_p^-2 I3, s_R^-2 I3) ,  b = J^T W r
//   p_b+    = p_b + a*dp ,  q_b+ = exp(a*dphi) (x) q_b            world-frame LEFT perturbation
//
// POSITION-DRIVEN by design (M2): orientation residuals do not enter. Documented, not dropped.
//
// Damping is dimensionally scaled Tikhonov, and this kernel and hjcdik/base_update.py implement
// the SAME system -- that is what lets the host reference stand as an oracle for this solve.
// H mixes units (translation columns dimensionless, rotation columns carrying metres), so the
// scales are what make lambda meaningful: the penalty is lambda*(||dp||^2/s_p^2 +
// ||dphi||^2/s_R^2), each block measured against its own characteristic size and the sum
// dimensionless. s_p = s_R = 1 gives D = I (plain Tikhonov) and is the default.
//
// An earlier version damped by lambda*diag(H) here while the reference used lambda*I. Both are
// defensible; they are not the same algorithm, and the divergence silently cost the oracle its
// only job. If you change the shift, change it in both places or say so loudly.
template<typename T>
__device__ __forceinline__
void base_update_warp(LMScratch<T>* st, const T* __restrict__ tgt_p, const T* __restrict__ tgt_q,
                      size_t pid, int K, const BaseUpdateCfg<T> cfg, T eps_pos, T eps_ori) {
    constexpr int D = hjcd::FLOATING_BASE_DOF;
    const int lane = threadIdx.x & 31;
    const T ephys_before = st->ephys;

    if (lane == 0) {
        ++st->base_att;
        #pragma unroll
        for (int i = 0; i < 3; ++i) st->bak_base_p[i] = st->s_base_p[i];
        #pragma unroll
        for (int i = 0; i < 4; ++i) st->bak_base_q[i] = st->s_base_q[i];

        T Rb[9];  quat_to_mat3<T>(st->s_base_q, Rb);          // column-major 3x3
        #pragma unroll
        for (int i = 0; i < D*D; ++i) st->s_Hb[i] = (T)0;
        #pragma unroll
        for (int i = 0; i < D; ++i) st->s_bb[i] = (T)0;

        for (int k = 0; k < hjcd::NT; ++k) {
            if (!((st->active >> k) & 1u)) continue;           // inactive: same mask as the cost
            const T w = st->s_wp[k];
            if (!(w > (T)0)) continue;                         // zero weight == don't care
            // fk_k in base frame -> lever arm in world:  x_k - p_b = R_b fk_k
            const T* fk = &st->s_target_X[16*k + 12];          // col-major 4x4: translation
            T arm[3], r[3];
            #pragma unroll
            for (int i = 0; i < 3; ++i)
                arm[i] = Rb[i]*fk[0] + Rb[3+i]*fk[1] + Rb[6+i]*fk[2];
            #pragma unroll
            for (int i = 0; i < 3; ++i)                        // r_world = R_b e_base
                r[i] = Rb[i]*st->s_e_pos[3*k+0] + Rb[3+i]*st->s_e_pos[3*k+1]
                     + Rb[6+i]*st->s_e_pos[3*k+2];
            // J = [I, -[arm]x]; build rows explicitly (3x6) and accumulate H += w J^T J, b += w J^T r
            T J[3*D];
            #pragma unroll
            for (int i = 0; i < 3*D; ++i) J[i] = (T)0;
            J[0*D+0] = J[1*D+1] = J[2*D+2] = (T)1;             // dI
            J[0*D+4] =  arm[2]; J[0*D+5] = -arm[1];            // -[arm]x, row 0
            J[1*D+3] = -arm[2]; J[1*D+5] =  arm[0];            // row 1
            J[2*D+3] =  arm[1]; J[2*D+4] = -arm[0];            // row 2
            #pragma unroll
            for (int a = 0; a < D; ++a) {
                T ba = (T)0;
                #pragma unroll
                for (int i = 0; i < 3; ++i) ba += J[i*D+a] * r[i];
                st->s_bb[a] += w * ba;
                #pragma unroll
                for (int c = 0; c < D; ++c) {
                    T h = (T)0;
                    #pragma unroll
                    for (int i = 0; i < 3; ++i) h += J[i*D+a] * J[i*D+c];
                    st->s_Hb[a*D + c] += w * h;                // col-major; H symmetric
                }
            }
        }
        // H_lambda = H + lambda*D, D = diag(s_p^-2 I3, s_R^-2 I3). Added HERE rather than by
        // posv because posv can only shift by rho*I or rho*diag(A), and D is neither.
        //
        // This also retires the zero-diagonal pin that lambda*diag(H) needed. That pin existed
        // because a relative shift adds NOTHING to a zero diagonal, so the Cholesky tripped and
        // the whole step collapsed (measured: with every contact at the base origin the entire
        // rotation block of H vanishes). A fixed positive D cannot have that failure mode: H is
        // PSD and lambda*D is PD, so H_lambda is PD for any lambda > 0 and the factorization is
        // safe by construction. The degenerate case now answers itself -- a DOF that moves no
        // active target has b_i = 0, so its step is lambda*D_ii scaled into exactly zero, which
        // is what the pin was hand-forcing.
        const T inv_sp2 = (T)1 / (cfg.damping_scale_p * cfg.damping_scale_p);
        const T inv_sR2 = (T)1 / (cfg.damping_scale_R * cfg.damping_scale_R);
        #pragma unroll
        for (int i = 0; i < D; ++i)
            st->s_Hb[i*D + i] += cfg.damping * (i < 3 ? inv_sp2 : inv_sR2);
        st->s_base_fail = 0;
    }
    __syncwarp(FULL_WARP_MASK);

    // Same trusted utility the joint LM uses, at D=6. REGULARIZE=false: the shift is already in
    // H_lambda above, and letting posv add a second one would solve a system the host oracle does
    // not. CHECK stays on -- lambda == 0 is permitted, and then H_lambda can legitimately be
    // singular (K < 3 leaves the rotation block rank-deficient; see "Rank structure of J_b").
    glass::warp::posv<T, D, /*NRHS=*/1, /*REGULARIZE=*/false, /*CHECK=*/true, /*REG_DIAG=*/false>(
        st->s_Hb, st->s_bb, (T)0, &st->s_base_fail);
    __syncwarp(FULL_WARP_MASK);

    if (lane == 0) {
        if (st->s_base_fail) {
            ++st->base_numfail;                 // non-PD even after the shift: skip, never NaN
        } else {
            T dp[3] = { st->s_bb[0], st->s_bb[1], st->s_bb[2] };
            T dr[3] = { st->s_bb[3], st->s_bb[4], st->s_bb[5] };
            if (!(isfinite(dp[0]) && isfinite(dp[1]) && isfinite(dp[2]) &&
                  isfinite(dr[0]) && isfinite(dr[1]) && isfinite(dr[2]))) {
                ++st->base_numfail;
                st->s_base_fail = 1;
            } else {
                // Clip translation and rotation INDEPENDENTLY, each by its own norm: they carry
                // different units, so one joint norm would depend on an arbitrary length scale.
                // Scale the block (preserves direction); clipping components would rotate the step.
                const T nt = sqrt(dp[0]*dp[0] + dp[1]*dp[1] + dp[2]*dp[2]);
                if (cfg.max_translation > (T)0 && nt > cfg.max_translation) {
                    const T s = cfg.max_translation / nt;
                    #pragma unroll
                    for (int i = 0; i < 3; ++i) dp[i] *= s;
                }
                const T nr = sqrt(dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]);
                if (cfg.max_rotation > (T)0 && nr > cfg.max_rotation) {
                    const T s = cfg.max_rotation / nr;
                    #pragma unroll
                    for (int i = 0; i < 3; ++i) dr[i] *= s;
                }
                const T a = cfg.step_scale;
                #pragma unroll
                for (int i = 0; i < 3; ++i) {
                    T v = st->s_base_p[i] + a * dp[i];
                    v = fmin(fmax(v, cfg.lo[i]), cfg.hi[i]);     // bounds
                    st->s_base_p[i] = v;
                }
                T w[3] = { a*dr[0], a*dr[1], a*dr[2] }, dq[4], qn[4];
                quat_from_rotvec<T>(w, dq);
                quat_mul_wxyz<T>(dq, st->s_base_q, qn);          // LEFT: world-frame perturbation
                const T n = sqrt(qn[0]*qn[0] + qn[1]*qn[1] + qn[2]*qn[2] + qn[3]*qn[3]);
                #pragma unroll
                for (int i = 0; i < 4; ++i) st->s_base_q[i] = qn[i] / n;   // stays on the manifold
            }
        }
    }
    __syncwarp(FULL_WARP_MASK);
    if (st->s_base_fail) return;                                  // base untouched

    base_retarget_and_eval_warp<T>(st, tgt_p, tgt_q, pid, K, eps_pos, eps_ori);

    if (lane == 0) {
        // Accept on the PHYSICAL merit, never on cost_lm: the row scales are re-frozen every
        // iteration, so consecutive scaled costs are in different units and are not comparable
        // (hjcd_kernel.cu:1457-1461). E_phys is the metric best_x already tracks.
        const T after = e_phys<T>(st->s_pn, st->s_on, st->active, eps_pos, eps_ori);
        st->take_best = 0;
        if (after < ephys_before) {
            st->ephys = after;
            ++st->base_acc;
        } else {
            #pragma unroll
            for (int i = 0; i < 3; ++i) st->s_base_p[i] = st->bak_base_p[i];
            #pragma unroll
            for (int i = 0; i < 4; ++i) st->s_base_q[i] = st->bak_base_q[i];
            st->take_best = 1;                                   // marker: re-evaluate below
        }
    }
    __syncwarp(FULL_WARP_MASK);
    if (st->take_best) {
        // Exact rollback: the base-frame targets are a deterministic function of (world targets,
        // base), so recomputing them from the restored base reproduces the pre-step state bit for
        // bit -- no saved copy of s_tgt_*/s_e_* needed.
        base_retarget_and_eval_warp<T>(st, tgt_p, tgt_q, pid, K, eps_pos, eps_ori);
    }
    __syncwarp(FULL_WARP_MASK);
    if (lane == 0) st->take_best = 0;
}

template<typename T>
__global__ void lm_multi_target_kernel(
    T* __restrict__ x,                        // B x N, in-place (seed -> refined)
    const T* __restrict__ tgt_p,              // B x NT x 3   (WORLD frame)
    const T* __restrict__ tgt_q,              // B x NT x 4 (wxyz, unit), WORLD frame
    const unsigned int* __restrict__ active,  // B
    const T* __restrict__ w_pos,              // B x NT
    const T* __restrict__ w_ori,              // B x NT
    T* __restrict__ base_p,                   // B x 3, candidate-level, IN/OUT; NULL => fixed base
    T* __restrict__ base_q,                   // B x 4 (wxyz, unit), IN/OUT; NULL => fixed base
    int* __restrict__ out_base_diag,          // B x 3 (attempted, accepted, numfail), may be null
    T* __restrict__ out_pn,                   // B x NT
    T* __restrict__ out_on,                   // B x NT
    T* __restrict__ out_cost,                 // B
    unsigned char* __restrict__ out_succ,     // B
    T* __restrict__ out_pose,                 // B x NT x 7 (x,y,z, qw,qx,qy,qz), may be null
    T* __restrict__ out_trace,                // B x trace_cap x TRACE_COLS, may be null
    const int trace_cap,
    const grid::robotModel<T>* __restrict__ RM,
    const T eps_pos, const T eps_ori, const T lambda_init, const int k_max, const int B,
    const int stop_on_first,
    const int stag_patience,                  // Policy B: 0 DISABLES stagnation stopping (default)
    const T stag_rel,                         //           relative E_phys improvement threshold
    const int seeds_per_problem,              // S: candidates that share one problem's targets/mask
    const BaseUpdateCfg<T> bcfg)              // alternating base step; .enabled=0 => never taken
{
    constexpr int K = hjcd::NT;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int wpb = max(1, (int)(blockDim.x >> 5));
    const int gp = blockIdx.x * wpb + warp;
    // pid selects PROBLEM-LEVEL data (targets, mask, weights); gp stays for CANDIDATE-LEVEL data
    // (seed, all outputs). With seeds_per_problem == 1 this is pid == gp -- byte-identical to the
    // pre-multi-problem path, where every candidate carried its own target copy.
    const int pid = gp / seeds_per_problem;

    extern __shared__ __align__(16) unsigned char s_lm_raw[];
    LMScratch<T>* all = reinterpret_cast<LMScratch<T>*>(s_lm_raw);
    LMScratch<T>* st = &all[warp];

    // Block-cooperative one-time load of the q-INDEPENDENT XmatsHom cells (identical for every
    // candidate); the per-warp FK refreshes the q-dependent cells. Only barriers in the kernel.
    {
        __shared__ T s_xhom_tmpl[grid::XHOM_T_COUNT];
        __shared__ T s_q_tmpl[N];
        __shared__ T s_tmp_tmpl[NX * 2];
        __shared__ int s_topo_tmpl[hjcd::TOPO];
        const int base0 = min(blockIdx.x * wpb, B - 1);
        if (threadIdx.x < N) s_q_tmpl[threadIdx.x] = x[(size_t)base0 * N + threadIdx.x];
        __syncthreads();
        grid::load_update_XmatsHom_helpers<T>(s_xhom_tmpl, s_topo_tmpl, s_q_tmpl, RM, s_tmp_tmpl);
        __syncthreads();
        for (int i = threadIdx.x; i < wpb * grid::XHOM_T_COUNT; i += blockDim.x) {
            const int w = i / grid::XHOM_T_COUNT;
            all[w].s_XmatsHom[i - w * grid::XHOM_T_COUNT] = s_xhom_tmpl[i - w * grid::XHOM_T_COUNT];
        }
        __syncthreads();
    }
    if (gp >= B) return;

    const T lambda_min = (T)1e-12, lambda_max = (T)1e6;
    const int stall_lim = 5;
    T lambda = lambda_init;

    // Diagnostic counters live in lane-0 REGISTERS and are published to global memory from INSIDE
    // the loop. Two things were verified the hard way against the trace: counters held in the shared
    // LMScratch read back as a stale value, and an int output buffer did not carry them correctly.
    // Registers + a floating-point output buffer + in-loop publication all agree with the trace.
    int c_trials = 0, c_lsearch = 0, c_accept = 0;

    if (lane < N) st->s_x[lane] = x[(size_t)gp * N + lane];      // seed: candidate-level
    // Base pose: CANDIDATE-level (gp), unlike the targets, which are problem-level (pid). Every
    // seed of a problem may sit at a different base -- that is the whole point of the feature.
    if (lane == 0) {
        if (base_p != nullptr) {
            #pragma unroll
            for (int c = 0; c < 3; ++c) st->s_base_p[c] = base_p[(size_t)gp * 3 + c];
            #pragma unroll
            for (int c = 0; c < 4; ++c) st->s_base_q[c] = base_q[(size_t)gp * 4 + c];
        } else {                                                 // fixed base: identity
            st->s_base_p[0] = st->s_base_p[1] = st->s_base_p[2] = (T)0;
            st->s_base_q[0] = (T)1;
            st->s_base_q[1] = st->s_base_q[2] = st->s_base_q[3] = (T)0;
        }
    }
    __syncwarp(FULL_WARP_MASK);        // lanes 1..K-1 read s_base_* just below
    if (lane < K) {
        st->s_wp[lane] = w_pos[(size_t)pid * K + lane];          // weights: problem-level
        st->s_wo[lane] = w_ori[(size_t)pid * K + lane];
        // Store this candidate's targets in ITS OWN base frame (hjcd_settings.h, eq. 2). A null
        // base_p takes the verbatim-copy branch, so fixed base stays bit-identical to before.
        world_target_to_base<T>(&tgt_p[((size_t)pid*K + lane)*3], &tgt_q[((size_t)pid*K + lane)*4],
                                base_p == nullptr ? nullptr : st->s_base_p, st->s_base_q,
                                &st->s_tgt_p[3*lane], &st->s_tgt_q[4*lane]);
    }
    if (lane == 0) {
        st->active = active[pid];                                // mask: problem-level
        st->s_break = 0; st->stall = 0; st->prev_cost = (T)-1;
        st->stag_n = 0;  st->prev_ephys = (T)-1;
        st->base_att = 0; st->base_acc = 0; st->base_numfail = 0; st->s_base_fail = 0;
    }
    __syncwarp(FULL_WARP_MASK);

    if (lane < hjcd::NT * 6) st->s_scale[lane] = (T)1;      // provisional; refined below
    __syncwarp(FULL_WARP_MASK);
    lm_refresh<T>(st, &st->cost);
    compute_row_scales_warp<T>(st->s_scale, st->s_jointX, st->s_target_X, st->active);
    weighted_cost_warp<T>(st->s_e_pos, st->s_e_ori, st->s_wp, st->s_wo, st->s_scale,
                          st->active, &st->cost);
    if (lane == 0) {
        worst_active_errors<T>(st->s_pn, st->s_on, st->s_wp, st->s_wo, st->active,
                               &st->max_pn, &st->max_on);
        st->best_cost = st->cost;
        // best_x is tracked on E_phys, NOT on the row-scaled cost. The row scales s_{k,r} =
        // 1/||J_{k,r}|| are re-frozen every iteration, so C^(t) and C^(t-1) are expressed in
        // DIFFERENT units -- comparing them across iterations is not a real comparison, and it made
        // 41.8% of seeds return a configuration physically WORSE than one they had already visited
        // (median 1.05x, worst 10.8x). E_phys is built from physical residual norms and fixed
        // tolerances, so it is comparable across iterations. Within an iteration nothing changes:
        // the trial acceptance, damping, line search and trust region all still use the row-scaled
        // cost exactly as before.
        st->best_ephys = e_phys<T>(st->s_pn, st->s_on, st->active, eps_pos, eps_ori);
        if (all_active_converged<T>(st->s_pn, st->s_on, st->s_wp, st->s_wo, st->active,
                                    eps_pos, eps_ori)) st->s_break = 1;
    }
    if (lane < N) st->best_x[lane] = st->s_x[lane];
    // The base is seeded WITH best_x, for the same reason the epilogue restores the two together:
    // best_base_* is otherwise written only when take_best/improved fires, and neither is
    // guaranteed to happen even once -- the block above sets s_break when the seed already
    // converged, so the loop is skipped entirely. The epilogue restores s_base_* from
    // best_base_* unconditionally, so leaving it unwritten hands a candidate that arrived
    // already solved a base of uninitialized shared memory.
    if (base_p != nullptr && lane == 0) {
        #pragma unroll
        for (int i = 0; i < 3; ++i) st->best_base_p[i] = st->s_base_p[i];
        #pragma unroll
        for (int i = 0; i < 4; ++i) st->best_base_q[i] = st->s_base_q[i];
    }
    __syncwarp(FULL_WARP_MASK);

    for (int it = 0; it < k_max && !st->s_break; ++it) {
        if (lane == 0) {
            // Cooperative early-exit: once ANY candidate in the launch has converged, the rest stop.
            // This is what makes the latency path (num_solutions == 1) fast -- without it every
            // candidate grinds all k_max iterations. Same g_stop flag the coarse search uses.
            if (stop_on_first && ((it & 1) == 0) && atomicAdd(&g_stop, 0)) st->s_break = 1;
        }
        __syncwarp(FULL_WARP_MASK);
        if (st->s_break) break;      // bailed before any work: NOT counted as a linearization


        // Re-derive the row preconditioner from the CURRENT Jacobian, then re-express the current
        // cost under it. Frozen for the rest of this iteration so the trial comparison is like-for-like.
        compute_row_scales_warp<T>(st->s_scale, st->s_jointX, st->s_target_X, st->active);
        weighted_cost_warp<T>(st->s_e_pos, st->s_e_ori, st->s_wp, st->s_wo, st->s_scale,
                              st->active, &st->cost);

        accumulate_normal_equations_warp<T>(st->A, st->b, st->s_jointX, st->s_target_X,
                                            st->s_e_pos, st->s_e_ori, st->s_wp, st->s_wo,
                                            st->s_scale, st->active);
        // Pin joints that cannot move ANY active target.
        //
        // Such a joint has an all-zero row AND column in A (by the ancestor mask) and a zero b --
        // including a ZERO DIAGONAL. Marquardt damping is lambda*diag(A), so it adds nothing there:
        // A + lambda*diag(A) stays singular, the Cholesky CHECK trips, dq collapses to 0 and the
        // whole solve freezes. (This is exactly why "both hands" and "both feet" failed on G1 while
        // "all four" passed -- with every limb active, every joint has a nonzero diagonal.)
        // A unit diagonal makes A positive-definite and gives such a joint a step of exactly zero,
        // which is the correct semantics: it is frozen, not merely undamped.
        {
            unsigned int act_joints = 0u;
            #pragma unroll
            for (int k = 0; k < K; ++k)
                if ((st->active >> k) & 1u) act_joints |= hjcd_gen::TARGET_ANCESTOR_MASK[k];
            if (lane < N && !((act_joints >> lane) & 1u)) {
                st->A[lane * N + lane] = (T)1;
                st->b[lane] = (T)0;
            }
        }
        __syncwarp(FULL_WARP_MASK);

        // posv overwrites A with its Cholesky factor and b with the solution, so keep diag(A) and
        // the gradient g = J^T W e first -- the gain ratio below needs the UNDAMPED gradient.
        if (lane < N) { st->diagA[lane] = st->A[lane * N + lane]; st->g[lane] = st->b[lane]; }
        __syncwarp(FULL_WARP_MASK);

        // (A + lambda*diag(A)) dq = b.   b is J^T W e with e = target - current, so dq is a
        // DESCENT direction with a PLUS sign: x <- x + a*dq. No sign flip anywhere.
        glass::warp::posv<T, N, /*NRHS=*/1, /*REGULARIZE=*/true, /*CHECK=*/true, /*REG_DIAG=*/true>(
            st->A, st->b, lambda, &st->s_fail);
        __syncwarp(FULL_WARP_MASK);
        ++c_trials;
        if (lane < N) st->dq[lane] = (st->s_fail == 0) ? st->b[lane] : (T)0;
        __syncwarp(FULL_WARP_MASK);

        if (lane == 0) {
            // Trust region + per-joint clip, keyed on the WORST active errors (not the mean).
            T R;
            if      (st->max_pn > (T)1e-2 || st->max_on > (T)0.6)  R = (T)0.38;
            else if (st->max_pn > (T)1e-3 || st->max_on > (T)0.25) R = (T)0.22;
            else if (st->max_pn > (T)2e-4 || st->max_on > (T)0.08) R = (T)0.12;
            else                                                   R = (T)0.05;
            T nrm = (T)0;
            for (int i = 0; i < N; ++i) nrm += st->dq[i] * st->dq[i];
            nrm = sqrt(nrm);
            if (nrm > R) { const T s = R / (nrm + (T)1e-18); for (int i = 0; i < N; ++i) st->dq[i] *= s; }
            const T clip = (st->max_pn > (T)1e-2) ? (T)0.30 :
                           (st->max_pn > (T)1e-3) ? (T)0.15 :
                           (st->max_pn > (T)2e-4) ? (T)0.08 : (T)0.03;
            for (int i = 0; i < N; ++i) {
                st->dq[i] = fmin(fmax(st->dq[i], -clip), clip);
                st->x_old[i] = st->s_x[i];
            }
            st->accepted = 0;
        }
        __syncwarp(FULL_WARP_MASK);

        // Backtracking on the RAW WEIGHTED cost. Same schedule as the single-target LM.
        T best_a = (T)0;
        for (int tries = 0; tries < 4; ++tries) {
            // a = 2^-tries: 1, 1/2, 1/4, 1/8. This WAS pow((T)0.5, tries - 1), which for T = float
            // is pow(float, int) -- and the standard promotes an integral exponent to double, so the
            // fp32 line search was calling a full double-precision pow on every backtracking trial,
            // in the hot loop, on a GPU that runs FP64 at 1/64 rate. Every value here is exactly
            // representable, so the shift is BITWISE identical in fp64 and the schedule is unchanged.
            const T a = (T)1 / (T)(1 << tries);
            if (lane < N) {
                T Llo, Lhi; joint_limit<T>(lane, &Llo, &Lhi);
                const T xi = st->x_old[lane] + a * st->dq[lane];
                st->s_x[lane] = fmin(fmax(xi, Llo), Lhi);
            }
            __syncwarp(FULL_WARP_MASK);
            lm_refresh<T>(st, &st->trial_cost);
            ++c_lsearch;
            if (lane == 0 && st->trial_cost + (T)1e-20 < st->cost) { st->accepted = 1; }
            __syncwarp(FULL_WARP_MASK);
            if (st->accepted) { best_a = a; break; }
        }
        __syncwarp(FULL_WARP_MASK);

        if (st->accepted) ++c_accept;
        if (lane == 0) {
            if (st->accepted) {
                const T ared = st->cost - st->trial_cost;
                // Predicted reduction, derived for THIS cost. The cost is C = sum_k w_k |e_k|^2 with
                // NO 1/2 factor (Phase-2 spec). Linearizing e(q+dq) ~ e - J dq:
                //     C(q+dq) ~ C - 2 dq^T b + dq^T A dq,   and the LM step gives A dq = b - lambda D dq
                //  => pred = 2 dq^T b - dq^T b + lambda dq^T D dq
                //          = sum_i ( dq_i * g_i + lambda * diagA_i * dq_i^2 )
                // The old single-target LM carried a 1/2 here because ITS cost was 0.5*||r||^2.
                // Keeping that 1/2 with the unhalved cost made rho exactly 2x too large, so rho > 0.9
                // fired constantly and lambda was driven to ~1e-11 -- i.e. the Levenberg damping was
                // switched off. Panda's A is rank-6 in a 7-DoF space (6 task rows), so an undamped
                // solve puts a large component along the null direction, which the trust region then
                // clips away: the step is wasted and convergence crawls. THAT is the 21x regression.
                T pred = (T)0;
                for (int i = 0; i < N; ++i) {
                    const T ad = best_a * st->dq[i];
                    pred += lambda * st->diagA[i] * ad * ad + ad * st->g[i];
                }
                pred = fmax((T)1e-20, pred);
                const T rho = ared / pred;
                if      (rho > (T)0.90) lambda = fmax(lambda * (T)0.3, lambda_min);
                else if (rho > (T)0.50) lambda = fmax(lambda * (T)0.5, lambda_min);
                else if (rho < (T)0.25) lambda = fmin(lambda * (T)3.0, lambda_max);
                st->cost = st->trial_cost;
                if (st->prev_cost > (T)0 &&
                    (st->prev_cost - st->cost) / st->prev_cost < (T)1e-9) ++st->stall;
                else st->stall = 0;
                st->prev_cost = st->cost;
            } else {
                lambda = fmin(lambda * (T)3.0, lambda_max);
                ++st->stall;
            }
        }
        __syncwarp(FULL_WARP_MASK);

        if (!st->accepted) {                       // restore: every trial overwrote s_x
            if (lane < N) st->s_x[lane] = st->x_old[lane];
            __syncwarp(FULL_WARP_MASK);
            lm_refresh<T>(st, &st->trial_cost);
        }

        if (lane == 0) {
            worst_active_errors<T>(st->s_pn, st->s_on, st->s_wp, st->s_wo, st->active,
                                   &st->max_pn, &st->max_on);
            if (st->cost < st->best_cost) st->best_cost = st->cost;   // reporting only

            // Policy B: stagnation on the STABLE physical metric E_phys. Computed HERE, from the
            // same s_pn/s_on that max_pn/max_on were just taken from, so trace columns 6, 7 and 10
            // all describe ONE state. (Computing it at the end of the iteration instead would read
            // the post-stall-kick residuals, and the trace row would silently mix two moments.)
            // patience == 0 disables the stop entirely -- the default.
            const T ep = e_phys<T>(st->s_pn, st->s_on, st->active, eps_pos, eps_ori);
            st->ephys = ep;
            // STRICTLY better physical merit -> this state becomes best_x. A tie does not displace
            // the incumbent, so the earliest state of equal merit wins (deterministic).
            st->take_best = (ep < st->best_ephys) ? 1 : 0;
            if (st->take_best) st->best_ephys = ep;
            if (stag_patience > 0) {
                const T prev = st->prev_ephys;
                if (prev > (T)0 && (prev - ep) / prev < stag_rel) ++st->stag_n;
                else st->stag_n = 0;
                if (st->stag_n >= stag_patience) st->s_break = 1;
            }
            st->prev_ephys = ep;
        }
        __syncwarp(FULL_WARP_MASK);
        if (lane < N && st->take_best) st->best_x[lane] = st->s_x[lane];
        // The base travels WITH best_x: best_x is the best JOINTS seen, and with a moving base
        // those joints only mean anything against the base they were scored at.
        if (lane == 0 && st->take_best) {
            #pragma unroll
            for (int i = 0; i < 3; ++i) st->best_base_p[i] = st->s_base_p[i];
            #pragma unroll
            for (int i = 0; i < 4; ++i) st->best_base_q[i] = st->s_base_q[i];
        }
        __syncwarp(FULL_WARP_MASK);

        // ---- alternating base step. REFINEMENT ONLY by design: the coarse sweep's greedy
        // per-joint accept/rollback is delicate, and a base move perturbs every target at once.
        // Placed AFTER the joint step and its E_phys, so st->ephys is the incumbent it must beat.
        if (bcfg.enabled && !st->s_break && bcfg.interval > 0 && (it % bcfg.interval) == 0) {
            base_update_warp<T>(st, tgt_p, tgt_q, (size_t)pid, K, bcfg, eps_pos, eps_ori);
            // `improved` MUST be warp-uniform. Reading st->best_ephys on every lane while lane 0
            // wrote it (with no __syncwarp between) is a real race -- lanes progress independently
            // since Volta, so some lanes saw the pre-write value and some the post-write one. The
            // flag then went non-uniform and the `lane < N` best_x store below wrote a MIXTURE of
            // the incumbent and current joints, paired with whichever base lane 0 chose. Decide on
            // ONE lane and broadcast, so every lane acts on the same decision by construction.
            int improved = 0;
            if (lane == 0) improved = (st->ephys < st->best_ephys) ? 1 : 0;
            improved = __shfl_sync(FULL_WARP_MASK, improved, 0);
            if (lane == 0 && improved) st->best_ephys = st->ephys;
            if (lane == 0 && improved) {
                #pragma unroll
                for (int i = 0; i < 3; ++i) st->best_base_p[i] = st->s_base_p[i];
                #pragma unroll
                for (int i = 0; i < 4; ++i) st->best_base_q[i] = st->s_base_q[i];
            }
            if (lane < N && improved) st->best_x[lane] = st->s_x[lane];
            __syncwarp(FULL_WARP_MASK);
        }

        if (lane == 0) st->kicked = 0;
        __syncwarp(FULL_WARP_MASK);
        if (lane == 0 && st->stall >= stall_lim) {
            for (int i = 0; i < N; ++i) {
                T Llo, Lhi; joint_limit<T>(i, &Llo, &Lhi);
                const T span = Lhi - Llo;                 // in T: no FP64 subtraction on the fp32 path
                const uint32_t u = 0x9E3779B9u ^ (uint32_t)(i * 0xC2B2AE35u);
                const T kick = (T)0.005 * span * ((u & 1u) ? (T)1 : (T)-1);
                st->s_x[i] = fmin(fmax(st->s_x[i] + kick, Llo), Lhi);
            }
            st->stall = 0;
            st->kicked = 1;
        }
        __syncwarp(FULL_WARP_MASK);
        // Only a kick invalidates the cached FK/residual state -- after an accept the trial loop
        // already left it consistent with s_x, so refreshing again was a whole wasted FK per iter.
        if (st->kicked) { lm_refresh<T>(st, &st->cost); }

        // Deterministic per-iteration trace -- the AUTHORITATIVE source of every LM diagnostic.
        // Written from inside the loop straight to global memory (the compact per-problem counter
        // buffer this replaced reported a stale value; see docs/PHASE3C in the report). Emitted only
        // when diagnostics are on: with out_trace == nullptr there is not a single extra store.
        if (out_trace && lane == 0 && it < trace_cap) {
            T* row = &out_trace[((size_t)gp * trace_cap + it) * TRACE_COLS];
            row[0] = (T)1;                              // VALID -- explicit, never inferred
            row[1] = (T)it;
            row[2] = (T)c_trials;
            row[3] = (T)(st->accepted ? 1 : 0);
            row[4] = (T)c_accept;
            row[5] = st->cost;
            row[6] = st->max_pn;
            row[7] = st->max_on;
            row[8] = lambda;
            row[9] = (T)c_lsearch;
            row[10] = st->ephys;      // same state as row[6]/row[7]
        }
        __syncwarp(FULL_WARP_MASK);

        // Policy A (stop THIS seed the moment it is solved) -- already here, and per-warp, so it
        // never terminates another seed in the batch. stop_on_first is 0 for the multi-target path.
        if (lane == 0 && all_active_converged<T>(st->s_pn, st->s_on, st->s_wp, st->s_wo,
                                                 st->active, eps_pos, eps_ori)) {
            st->s_break = 1;
            if (stop_on_first) atomicCAS(&g_stop, 0, 1);
        }
        __syncwarp(FULL_WARP_MASK);
    }

    // Report the BEST config seen, not the last one.
    if (lane < N) st->s_x[lane] = st->best_x[lane];
    // ... and the base it was SCORED AGAINST. Restoring the joints alone would have lm_refresh
    // below re-score them against whatever base happened to be current, which is a different pose.
    if (base_p != nullptr) {
        if (lane == 0) {
            #pragma unroll
            for (int i = 0; i < 3; ++i) st->s_base_p[i] = st->best_base_p[i];
            #pragma unroll
            for (int i = 0; i < 4; ++i) st->s_base_q[i] = st->best_base_q[i];
        }
        __syncwarp(FULL_WARP_MASK);
        if (lane < K)
            world_target_to_base<T>(&tgt_p[((size_t)pid*K + lane)*3],
                                    &tgt_q[((size_t)pid*K + lane)*4],
                                    st->s_base_p, st->s_base_q,
                                    &st->s_tgt_p[3*lane], &st->s_tgt_q[4*lane]);
    }
    __syncwarp(FULL_WARP_MASK);
    lm_refresh<T>(st, &st->cost);

    if (lane < K) {
        out_pn[(size_t)gp * K + lane] = st->s_pn[lane];
        out_on[(size_t)gp * K + lane] = st->s_on[lane];
        if (out_pose) {
            const T* Xt = &st->s_target_X[16 * lane];
            T qq[4];
            mat_to_quat(Xt, qq);
            T* o = &out_pose[((size_t)gp * K + lane) * 7];
            o[0]=Xt[12]; o[1]=Xt[13]; o[2]=Xt[14];
            o[3]=qq[0];  o[4]=qq[1];  o[5]=qq[2]; o[6]=qq[3];
        }
    }
    if (lane < N) x[(size_t)gp * N + lane] = st->s_x[lane];
    // The optimized base is an OUTPUT too, written in place exactly as x is. Fixed-base solves
    // pass null and never reach here.
    if (lane == 0 && base_p != nullptr) {
        #pragma unroll
        for (int i = 0; i < 3; ++i) base_p[(size_t)gp * 3 + i] = st->s_base_p[i];
        #pragma unroll
        for (int i = 0; i < 4; ++i) base_q[(size_t)gp * 4 + i] = st->s_base_q[i];
        // Diagnostics, [B,3] = (attempted, accepted, numerical failures). Nullable: a caller that
        // does not ask pays one predicated store. Without these the acceptance RATE is
        // unobservable from outside -- a base update that proposes constantly and is rejected
        // every time is indistinguishable from one that never runs, and both look like "the base
        // barely moved".
        if (out_base_diag) {
            out_base_diag[(size_t)gp * 3 + 0] = st->base_att;
            out_base_diag[(size_t)gp * 3 + 1] = st->base_acc;
            out_base_diag[(size_t)gp * 3 + 2] = st->base_numfail;
        }
    }
    if (lane == 0) {
        out_cost[gp] = st->cost;
        out_succ[gp] = all_active_converged<T>(st->s_pn, st->s_on, st->s_wp, st->s_wo,
                                               st->active, eps_pos, eps_ori) ? 1 : 0;
        // Zero-iteration case (already converged on entry): no trace row was written, so the host
        // derives lm_iterations == 0 and every other counter == 0. That is the documented convention.
    }
}

// ---------------------------------------------------------------------------
// Phase 5: multi-target coarse search -- aggregate weighted coordinate Gauss-Newton.
//
// One warp per problem; lane j owns joint j (N <= 31). Per outer coarse iteration:
//
//   1. Freeze the Phase-3B row preconditioner  s_{k,r} = 1/(||J_{k,r}|| + eps)  ONCE. The same frozen
//      scaling ranks the proposals AND scores the exact trial, so they compare like with like.
//   2. Every lane forms ONE aggregate scalar proposal for its joint:
//          g_j = sum_{k in A_j} ( Jv_kj^T W_p,k e_p,k + Jw_kj^T W_R,k e_R,k )
//          h_j = sum_{k in A_j} ( Jv_kj^T W_p,k Jv_kj + Jw_kj^T W_R,k Jw_kj )
//          delta_j = g_j / (h_j + lambda_coord)
//      with W_{k,r} = w_{k,r} s_{k,r}^2. de/dq = -J, so the numerator sign is POSITIVE.
//      A_j = JOINT_TARGET_MASK[j] & active_target_mask; A_j == 0 => invalid proposal.
//   3. Clip to the max coordinate step and to the joint limits, then take the LINEARIZED predicted
//      improvement of the clipped step:  pred_j = 2 delta_j g_j - delta_j^2 h_j.
//   4. Warp-wide reduction picks the single best proposal.
//   5. ONLY the winner is evaluated exactly: apply -> Phase-4 subtree FK -> recompose only the
//      affected targets -> exact scaled cost. Accept iff the exact cost improves (and, when a
//      collision environment is bound, iff the config is exactly collision-free); otherwise the
//      validated Phase-4 rollback restores the cached state.
//
// No FK, subtree FK or collision check is ever run for a losing proposal.
// ---------------------------------------------------------------------------
static constexpr int CTRACE_COLS = 13;   // valid, it, joint, delta, pred, cost_before, cost_after,
                                         // accepted, stall, perturbed(=RETAINED kick),
                                         // pert_attempts, pert_collision_rejects, pert_exhausted
static constexpr int MAX_PERT_ATTEMPTS_DEFAULT = 4;

template<typename T>
struct CoarseScratch {
    T s_x[N], best_x[N];
    T s_XmatsHom[grid::XHOM_T_COUNT];
    T s_jointX[N * 16];
    T s_target_X[hjcd::NT * 16];
    T s_tgt_p[hjcd::NT * 3], s_tgt_q[hjcd::NT * 4];   // in the CANDIDATE'S BASE FRAME
    // Floating base, per candidate (hjcd::FLOATING_BASE_DOF). s_tgt_p/s_tgt_q above are stored
    // in THIS base's frame -- the only way the base enters the solver. The FK, the coordinate
    // machinery and the cost never see it. Identity for a fixed-base solve, which then reduces
    // to exactly the previous bytes.
    T s_base_p[3], s_base_q[4];
    T s_wp[hjcd::NT], s_wo[hjcd::NT];
    T s_e_pos[hjcd::NT * 3], s_e_ori[hjcd::NT * 3];
    T s_pn[hjcd::NT], s_on[hjcd::NT], s_ck[hjcd::NT];
    T s_scale[hjcd::NT * 6];
    T s_loc16[16];
    // Phase-4 rollback shadow (single-joint proposal)
    T v_e_pos[hjcd::NT * 3], v_e_ori[hjcd::NT * 3];
    T v_pn[hjcd::NT], v_on[hjcd::NT], v_ck[hjcd::NT];
    T v_total, v_theta;
    // Perturbation shadow. A kick rewrites EVERY joint, so unlike a coordinate proposal there is no
    // subtree to exploit: we save the whole configuration and rebuild from it. coarse_full_refresh
    // is a pure function of s_x (plus the targets/weights/active mask, none of which a kick
    // touches), so restoring s_x bitwise and re-running it reproduces the transforms, residuals and
    // costs exactly. p_total is kept anyway so the scalar cost is restored from the saved value
    // rather than recomputed.
    T p_x[N];
    T p_total;
    T total, best_total, trial_total;
    T best_ephys;          // cross-iteration best-state metric (STABLE, physical)
    int take_best;
    T max_pn, max_on;
    unsigned int active;
    int stall, accepted, win_j;
    T win_v, win_pred;
};

// One lane's aggregate coordinate Gauss-Newton proposal. Returns the clipped step in *out_delta**,
// the absolute new joint value in *out_v*, the linearized predicted improvement in *out_pred*.
// Invalid (no affected target, or curvature below the floor) => *out_pred = -1.
template<typename T>
__device__ __forceinline__
void coord_proposal(const CoarseScratch<T>* st, int j, T lambda_coord, T h_min, T max_step,
                    T* out_v, T* out_delta, T* out_pred)
{
    *out_pred = (T)-1;
    *out_delta = (T)0;
    *out_v = st->s_x[j];

    const unsigned int affected = hjcd_gen::JOINT_TARGET_MASK[j] & st->active;
    if (affected == 0u) return;                      // this joint cannot move any active target

    const T* __restrict__ Ci = &st->s_jointX[16 * j];
    T ax[3];
    joint_world_axis<T>(Ci, j, ax);
    const bool prism = grid::HAS_PRISMATIC && grid::JOINT_IS_PRISMATIC[j];

    T g = (T)0, h = (T)0;
    #pragma unroll
    for (int k = 0; k < hjcd::NT; ++k) {
        if (!((affected >> k) & 1u)) continue;
        const T* __restrict__ Xt = &st->s_target_X[16 * k];
        const T rx = Xt[12] - Ci[12], ry = Xt[13] - Ci[13], rz = Xt[14] - Ci[14];
        T Jv[3], Jw[3];
        Jv[0] = prism ? ax[0] : (ax[1]*rz - ax[2]*ry);
        Jv[1] = prism ? ax[1] : (ax[2]*rx - ax[0]*rz);
        Jv[2] = prism ? ax[2] : (ax[0]*ry - ax[1]*rx);
        Jw[0] = prism ? (T)0 : ax[0];
        Jw[1] = prism ? (T)0 : ax[1];
        Jw[2] = prism ? (T)0 : ax[2];
        const T wp = st->s_wp[k], wo = st->s_wo[k];
        #pragma unroll
        for (int r = 0; r < 3; ++r) {
            const T sp = st->s_scale[6*k + r], so = st->s_scale[6*k + 3 + r];
            const T Wp = wp * sp * sp, Wo = wo * so * so;
            g += Wp * Jv[r] * st->s_e_pos[3*k + r] + Wo * Jw[r] * st->s_e_ori[3*k + r];
            h += Wp * Jv[r] * Jv[r]                + Wo * Jw[r] * Jw[r];
        }
    }
    if (!(h > h_min)) return;                        // curvature too flat: invalidate, do not damp it away

    T delta = g / (h + lambda_coord);
    delta = fmin(fmax(delta, -max_step), max_step);                 // max coordinate step
    T Llo, Lhi; joint_limit<T>(j, &Llo, &Lhi);
    const T v = fmin(fmax(st->s_x[j] + delta, Llo), Lhi);           // joint limits
    delta = v - st->s_x[j];                                         // effective step after clipping
    if (delta == (T)0) return;

    const T pred = (T)2 * delta * g - delta * delta * h;            // linearized improvement
    if (!(pred > (T)0)) return;
    *out_v = v;
    *out_delta = delta;
    *out_pred = pred;
}

template<typename T>
__device__ __forceinline__
void coarse_full_refresh(CoarseScratch<T>* st) {
    grid::ee_pose_inner_warp<T>(st->s_jointX, st->s_XmatsHom, st->s_x, N - 1);
    __syncwarp(FULL_WARP_MASK);
    compose_target_frames_warp<T>(st->s_target_X, st->s_jointX);
    compute_row_scales_warp<T>(st->s_scale, st->s_jointX, st->s_target_X, st->active);
    eval_targets_full_warp<T>(st->s_target_X, st->s_tgt_p, st->s_tgt_q, st->s_wp, st->s_wo,
                              st->s_scale, st->active, st->s_e_pos, st->s_e_ori,
                              st->s_pn, st->s_on, st->s_ck, &st->total);
}

// HARD is a COMPILE-TIME flag, not a runtime one, and that is load-bearing. Pulling the sidecar
// geometry into this TU costs +153 registers and a 2096-byte stack frame -- ptxas inlines the
// __noinline__ entry points regardless -- and paying that in `off`/`final` would silently halve
// their occupancy. Templating means the <T,false> instantiation never references the sidecar at
// all, so dead-code elimination gives back the exact pre-hard-mode kernel, and only hard mode pays
// hard mode's cost. Verified in the ptxas report, not assumed.
// ORACLE is likewise compile-time. Folding the debug oracle's full 351-pair sweep into the hot
// instantiation cost 29 more registers -- pushing it to the 255 ceiling -- and introduced the first
// register spills in this kernel. The spec requires the oracle to be OFF in performance runs; a
// third template parameter is what makes "off" mean "not compiled in" rather than "branch not
// taken". <T,true,false> is the fast hard path; <T,true,true> is validation only.
template<typename T, bool HARD, bool ORACLE>
__global__ void coarse_search_mt_kernel(
    T* __restrict__ x,                        // B x N   seeds in, refined out (best seen)
    const T* __restrict__ tgt_p,              // B x NT x 3   (WORLD frame)
    const T* __restrict__ tgt_q,              // B x NT x 4   (WORLD frame)
    const unsigned int* __restrict__ active,  // B
    const T* __restrict__ w_pos, const T* __restrict__ w_ori,
    const T* __restrict__ base_p,             // B x 3, candidate-level; NULL => fixed base
    const T* __restrict__ base_q,             // B x 4 (wxyz, unit); NULL => fixed base
    T* __restrict__ out_pn, T* __restrict__ out_on,   // B x NT
    T* __restrict__ out_cost,                 // B
    unsigned char* __restrict__ out_succ,     // B
    T* __restrict__ out_trace,                // B x trace_cap x CTRACE_COLS, may be null
    const int trace_cap,
    const grid::robotModel<T>* __restrict__ RM,
    const T eps_pos, const T eps_ori,
    const T lambda_coord, const T h_min, const T max_step,
    const int k_max, const int stall_lim, const int B,
    const int seeds_per_problem,              // S: candidates that share one problem's targets/mask
    const int use_incremental,                // 1 = Phase-4 subtree FK, 0 = full FK (ablation)
    const uint64_t seed,
    // 5D.14c: [P] semantic per-problem RNG roots; null => slot-derived fallback.
    const unsigned int* __restrict__ problem_seeds,
    const int max_pert_attempts,              // bounded retries for a collision-free kick
    // --- Stage 3D/3E self-collision HARD mode. hard_enabled == 0 restores the exact prior path:
    // every branch below is guarded on it, so `off` and `final` execute the same instructions they
    // did before, with the same operands, in the same order.
    const int hard_enabled,
    const int hard_top_k,                     // ranked proposals collision-checked per iteration
    const int hard_oracle_every,              // debug oracle period (0 = off; validation only)
    const float hard_margin,
    g1sc::HardWorkspace hard_ws,              // per-seed persistent state (global memory)
#if defined(HJCD_HAS_COLLISION)
    const int cc_enabled,                     // 1 = exact collision gate (proposals AND kicks)
    const grid::robotModel<float>* __restrict__ RM_cc,
    grid_collision::Environment<float> cc_env)
#else
    const int cc_enabled)
#endif
{
    constexpr int K = hjcd::NT;
    // Compile-time false in the <T,false> instantiation -> every hard-mode block below is
    // eliminated, restoring the byte-identical baseline kernel and its register frame.
    const bool hard_on = HARD && HJCD_HARD_AVAILABLE && (hard_enabled != 0);
    const int lane = threadIdx.x & 31;
    const int gp = blockIdx.x;
    if (gp >= B) return;
    // pid selects problem-level data; gp stays for candidate-level. S == 1 -> pid == gp (identical
    // to the pre-multi-problem path).
    const int pid = gp / seeds_per_problem;
    // 5D.14c: RNG identity must be SEMANTIC. `gp` is the FLATTENED (p,s) block index, so using it
    // made a problem's random stream depend on its slot and on P. `s_local` is the sample index
    // WITHIN the problem; `rng_root` is the planner's per-problem seed.
    const uint32_t s_local  = (uint32_t)(gp - pid * seeds_per_problem);
    const uint32_t rng_root = (problem_seeds != nullptr)
        ? problem_seeds[pid]
        : seed64_to_32(seed ^ ((unsigned long long)pid * 0x9E3779B97F4A7C15ull));

    // STATIC shared, deliberately: grid_collision::config_free uses the DYNAMIC shared arena for its
    // own sphere-FK extractor (an extern __shared__ inside GRiD), so the coarse scratch cannot live
    // there too -- they would alias at offset 0. The dynamic arena is reserved for the collision
    // evaluator; when collision is off, the kernel is launched with none.
    __shared__ CoarseScratch<T> st_storage;
    CoarseScratch<T>* st = &st_storage;

    __shared__ T s_tmp[NX * 2];
    __shared__ int s_topo[hjcd::TOPO];
#if defined(HJCD_HAS_COLLISION)
    namespace gc = grid_collision;
    constexpr int NS = gc::NUM_COLLISION_SPHERES;
    __shared__ float s_ccq[N];
    __shared__ float s_ccpos[3 * NS];
    __shared__ float s_ccr[NS];
#endif

    if (lane < N) st->s_x[lane] = x[(size_t)gp * N + lane];      // seed: candidate-level
    // Base pose: CANDIDATE-level (gp), unlike the targets, which are problem-level (pid). Every
    // seed of a problem may sit at a different base -- that is the whole point of the feature.
    if (lane == 0) {
        if (base_p != nullptr) {
            #pragma unroll
            for (int c = 0; c < 3; ++c) st->s_base_p[c] = base_p[(size_t)gp * 3 + c];
            #pragma unroll
            for (int c = 0; c < 4; ++c) st->s_base_q[c] = base_q[(size_t)gp * 4 + c];
        } else {                                                 // fixed base: identity
            st->s_base_p[0] = st->s_base_p[1] = st->s_base_p[2] = (T)0;
            st->s_base_q[0] = (T)1;
            st->s_base_q[1] = st->s_base_q[2] = st->s_base_q[3] = (T)0;
        }
    }
    __syncwarp(FULL_WARP_MASK);        // lanes 1..K-1 read s_base_* just below
    if (lane < K) {
        st->s_wp[lane] = w_pos[(size_t)pid * K + lane];          // weights: problem-level
        st->s_wo[lane] = w_ori[(size_t)pid * K + lane];
        // Store this candidate's targets in ITS OWN base frame (hjcd_settings.h, eq. 2). A null
        // base_p takes the verbatim-copy branch, so fixed base stays bit-identical to before.
        world_target_to_base<T>(&tgt_p[((size_t)pid*K + lane)*3], &tgt_q[((size_t)pid*K + lane)*4],
                                base_p == nullptr ? nullptr : st->s_base_p, st->s_base_q,
                                &st->s_tgt_p[3*lane], &st->s_tgt_q[4*lane]);
    }
    if (lane == 0) { st->active = active[pid]; st->stall = 0; }  // mask: problem-level
    __syncwarp(FULL_WARP_MASK);

    grid::load_update_XmatsHom_helpers<T>(st->s_XmatsHom, s_topo, st->s_x, RM, s_tmp);
    __syncwarp(FULL_WARP_MASK);
    coarse_full_refresh<T>(st);
    __syncwarp(FULL_WARP_MASK);

    // ------------------------------------------------------------------------------------------
    // THE FEASIBILITY INVARIANT: best_x -- and therefore the returned configuration -- is only ever
    // copied from a state that satisfies every enabled hard constraint. `cur_free` tracks whether
    // the CURRENT search state is feasible, and it is the gate on every best_x write below.
    //
    // The seed is a candidate like any other, so it is checked too: a colliding seed must not become
    // the answer merely because nothing better turned up. When it does collide we start best_total
    // at +inf, so the first feasible state displaces it however poor its task cost. If NOTHING
    // feasible is ever reached, best_x falls back to the seed -- there is no feasible answer to
    // give, and the caller's own collision_free() check will say so.
    // ------------------------------------------------------------------------------------------
    int cur_free = 1;
#if defined(HJCD_HAS_COLLISION)
    if (cc_enabled) {
        for (int i = lane; i < N; i += WARP_SIZE) s_ccq[i] = (float)st->s_x[i];
        __syncthreads();
        cur_free = gc::config_free<float>(s_ccq, RM_cc, cc_env, s_ccpos, s_ccr, nullptr) ? 1 : 0;
        __syncthreads();
    }
#endif
    // best_x is tracked on E_phys, not on the row-scaled `total` -- the coarse search re-freezes its
    // row scales every iteration too, so `total` is no more comparable across iterations here than it
    // was in the LM. An INFEASIBLE seed starts at +inf so the first feasible state displaces it.
    if (lane == 0)
        st->best_ephys = cur_free ? e_phys<T>(st->s_pn, st->s_on, st->active, eps_pos, eps_ori)
                                  : (T)INFINITY;
    if (lane == 0) st->best_total = cur_free ? st->total : (T)INFINITY;
    if (lane < N) st->best_x[lane] = st->s_x[lane];
    __syncwarp(FULL_WARP_MASK);

    int c_iters = 0, c_accept = 0, c_reject = 0, c_stalls = 0, c_perturb = 0;

    // Stage 3D contract: a seed for which no collision-free configuration could be found does NOT
    // enter coordinate search. Zero iterations leaves best_x == the seed, so the caller still gets
    // a well-formed row (errors, cost) -- it is just marked failed host-side, never returned as a
    // collision-free answer.
    const int iters_max = (hard_on && !(hard_ws.flags[gp] & g1sc::HARD_FLAG_STATE_VALID)) ? 0 : k_max;

    for (int it = 0; it < iters_max; ++it) {
        ++c_iters;

        // --- 1. freeze the row scaling and re-express the cost under it -------------------------
        compute_row_scales_warp<T>(st->s_scale, st->s_jointX, st->s_target_X, st->active);
        eval_targets_full_warp<T>(st->s_target_X, st->s_tgt_p, st->s_tgt_q, st->s_wp, st->s_wo,
                                  st->s_scale, st->active, st->s_e_pos, st->s_e_ori,
                                  st->s_pn, st->s_on, st->s_ck, &st->total);

        // --- 2/3. one aggregate proposal per joint lane, clipped, with linearized improvement ----
        T v = (T)0, delta = (T)0, pred = (T)-1;
        if (lane < N) coord_proposal<T>(st, lane, lambda_coord, h_min, max_step, &v, &delta, &pred);

        const T cost_before = st->total;
        int accepted = 0, perturbed = 0;
        int best_j = -1; T best_p = (T)-1, best_v = (T)0, best_d = (T)0;

        // ------------------------------------------------------------------------------------
        // STAGE 3E -- TOP-K RANKED, COLLISION-GATED COMMIT.
        //
        // `off`/`final` run this loop exactly once (n_ranks == 1) over the same reduction, the same
        // trial and the same accept test as before -- byte-identity is by construction, not by
        // re-derivation. Hard mode retains the top `hard_top_k` DISTINCT proposals (distinct by
        // joint: `tried` masks a joint out once it has been offered, so a duplicate proposal can
        // never occupy two ranks) and commits the first one that is collision-free.
        //
        // A rank is consumed ONLY by a collision rejection. A proposal that fails the ordinary
        // exact-cost test ends the iteration exactly as it does in off mode and falls through to
        // the existing stagnation behaviour -- hard mode adds a collision filter, it does not
        // change what counts as progress.
        // ------------------------------------------------------------------------------------
        const int n_ranks = hard_on
                          ? (hard_top_k < 1 ? 1 : (hard_top_k > g1sc::HARD_MAX_K
                                                   ? g1sc::HARD_MAX_K : hard_top_k))
                          : 1;
        unsigned int tried = 0u;
        int hard_rank = -1, hard_coll_rejects = 0;

        for (int rank = 0; rank < n_ranks; ++rank) {
        // --- 4. warp-wide best-proposal reduction (ties -> lowest joint index) -------------------
        // Restricted to joints not yet offered this iteration. An INVALID proposal (pred <= 0 --
        // no affected target, curvature below the floor, or a joint-limit projection that left the
        // value unmoved) is excluded by the same `pred > 0` test as before, so it is never
        // collision-checked and never occupies a rank.
        best_j = ((pred > (T)0) && !((tried >> lane) & 1u)) ? lane : -1;
        best_p = pred; best_v = v; best_d = delta;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const T   op = __shfl_down_sync(FULL_WARP_MASK, best_p, off);
            const int oj = __shfl_down_sync(FULL_WARP_MASK, best_j, off);
            const T   ov = __shfl_down_sync(FULL_WARP_MASK, best_v, off);
            const T   od = __shfl_down_sync(FULL_WARP_MASK, best_d, off);
            const bool take = (oj >= 0) && (best_j < 0 || op > best_p ||
                                            (op == best_p && oj < best_j));
            if (take) { best_p = op; best_j = oj; best_v = ov; best_d = od; }
        }
        best_j = __shfl_sync(FULL_WARP_MASK, best_j, 0);
        best_p = __shfl_sync(FULL_WARP_MASK, best_p, 0);
        best_v = __shfl_sync(FULL_WARP_MASK, best_v, 0);
        best_d = __shfl_sync(FULL_WARP_MASK, best_d, 0);
        if (best_j < 0) break;                  // no untried valid proposal left
        tried |= (1u << best_j);

        int hard_collided = 0;
        {
            const unsigned int desc = hjcd_gen::JOINT_DESCENDANT_MASK[best_j];
            const unsigned int tm   = hjcd_gen::JOINT_TARGET_MASK[best_j] & st->active;

            // --- 5a. save (Phase-4 rollback shadow) --------------------------------------------
            if (lane == 0) { st->v_theta = st->s_x[best_j]; st->v_total = st->total; }
            if (lane < K && ((tm >> lane) & 1u)) {
                const int k = lane;
                #pragma unroll
                for (int c = 0; c < 3; ++c) { st->v_e_pos[3*k+c] = st->s_e_pos[3*k+c];
                                              st->v_e_ori[3*k+c] = st->s_e_ori[3*k+c]; }
                st->v_pn[k] = st->s_pn[k]; st->v_on[k] = st->s_on[k]; st->v_ck[k] = st->s_ck[k];
            }
            __syncwarp(FULL_WARP_MASK);

            // --- 5b. apply, subtree FK, affected targets only, exact cost ----------------------
            if (lane == 0) st->s_x[best_j] = best_v;
            __syncwarp(FULL_WARP_MASK);
            if (use_incremental) {
                update_joint_local_warp<T>(st->s_XmatsHom, st->s_loc16, best_j, best_v);
                subtree_fk_warp<T>(st->s_jointX, st->s_XmatsHom, desc);
                compose_target_frames_masked_warp<T>(st->s_target_X, st->s_jointX, tm);
                if (lane == 0) st->trial_total = st->total;
                __syncwarp(FULL_WARP_MASK);
                eval_targets_masked_warp<T>(st->s_target_X, st->s_tgt_p, st->s_tgt_q,
                                            st->s_wp, st->s_wo, st->s_scale, tm,
                                            st->s_e_pos, st->s_e_ori, st->s_pn, st->s_on, st->s_ck,
                                            &st->trial_total);
            } else {                                   // ablation: full FK + full rescore
                grid::ee_pose_inner_warp<T>(st->s_jointX, st->s_XmatsHom, st->s_x, N - 1);
                __syncwarp(FULL_WARP_MASK);
                compose_target_frames_warp<T>(st->s_target_X, st->s_jointX);
                eval_targets_full_warp<T>(st->s_target_X, st->s_tgt_p, st->s_tgt_q,
                                          st->s_wp, st->s_wo, st->s_scale, st->active,
                                          st->s_e_pos, st->s_e_ori, st->s_pn, st->s_on, st->s_ck,
                                          &st->trial_total);
            }
            __syncwarp(FULL_WARP_MASK);

            // --- 6. accept only on an exact improvement AND exact collision-freedom -------------
            // The task test comes first and is cheap; the collision evaluator only runs when the
            // trial already improves the objective. This is HARD feasibility (grid_collision's
            // config_free: SELF + environment), not a finite penalty -- a colliding trial is
            // rejected outright however good its task cost.
            accepted = (st->trial_total + (T)1e-20 < cost_before) ? 1 : 0;
#if defined(HJCD_HAS_COLLISION)
            if (accepted && cc_enabled) {
                for (int i = lane; i < N; i += WARP_SIZE) s_ccq[i] = (float)st->s_x[i];
                __syncthreads();
                const bool free_ok = gc::config_free<float>(s_ccq, RM_cc, cc_env,
                                                            s_ccpos, s_ccr, nullptr);
                __syncthreads();
                if (!free_ok) accepted = 0;      // colliding: reject, roll back below
            }
#endif
            // --- 6b. SELF-collision gate (Stage 3E). Ordered exactly as the spec requires:
            //         HJCD trial prepared above -> sidecar descendant trial -> incremental verdict
            //         -> commit BOTH or discard BOTH. ws.qc[best_j] is written only on the commit
            //         path, so a partially-updated committed state is unreachable.
            if (hard_on && accepted) {
                hard_collided = g1sc::sidecar_hard_trial(hard_ws, gp, best_j, (float)best_v,
                                                         hard_margin, lane, nullptr);
                // Debug oracle (spec section 11): deterministically sampled, never on in a
                // performance run. The workspace still holds the TRIAL transforms here, which is
                // exactly what the full sweep must see.
                if (ORACLE && hard_oracle_every > 0 && (c_iters % hard_oracle_every) == 0) {
                    const int bad = g1sc::sidecar_hard_oracle(hard_ws, gp, hard_margin, lane,
                                                              hard_collided);
                    if (lane == 0) {
                        g1sc::hard_ctr_add(hard_ws, gp, g1sc::HARD_CTR_ORACLE_CHECKS, 1);
                        if (bad) g1sc::hard_ctr_add(hard_ws, gp, g1sc::HARD_CTR_ORACLE_MISMATCH, 1);
                    }
                }
                if (hard_collided) {
                    g1sc::sidecar_hard_restore(hard_ws, gp, best_j, lane);  // byte-identical undo
                    accepted = 0;
                    ++hard_coll_rejects;
                } else {
                    g1sc::sidecar_hard_commit(hard_ws, gp, best_j, (float)best_v, lane);
                    g1sc::sidecar_hard_mark_free(hard_ws, gp, lane);
                }
            }

            if (accepted) {
                // It cleared the gate above (or there is no gate), so the new state is feasible.
                cur_free = 1;
                hard_rank = rank;
                if (lane == 0) { st->total = st->trial_total; st->stall = 0; }
            } else {
                // --- 5c/6. validated rollback -------------------------------------------------
                if (lane == 0) st->s_x[best_j] = st->v_theta;
                __syncwarp(FULL_WARP_MASK);
                update_joint_local_warp<T>(st->s_XmatsHom, st->s_loc16, best_j, st->v_theta);
                subtree_fk_warp<T>(st->s_jointX, st->s_XmatsHom, desc);
                compose_target_frames_masked_warp<T>(st->s_target_X, st->s_jointX, tm);
                if (lane < K && ((tm >> lane) & 1u)) {
                    const int k = lane;
                    #pragma unroll
                    for (int c = 0; c < 3; ++c) { st->s_e_pos[3*k+c] = st->v_e_pos[3*k+c];
                                                  st->s_e_ori[3*k+c] = st->v_e_ori[3*k+c]; }
                    st->s_pn[k] = st->v_pn[k]; st->s_on[k] = st->v_on[k]; st->s_ck[k] = st->v_ck[k];
                }
                if (lane == 0) st->total = st->v_total;
                __syncwarp(FULL_WARP_MASK);
            }
        }
        if (accepted) break;              // committed -- this iteration is done
        if (!hard_collided) break;        // ordinary cost rejection: off-mode semantics, stop here
        }   // ---- end top-K rank loop ----

        // A rejected iteration bumps `stall` EXACTLY ONCE, whatever the reason and however many
        // ranks hard mode burned on it: the counter measures iterations without progress, and a
        // top-K sweep is still one iteration. (In off mode n_ranks == 1, so this is the same
        // single increment the rollback used to make inline.)
        if (!accepted && lane == 0) ++st->stall;
        __syncwarp(FULL_WARP_MASK);
        if (hard_on && hard_ws.ctr && lane == 0) {
            if (accepted && hard_rank >= 0)
                g1sc::hard_ctr_add(hard_ws, gp, g1sc::HARD_CTR_ACCEPT_RANK0 + hard_rank, 1);
            else if (hard_coll_rejects > 0)
                g1sc::hard_ctr_add(hard_ws, gp, g1sc::HARD_CTR_ALLK, 1);
        }
        __syncwarp(FULL_WARP_MASK);

        if (accepted) ++c_accept; else ++c_reject;

        // --- best-state preservation (in the UNSCALED sense the caller cares about) ------------
        // Gated on cur_free: an infeasible current state can never become best_x.
        // Feasibility first (a colliding state can NEVER become best_x), then strictly-better
        // physical merit. A tie leaves the incumbent, so the earliest state of equal merit wins.
        if (lane == 0) {
            st->take_best = 0;
            if (cur_free) {
                const T ep = e_phys<T>(st->s_pn, st->s_on, st->active, eps_pos, eps_ori);
                if (ep < st->best_ephys) { st->best_ephys = ep; st->take_best = 1; }
                if (st->total < st->best_total) st->best_total = st->total;   // reporting only
            }
        }
        __syncwarp(FULL_WARP_MASK);
        if (lane < N && st->take_best) st->best_x[lane] = st->s_x[lane];
        __syncwarp(FULL_WARP_MASK);

        // --- stall -> COLLISION-GATED random perturbation ---------------------------------------
        // A perturbation is EXPLORATORY: unlike a coordinate proposal it does NOT have to reduce the
        // task cost to become the current state -- that is the whole point of a kick. But it is not
        // exempt from the HARD constraints. Before this gate existed the kick rewrote every joint
        // unchecked, and because best_x is copied from the current state, a colliding kick could
        // become the returned answer. Measured on bookshelf_small_panda with collision-free seeds:
        // 5/64 returned configs collided at stall_lim=5, and 0/64 with kicks disabled.
        //
        // Semantics: save -> kick -> full refresh -> exact config_free -> keep, or restore exactly.
        // Bounded retries (max_pert_attempts), stopping at the first feasible kick. If every attempt
        // collides the pre-perturbation state is restored bitwise, best_x is left alone, and the
        // stall counter is reset anyway -- the alternative (leaving stall >= stall_lim) would retry
        // the kick on every subsequent iteration and spin. The event is counted as `exhausted` so a
        // pathologically boxed-in problem is visible rather than silent.
        int p_att = 0, p_rej = 0, p_exh = 0;
        if (hard_on && st->stall >= stall_lim) {
            // ---- Stage 3F boundary (deliberately NOT crossed in this checkpoint) ----------------
            // A kick rewrites EVERY joint at once, which invalidates the committed sidecar state
            // wholesale -- there is no single-joint descendant subtree to refresh, so validating one
            // needs a full re-check and a full re-init of the committed transforms. That is exactly
            // the random-perturbation collision integration this checkpoint is scoped to exclude.
            // Retain the collision-free state untouched, count the skip so it is visible rather than
            // silent, and clear `stall` so the search does not respin on the same trigger forever.
            ++c_stalls;
            if (lane == 0) {
                st->stall = 0;
                g1sc::hard_ctr_add(hard_ws, gp, g1sc::HARD_CTR_PERT_SKIPPED, 1);
            }
            __syncwarp(FULL_WARP_MASK);
        } else if (st->stall >= stall_lim) {
            ++c_stalls;

            // save the pre-perturbation state (bitwise; a kick touches every joint)
            if (lane < N) st->p_x[lane] = st->s_x[lane];
            if (lane == 0) st->p_total = st->total;
            __syncwarp(FULL_WARP_MASK);

            int kick_ok = 0;
            for (int att = 0; att < max_pert_attempts; ++att) {
                ++p_att;
                // perturb from the SAVED config, not the last rejected attempt, so retries do not
                // compound into an ever-larger jump.
                if (lane < N) {
                    T Llo, Lhi; joint_limit<T>(lane, &Llo, &Lhi);
                    const uint32_t h = semantic_rng(
                        rng_root, RNG_SUB_PO_CCD_STALL_PERTURBATION,
                        s_local, (uint32_t)it, (uint32_t)lane, (uint32_t)att);
                    const T u = (T)((h & 0xFFFFFFu) / (T)0x1000000u) - (T)0.5;
                    const T span = Lhi - Llo;             // in T: no FP64 subtraction on the fp32 path
                    st->s_x[lane] = fmin(fmax(st->p_x[lane] + (T)0.1 * span * u, Llo), Lhi);
                }
                __syncwarp(FULL_WARP_MASK);
                coarse_full_refresh<T>(st);        // a kick invalidates FK, targets, residuals, cost
                __syncwarp(FULL_WARP_MASK);

                int ok = 1;
#if defined(HJCD_HAS_COLLISION)
                if (cc_enabled) {                  // the SAME evaluator the proposal gate uses
                    for (int i = lane; i < N; i += WARP_SIZE) s_ccq[i] = (float)st->s_x[i];
                    __syncthreads();
                    ok = gc::config_free<float>(s_ccq, RM_cc, cc_env,
                                                s_ccpos, s_ccr, nullptr) ? 1 : 0;
                    __syncthreads();
                }
#endif
                if (ok) { kick_ok = 1; break; }    // uniform across the warp: config_free is collective
                ++p_rej;
            }

            if (kick_ok) {
                cur_free = 1;                      // it passed the gate (or there is no gate)
                if (lane == 0) st->stall = 0;
                perturbed = 1;
                ++c_perturb;
            } else {
                // every attempt collided -> restore the saved state EXACTLY and leave best_x alone
                if (lane < N) st->s_x[lane] = st->p_x[lane];
                __syncwarp(FULL_WARP_MASK);
                coarse_full_refresh<T>(st);        // pure function of s_x -> reproduces it exactly
                if (lane == 0) { st->total = st->p_total; st->stall = 0; }
                p_exh = 1;                         // cur_free is UNCHANGED: same state as before
            }
            __syncwarp(FULL_WARP_MASK);
        }
        __syncwarp(FULL_WARP_MASK);

        if (out_trace && lane == 0 && it < trace_cap) {
            T* row = &out_trace[((size_t)gp * trace_cap + it) * CTRACE_COLS];
            row[0] = (T)1;                    // VALID -- explicit
            row[1] = (T)it;
            row[2] = (T)best_j;
            row[3] = best_d;
            row[4] = (best_j >= 0) ? best_p : (T)0;
            row[5] = cost_before;
            row[6] = st->total;
            row[7] = (T)accepted;
            row[8] = (T)st->stall;
            row[9]  = (T)perturbed;   // a kick was RETAINED (collision-free)
            row[10] = (T)p_att;       // kick attempts made this iteration
            row[11] = (T)p_rej;       // attempts rejected by the collision gate
            row[12] = (T)p_exh;       // 1 = every attempt collided, state restored
        }
        __syncwarp(FULL_WARP_MASK);

        if (lane == 0)
            worst_active_errors<T>(st->s_pn, st->s_on, st->s_wp, st->s_wo, st->active,
                                   &st->max_pn, &st->max_on);
        __syncwarp(FULL_WARP_MASK);
        if (all_active_converged<T>(st->s_pn, st->s_on, st->s_wp, st->s_wo, st->active,
                                    eps_pos, eps_ori)) break;
    }

    // report the BEST config seen
    if (lane < N) st->s_x[lane] = st->best_x[lane];
    __syncwarp(FULL_WARP_MASK);
    coarse_full_refresh<T>(st);

    // ---- Stage 3D/3E: publish the collision-free coarse state the LM fallback will use ----------
    // best_x is only ever copied from a state that passed the collision gate (the seed, verified
    // free before the search started, or a committed proposal), so the CONFIGURATION the coarse
    // stage returns is itself collision-free. Recording it here -- rather than whichever accepted
    // state happened to be last -- gives the fallback the best free coarse pose instead of an
    // arbitrary one. The committed q is re-synced to it so the sidecar's q and the solver's q still
    // describe the same configuration on exit. The committed TRANSFORMS are left describing the
    // last trial; nothing reads them after this point, and the next call re-inits them.
    // ONLY for a seed that actually has a valid committed state. best_x for a Stage-3D failure is
    // the colliding seed itself -- publishing that as last_collision_free_coarse_q would hand the
    // section-8 fallback a colliding pose and call it free. (Measured before this guard existed:
    // 13/256 seeds failed Stage 3D and all 13 produced a colliding "collision-free" fallback.)
    //
    // `qc` is deliberately NOT rewritten here: it must keep describing the same configuration as
    // the committed transforms (spec section 5), and those describe the last committed trial, not
    // best_x. qfree is the published answer; qc stays the coherent committed state.
    if (hard_on && (hard_ws.flags[gp] & g1sc::HARD_FLAG_STATE_VALID)) {
        for (int i = lane; i < N && i < g1sc::N_JOINTS; i += WARP_SIZE)
            hard_ws.qfree[(size_t)gp * g1sc::N_JOINTS + i] = (float)st->best_x[i];
        if (lane == 0) hard_ws.flags[gp] |= g1sc::HARD_FLAG_HAS_FREE_Q;
        __syncwarp(FULL_WARP_MASK);
    }

    if (lane < K) {
        out_pn[(size_t)gp * K + lane] = st->s_pn[lane];
        out_on[(size_t)gp * K + lane] = st->s_on[lane];
    }
    if (lane < N) x[(size_t)gp * N + lane] = st->s_x[lane];
    if (lane == 0) {
        out_cost[gp] = st->total;
        out_succ[gp] = all_active_converged<T>(st->s_pn, st->s_on, st->s_wp, st->s_wo,
                                               st->active, eps_pos, eps_ori) ? 1 : 0;
    }
    (void)c_iters; (void)c_accept; (void)c_reject; (void)c_stalls; (void)c_perturb;
}

// ---------------------------------------------------------------------------
// Phase 4 probe: run a SEQUENCE of coordinate updates through the incremental path, each accepted or
// rejected, and dump the resulting state. Tests compare it against a fresh full FK at the final q.
// This is the exact trial/rollback machinery Phase 5's coarse search will call.
//
// Trial:     save theta_j + the affected targets' cached state + C  ->  apply  ->  update j's local
//            ->  subtree FK  ->  recompose affected targets  ->  incremental cost.
// Rollback:  restore theta_j -> update j's local -> subtree FK -> recompose affected targets
//            -> restore the cached residual/cost state bitwise.
// Nothing copies the whole transform array.
// ---------------------------------------------------------------------------
template<typename T>
__global__ void incremental_probe_kernel(
    const T* __restrict__ q0,                 // B x N
    const int* __restrict__ upd_j,            // B x M   joint index per step
    const T* __restrict__ upd_v,              // B x M   new joint value
    const unsigned char* __restrict__ accept, // B x M   1 = keep, 0 = roll back
    const int M,
    const T* __restrict__ tgt_p, const T* __restrict__ tgt_q,
    const unsigned int* __restrict__ active,
    const T* __restrict__ w_pos, const T* __restrict__ w_ori,
    T* __restrict__ out_q,        // B x N
    T* __restrict__ out_jointX,   // B x N*16
    T* __restrict__ out_targetX,  // B x K*16
    T* __restrict__ out_e_pos, T* __restrict__ out_e_ori,
    T* __restrict__ out_pn, T* __restrict__ out_on, T* __restrict__ out_ck,
    T* __restrict__ out_total,    // B
    const grid::robotModel<T>* __restrict__ RM, const int B)
{
    const int b = blockIdx.x;
    if (b >= B) return;
    constexpr int K = hjcd::NT;
    const int lane = threadIdx.x & 31;

    __shared__ T s_q[N], s_XmatsHom[grid::XHOM_T_COUNT], s_jointX[N*16], s_target_X[K*16];
    __shared__ T s_tmp[NX*2], s_loc16[16];
    __shared__ int s_topo[hjcd::TOPO];
    __shared__ T s_tp[K*3], s_tq[K*4], s_wp[K], s_wo[K];
    __shared__ T s_e_pos[K*3], s_e_ori[K*3], s_pn[K], s_on[K], s_ck[K], s_total;
    // rollback cache (only the affected targets are ever touched, but sizing by K is trivial)
    __shared__ T v_e_pos[K*3], v_e_ori[K*3], v_pn[K], v_on[K], v_ck[K], v_total, v_theta;

    for (int i = threadIdx.x; i < N; i += blockDim.x) s_q[i] = q0[(size_t)b*N + i];
    for (int i = threadIdx.x; i < K*3; i += blockDim.x) s_tp[i] = tgt_p[(size_t)b*K*3 + i];
    for (int i = threadIdx.x; i < K*4; i += blockDim.x) s_tq[i] = tgt_q[(size_t)b*K*4 + i];
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        s_wp[i] = w_pos[(size_t)b*K + i];
        s_wo[i] = w_ori[(size_t)b*K + i];
    }
    __syncthreads();

    // Baseline: one full FK + full target compose + full residual/cost cache.
    grid::load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topo, s_q, RM, s_tmp);
    __syncthreads();
    grid::ee_pose_inner_warp<T>(s_jointX, s_XmatsHom, s_q, N - 1);
    __syncwarp(FULL_WARP_MASK);
    compose_target_frames_warp<T>(s_target_X, s_jointX);
    const unsigned int act = active[b];
    eval_targets_full_warp<T>(s_target_X, s_tp, s_tq, s_wp, s_wo, /*s_scale=*/nullptr, act,
                              s_e_pos, s_e_ori, s_pn, s_on, s_ck, &s_total);

    for (int m = 0; m < M; ++m) {
        const int j = upd_j[(size_t)b*M + m];
        const T   v = upd_v[(size_t)b*M + m];
        const bool keep = accept[(size_t)b*M + m] != 0;
        const unsigned int desc = hjcd_gen::JOINT_DESCENDANT_MASK[j];
        const unsigned int tm   = hjcd_gen::JOINT_TARGET_MASK[j] & act;

        // 1. save
        if (lane == 0) { v_theta = s_q[j]; v_total = s_total; }
        if (lane < K && ((tm >> lane) & 1u)) {
            const int k = lane;
            #pragma unroll
            for (int c = 0; c < 3; ++c) { v_e_pos[3*k+c] = s_e_pos[3*k+c];
                                          v_e_ori[3*k+c] = s_e_ori[3*k+c]; }
            v_pn[k] = s_pn[k]; v_on[k] = s_on[k]; v_ck[k] = s_ck[k];
        }
        __syncwarp(FULL_WARP_MASK);

        // 2-4. apply, subtree FK, affected targets only
        if (lane == 0) s_q[j] = v;
        __syncwarp(FULL_WARP_MASK);
        update_joint_local_warp<T>(s_XmatsHom, s_loc16, j, v);
        subtree_fk_warp<T>(s_jointX, s_XmatsHom, desc);
        compose_target_frames_masked_warp<T>(s_target_X, s_jointX, tm);
        eval_targets_masked_warp<T>(s_target_X, s_tp, s_tq, s_wp, s_wo, /*s_scale=*/nullptr, tm,
                                    s_e_pos, s_e_ori, s_pn, s_on, s_ck, &s_total);

        // 5-6. accept, or restore
        if (!keep) {
            if (lane == 0) s_q[j] = v_theta;
            __syncwarp(FULL_WARP_MASK);
            update_joint_local_warp<T>(s_XmatsHom, s_loc16, j, v_theta);
            subtree_fk_warp<T>(s_jointX, s_XmatsHom, desc);
            compose_target_frames_masked_warp<T>(s_target_X, s_jointX, tm);
            if (lane < K && ((tm >> lane) & 1u)) {
                const int k = lane;
                #pragma unroll
                for (int c = 0; c < 3; ++c) { s_e_pos[3*k+c] = v_e_pos[3*k+c];
                                              s_e_ori[3*k+c] = v_e_ori[3*k+c]; }
                s_pn[k] = v_pn[k]; s_on[k] = v_on[k]; s_ck[k] = v_ck[k];
            }
            if (lane == 0) s_total = v_total;
            __syncwarp(FULL_WARP_MASK);
        }
    }

    for (int i = threadIdx.x; i < N; i += blockDim.x) out_q[(size_t)b*N + i] = s_q[i];
    for (int i = threadIdx.x; i < N*16; i += blockDim.x) out_jointX[(size_t)b*N*16 + i] = s_jointX[i];
    for (int i = threadIdx.x; i < K*16; i += blockDim.x) out_targetX[(size_t)b*K*16 + i] = s_target_X[i];
    for (int i = threadIdx.x; i < K*3; i += blockDim.x) {
        out_e_pos[(size_t)b*K*3 + i] = s_e_pos[i];
        out_e_ori[(size_t)b*K*3 + i] = s_e_ori[i];
    }
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        out_pn[(size_t)b*K + i] = s_pn[i];
        out_on[(size_t)b*K + i] = s_on[i];
        out_ck[(size_t)b*K + i] = s_ck[i];
    }
    if (threadIdx.x == 0) out_total[b] = s_total;
}

// Micro-benchmark: `iters` coordinate updates on joint j, either through the incremental subtree
// path (mode 1) or a full FK + full recompose + full rescore each time (mode 0).
template<typename T>
__global__ void fk_bench_kernel(
    const T* __restrict__ q0, const int j, const int iters, const int mode,
    const T* __restrict__ tgt_p, const T* __restrict__ tgt_q,
    const unsigned int* __restrict__ active,
    const T* __restrict__ w_pos, const T* __restrict__ w_ori,
    T* __restrict__ sink, const grid::robotModel<T>* __restrict__ RM, const int B)
{
    const int b = blockIdx.x;
    if (b >= B) return;
    constexpr int K = hjcd::NT;
    const int lane = threadIdx.x & 31;

    __shared__ T s_q[N], s_XmatsHom[grid::XHOM_T_COUNT], s_jointX[N*16], s_target_X[K*16];
    __shared__ T s_tmp[NX*2], s_loc16[16];
    __shared__ int s_topo[hjcd::TOPO];
    __shared__ T s_tp[K*3], s_tq[K*4], s_wp[K], s_wo[K];
    __shared__ T s_e_pos[K*3], s_e_ori[K*3], s_pn[K], s_on[K], s_ck[K], s_total;

    for (int i = threadIdx.x; i < N; i += blockDim.x) s_q[i] = q0[(size_t)b*N + i];
    for (int i = threadIdx.x; i < K*3; i += blockDim.x) s_tp[i] = tgt_p[(size_t)b*K*3 + i];
    for (int i = threadIdx.x; i < K*4; i += blockDim.x) s_tq[i] = tgt_q[(size_t)b*K*4 + i];
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        s_wp[i] = w_pos[(size_t)b*K + i];
        s_wo[i] = w_ori[(size_t)b*K + i];
    }
    __syncthreads();
    grid::load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topo, s_q, RM, s_tmp);
    __syncthreads();
    grid::ee_pose_inner_warp<T>(s_jointX, s_XmatsHom, s_q, N - 1);
    __syncwarp(FULL_WARP_MASK);
    compose_target_frames_warp<T>(s_target_X, s_jointX);
    const unsigned int act = active[b];
    eval_targets_full_warp<T>(s_target_X, s_tp, s_tq, s_wp, s_wo, /*s_scale=*/nullptr, act,
                              s_e_pos, s_e_ori, s_pn, s_on, s_ck, &s_total);

    const unsigned int desc = hjcd_gen::JOINT_DESCENDANT_MASK[j];
    const unsigned int tm   = hjcd_gen::JOINT_TARGET_MASK[j] & act;
    T acc = (T)0;
    for (int it = 0; it < iters; ++it) {
        const T v = s_q[j] + (T)1e-7;
        if (mode == 1 || mode == 2) {
            if (lane == 0) s_q[j] = v;
            __syncwarp(FULL_WARP_MASK);
            update_joint_local_warp<T>(s_XmatsHom, s_loc16, j, v);
            if (mode == 1) subtree_fk_warp<T>(s_jointX, s_XmatsHom, desc);       // shipped: 0..N scan
            else           subtree_fk_ffs_warp<T>(s_jointX, s_XmatsHom, desc);  // comparator: __ffs
            compose_target_frames_masked_warp<T>(s_target_X, s_jointX, tm);
            eval_targets_masked_warp<T>(s_target_X, s_tp, s_tq, s_wp, s_wo, /*s_scale=*/nullptr, tm,
                                        s_e_pos, s_e_ori, s_pn, s_on, s_ck, &s_total);
        } else {
            if (lane == 0) s_q[j] = v;
            __syncwarp(FULL_WARP_MASK);
            grid::ee_pose_inner_warp<T>(s_jointX, s_XmatsHom, s_q, N - 1);
            __syncwarp(FULL_WARP_MASK);
            compose_target_frames_warp<T>(s_target_X, s_jointX);
            eval_targets_full_warp<T>(s_target_X, s_tp, s_tq, s_wp, s_wo, /*s_scale=*/nullptr, act,
                                      s_e_pos, s_e_ori, s_pn, s_on, s_ck, &s_total);
        }
        acc += s_total;
    }
    if (threadIdx.x == 0) sink[b] = acc;
}

// Test/reference entry point: dump the accumulated A and b for a batch. Uses the SAME device
// function the LM uses, so a CPU stacked-Jacobian comparison validates the real accumulation.
template<typename T>
__global__ void normal_equations_kernel(
    const T* __restrict__ q, const T* __restrict__ tgt_p, const T* __restrict__ tgt_q,
    const unsigned int* __restrict__ active,
    const T* __restrict__ w_pos, const T* __restrict__ w_ori,
    T* __restrict__ out_A, T* __restrict__ out_b,
    const grid::robotModel<T>* __restrict__ RM, const int B)
{
    const int b = blockIdx.x;
    if (b >= B) return;
    constexpr int K = hjcd::NT;

    __shared__ T s_q[N], s_XmatsHom[grid::XHOM_T_COUNT], s_jointX[N*16], s_target_X[K*16];
    __shared__ T s_tmp[NX*2];
    __shared__ int s_topo[hjcd::TOPO];
    __shared__ T s_e_pos[K*3], s_e_ori[K*3], s_pn[K], s_on[K], s_wp[K], s_wo[K];
    __shared__ T s_tp[K*3], s_tq[K*4];
    __shared__ T s_A[N*N], s_b[N], s_cost;

    for (int i = threadIdx.x; i < N; i += blockDim.x) s_q[i] = q[(size_t)b*N + i];
    for (int i = threadIdx.x; i < K*3; i += blockDim.x) s_tp[i] = tgt_p[(size_t)b*K*3 + i];
    for (int i = threadIdx.x; i < K*4; i += blockDim.x) s_tq[i] = tgt_q[(size_t)b*K*4 + i];
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        s_wp[i] = w_pos[(size_t)b*K + i];
        s_wo[i] = w_ori[(size_t)b*K + i];
    }
    __syncthreads();

    grid::load_update_XmatsHom_helpers<T>(s_XmatsHom, s_topo, s_q, RM, s_tmp);
    __syncthreads();
    grid::ee_pose_inner_warp<T>(s_jointX, s_XmatsHom, s_q, N - 1);
    __syncwarp(FULL_WARP_MASK);
    compose_target_frames_warp<T>(s_target_X, s_jointX);

    const unsigned int act = active[b];
    eval_targets_warp<T>(s_target_X, s_tp, s_tq, s_wp, s_wo, act,
                         s_e_pos, s_e_ori, s_pn, s_on, &s_cost);
    accumulate_normal_equations_warp<T>(s_A, s_b, s_jointX, s_target_X,
                                        s_e_pos, s_e_ori, s_wp, s_wo, /*s_scale=*/nullptr, act);

    for (int i = threadIdx.x; i < N*N; i += blockDim.x) out_A[(size_t)b*N*N + i] = s_A[i];
    for (int i = threadIdx.x; i < N;   i += blockDim.x) out_b[(size_t)b*N + i]   = s_b[i];
}



// ---------------------------------------------------------------------------
// Shared collision-environment binding. The obstacle set and the fp32 robot model are constant per
// problem, so both the coarse search (winner gate) and the post-solve scorers bind the SAME cached
// upload. Returns true when a usable environment is bound.
// ---------------------------------------------------------------------------
#if defined(HJCD_HAS_COLLISION)
static hjcd_env::DeviceEnv g_cc_env;
static std::string g_cc_key;
static bool g_cc_ready = false;
static const grid::robotModel<float>* g_cc_model = nullptr;

bool bind_collision_env(const char* json, const char* set_name, int idx)
{
    if (!json || !set_name) return false;
    const std::string key = std::string(set_name) + "#" + std::to_string(idx);
    if (g_cc_ready && g_cc_key == key) return true;
    if (g_cc_ready) { hjcd_env::free_env(g_cc_env); g_cc_ready = false; }

    nlohmann::json all = nlohmann::json::parse(json);
    nlohmann::json data = hjcd_env::select_problem_instance(all.at("problems"), set_name, idx);
    if (data.contains("valid") && !bool(data["valid"])) return false;

    hjcd_env::HostEnv h = hjcd_env::problem_dict_to_env(data);
    g_cc_env = hjcd_env::upload_env(h);
    g_cc_key = key;
    g_cc_ready = true;
    if (!g_cc_model) g_cc_model = grid::init_robotModel<float>();
    return true;
}
const void* collision_model_ptr() { return g_cc_ready ? (const void*)g_cc_model : nullptr; }
const void* collision_env_ptr()   { return g_cc_ready ? (const void*)&g_cc_env.env : nullptr; }
#else
bool bind_collision_env(const char*, const char*, int) { return false; }
const void* collision_model_ptr() { return nullptr; }
const void* collision_env_ptr()   { return nullptr; }
#endif


// =============================================================================================
// PRECISION (Phase 0B). The GPU compute type CT is float or double. The public C++/Python I/O
// stays double: targets, weights and tolerances are NARROWED to CT once, at the call boundary,
// before the kernel launches, and the results are WIDENED back once, after it finishes. Nothing
// inside the solve loop ever touches the other precision -- there is no per-iteration conversion
// and no double accumulator in an fp32 solve. `q_ct` carries the configuration back in CT so the
// caller can have it at the precision it asked for; errors/costs are reported in double.
//
// The joint limits are the one thing the kernel reads from constant memory rather than from a
// narrowed buffer, so they have their own CT mirror (see c_joint_limits_f / joint_limit<T>).
// =============================================================================================
template<typename CT>
static std::vector<CT> narrow_h(const double* p, size_t n) {
    std::vector<CT> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = (CT)p[i];
    return v;
}
template<typename CT>
static void widen_d2h(const CT* d_src, std::vector<double>& dst, size_t n) {
    std::vector<CT> tmp(n);
    CUDA_OK(cudaMemcpy(tmp.data(), d_src, sizeof(CT) * n, cudaMemcpyDeviceToHost));
    dst.resize(n);
    for (size_t i = 0; i < n; ++i) dst[i] = (double)tmp[i];
}

// =============================================================================================
// PERSISTENT DEVICE WORKSPACE (Phase 0E-A)
//
// Every solve used to cudaMalloc ~10 device buffers, cudaFree them, allocate several std::vector
// staging buffers on the host, and copy the B x N configuration FOUR times on the way out
// (D2H -> widen to double -> narrow to float -> pybind copy into numpy). At B=2000 that marshalling
// was ~5.2 ms of a ~20 ms solve -- the largest single stage after the two kernels.
//
// The workspace is a single capacity-based device arena, sub-allocated with a bump pointer that is
// rewound at the start of every launch. Capacity grows geometrically and NEVER shrinks, so a steady
// stream of same-or-smaller solves performs ZERO cudaMalloc and ZERO cudaFree.
//
// The KEY is (device, precision, batch capacity, trace capacity). A request that changes any of them
// either fits inside the existing capacity or triggers exactly one growth; it never silently reuses
// an incompatible layout.
//
// OWNERSHIP: a workspace is owned by exactly one HJCDSolver. It is NOT a process-global. It is bound
// to the CUDA device current at construction and to the default stream. It is NOT thread-safe: one
// active call per instance, enforced by a re-entrancy guard above this layer.
// =============================================================================================
class HjcdWorkspace {
public:
    HjcdWorkspace() { cudaGetDevice(&device_); }
    ~HjcdWorkspace() { release(); }
    HjcdWorkspace(const HjcdWorkspace&) = delete;
    HjcdWorkspace& operator=(const HjcdWorkspace&) = delete;

    // Bytes needed for one launch at (P problems, B candidates, trace_cap) with element `elem`.
    // PROBLEM-level buffers (targets, weights, mask) are sized by P; CANDIDATE-level buffers by B.
    // The legacy path passes P == B, so this reduces exactly to the old sizing.
    static size_t bytes_for(int P, int B, int trace_cap, size_t elem) {
        const size_t K = (size_t)hjcd::NT, n = (size_t)N, b = (size_t)B, p = (size_t)P;
        size_t t = 0;
        t += align(b * n * elem);                       // q                (candidate)
        t += align(p * K * 3 * elem);                   // tgt_p            (problem)
        t += align(p * K * 4 * elem);                   // tgt_q            (problem)
        t += align(p * K * elem) * 2;                   // wp, wo           (problem)
        t += align(b * K * elem) * 2;                   // pn, on           (candidate)
        t += align(b * elem);                           // cost             (candidate)
        t += align(p * sizeof(unsigned int));           // active           (problem)
        t += align(b * sizeof(unsigned char));          // success          (candidate)
        if (trace_cap > 0) t += align(b * (size_t)trace_cap * 16 * elem);  // trace (>= CTRACE_COLS)
        return t;
    }

    // Grow to fit (P, B, trace_cap, elem). Geometric, never shrinks. Returns true if it allocated.
    bool ensure(int P, int B, int precision, int trace_cap) {
        const size_t elem = precision == 1 ? sizeof(float) : sizeof(double);
        const size_t need = bytes_for(P, B, trace_cap, elem);
        if (need <= bytes_ && precision_ == precision) { head_ = 0; return false; }
        // a precision change reuses the arena when it is already big enough -- the LAYOUT is rebuilt
        // per launch by the bump allocator, so only total size matters.
        if (need <= bytes_) { precision_ = precision; head_ = 0; return false; }
        size_t grow = bytes_ ? bytes_ : need;
        while (grow < need) grow *= 2;                  // geometric
        release();
        CUDA_OK(cudaMalloc(&arena_, grow));
        ++n_malloc_;
        bytes_ = grow; precision_ = precision; head_ = 0;
        cap_B_ = std::max(cap_B_, B); cap_P_ = std::max(cap_P_, P);
        cap_trace_ = std::max(cap_trace_, trace_cap);
        return true;
    }

    // Grow to at least `need` bytes. Used by the batched solve_problems orchestrator, which lays out
    // many buffers (coarse_q, lm_q, per-candidate metrics, selected [P,...] outputs) in ONE arena
    // and bump-allocates them itself. Geometric, never shrinks.
    bool ensure_raw(size_t need, int precision) {
        if (need <= bytes_ && precision_ == precision) { head_ = 0; return false; }
        if (need <= bytes_) { precision_ = precision; head_ = 0; return false; }
        size_t grow = bytes_ ? bytes_ : need;
        while (grow < need) grow *= 2;
        release();
        CUDA_OK(cudaMalloc(&arena_, grow));
        ++n_malloc_;
        bytes_ = grow; precision_ = precision; head_ = 0;
        return true;
    }

    template <typename T> T* take(size_t n) {
        const size_t need = align(n * sizeof(T));
        if (head_ + need > bytes_) return nullptr;      // caller falls back to raw cudaMalloc
        T* p = reinterpret_cast<T*>(static_cast<char*>(arena_) + head_);
        head_ += need;
        return p;
    }
    void rewind() { head_ = 0; }
    void release() {
        if (arena_) { cudaFree(arena_); ++n_free_; arena_ = nullptr; }
        bytes_ = 0; head_ = 0;
    }
    // instrumentation for the allocation-count test
    size_t n_malloc() const { return n_malloc_; }
    size_t n_free() const { return n_free_; }
    size_t bytes() const { return bytes_; }
    int cap_B() const { return cap_B_; }
    int device() const { return device_; }

private:
    static size_t align(size_t n) { return (n + 255u) & ~(size_t)255u; }
    void* arena_ = nullptr;
    size_t bytes_ = 0, head_ = 0;
    size_t n_malloc_ = 0, n_free_ = 0;
    int precision_ = -1, cap_B_ = 0, cap_P_ = 0, cap_trace_ = 0, device_ = -1;
};



// Upload one input array. When the caller's dtype already IS the compute type, this is a plain H2D
// from the numpy buffer -- no host loop, no staging vector. Otherwise it narrows once, into a
// reusable per-launch staging buffer.
// D2H mirror of upload_in. Used for the base, which is an IN/OUT buffer: the caller supplies the
// seed base and reads the optimized base back from the same array.
template <typename CT>
static void download_out(void* dst, const CT* src, bool dst_f32, size_t n, std::vector<CT>& stage) {
    const bool ct_is_f32 = std::is_same<CT, float>::value;
    if (dst_f32 == ct_is_f32) {                       // dtypes match -> straight D2H
        CUDA_OK(cudaMemcpy(dst, src, sizeof(CT) * n, cudaMemcpyDeviceToHost));
        return;
    }
    stage.resize(n);
    CUDA_OK(cudaMemcpy(stage.data(), src, sizeof(CT) * n, cudaMemcpyDeviceToHost));
    if (dst_f32) { float*  q = (float*)dst;  for (size_t i=0;i<n;++i) q[i]=(float)stage[i]; }
    else         { double* q = (double*)dst; for (size_t i=0;i<n;++i) q[i]=(double)stage[i]; }
}

template <typename CT>
static void upload_in(CT* dst, const void* src, bool src_f32, size_t n, std::vector<CT>& stage) {
    const bool ct_is_f32 = std::is_same<CT, float>::value;
    if (src_f32 == ct_is_f32) {                       // dtypes match -> straight H2D
        CUDA_OK(cudaMemcpy(dst, src, sizeof(CT) * n, cudaMemcpyHostToDevice));
        return;
    }
    stage.resize(n);
    if (src_f32) { const float*  p = (const float*)src;  for (size_t i=0;i<n;++i) stage[i]=(CT)p[i]; }
    else         { const double* p = (const double*)src; for (size_t i=0;i<n;++i) stage[i]=(CT)p[i]; }
    CUDA_OK(cudaMemcpy(dst, stage.data(), sizeof(CT) * n, cudaMemcpyHostToDevice));
}

// D2H a CT device buffer into a double host buffer (widening once, in place, no extra vector).
template <typename CT>
static void download_widen(const CT* src, double* dst, size_t n, std::vector<CT>& stage) {
    if (std::is_same<CT, double>::value) {
        CUDA_OK(cudaMemcpy(dst, src, sizeof(double) * n, cudaMemcpyDeviceToHost));
        return;
    }
    stage.resize(n);
    CUDA_OK(cudaMemcpy(stage.data(), src, sizeof(CT) * n, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i) dst[i] = (double)stage[i];
}

// The float robot model is a process-lifetime constant, cached like the double one.
template<typename CT>
static const grid::robotModel<CT>* robot_model_for(const grid::robotModel<double>* d64);
template<> const grid::robotModel<double>* robot_model_for<double>(
        const grid::robotModel<double>* d64) { return d64; }
template<> const grid::robotModel<float>* robot_model_for<float>(
        const grid::robotModel<double>*) {
    static const grid::robotModel<float>* m = grid::init_robotModel<float>();
    return m;
}


// =================================================================================================
// STAGE 3D -- INITIAL COLLISION-FREE STATE + the persistent hard-mode workspace.
//
// Hard mode may not begin coordinate search from a colliding configuration: every later guarantee
// ("every accepted update is collision-free", "the committed verdict for an unaffected pair is
// zero") is inductive, and this is the base case. Each active seed gets ONE batched full check
// through the separate 351-pair checker; a colliding seed is re-drawn with the solver's OWN
// existing kick policy (the same wanghash / 0.1*span formula the stall perturbation uses -- this
// checkpoint introduces no new seed distribution) and re-checked, up to a bounded retry count.
// A seed that never comes back free is marked failed and never enters coordinate search.
// =================================================================================================
namespace g1s = g1_sidecar;

extern "C" const void* sidecar_device_sdf_ptr(int cid);
extern "C" const void* sidecar_device_convex_ptr();

namespace hjcd_hard {

// Persistent, grow-on-demand per-seed state. Allocated ONLY on the first hard-mode call: `off` and
// `final` never reach ensure(), so they allocate nothing and launch nothing (asserted by test).
class Owner {
public:
    int cap_B = 0, n_alloc = 0;
    float*  Tf = nullptr; double* Td = nullptr;
    float*  qc = nullptr; float*  qfree = nullptr; float* q0 = nullptr;
    unsigned char* flags = nullptr; unsigned char* verdict = nullptr; unsigned char* todo = nullptr;
    int* ctr = nullptr;

    // ---- Checkpoint 3D.1 reseed arenas. Sized by (failed seeds x candidates); grown separately
    // from the per-seed state because R is a caller knob and F varies per call.
    int cap_FR = 0, n_alloc_rs = 0;
    float* cand_q = nullptr;            // [F*R, 29] candidate configurations
    float* cand_dist = nullptr;         // [F*R]     normalized joint-space distance to the seed
    unsigned char* cand_free = nullptr; // [F*R]     1 = collision-free
    unsigned char* cand_comp = nullptr; // [F*R]     which distribution component produced it
    int* fail_idx = nullptr;            // [B]       physical row of each still-colliding seed
    unsigned int* fail_fp = nullptr;    // [B]       that seed's CONTENT fingerprint (RNG identity)
    int* sel = nullptr;                 // [B]       chosen candidate ordinal, -1 = none found

    bool ensure_reseed(int FR, int B) {
        if (FR <= cap_FR && fail_idx) return false;
        release_reseed();
        CUDA_OK(cudaMalloc(&cand_q,    (size_t)FR * g1s::N_JOINTS * sizeof(float)));
        CUDA_OK(cudaMalloc(&cand_dist, (size_t)FR * sizeof(float)));
        CUDA_OK(cudaMalloc(&cand_free, (size_t)FR));
        CUDA_OK(cudaMalloc(&cand_comp, (size_t)FR));
        CUDA_OK(cudaMalloc(&fail_idx,  (size_t)B * sizeof(int)));
        CUDA_OK(cudaMalloc(&fail_fp,   (size_t)B * sizeof(unsigned int)));
        CUDA_OK(cudaMalloc(&sel,       (size_t)B * sizeof(int)));
        cap_FR = FR; ++n_alloc_rs;
        return true;
    }
    void release_reseed() {
        for (void* p : {(void*)cand_q,(void*)cand_dist,(void*)cand_free,(void*)cand_comp,
                        (void*)fail_idx,(void*)fail_fp,(void*)sel}) if (p) cudaFree(p);
        cand_q=nullptr; cand_dist=nullptr; cand_free=nullptr; cand_comp=nullptr;
        fail_idx=nullptr; fail_fp=nullptr; sel=nullptr; cap_FR=0;
    }

    bool ensure(int B) {
        if (B <= cap_B) return false;
        release();
        const size_t L = (size_t)g1s::N_LINKS * 16, J = (size_t)g1s::N_JOINTS, b = (size_t)B;
        CUDA_OK(cudaMalloc(&Tf,      b * L * sizeof(float)));
        CUDA_OK(cudaMalloc(&Td,      b * L * sizeof(double)));
        CUDA_OK(cudaMalloc(&qc,      b * J * sizeof(float)));
        CUDA_OK(cudaMalloc(&qfree,   b * J * sizeof(float)));
        CUDA_OK(cudaMalloc(&q0,      b * J * sizeof(float)));
        CUDA_OK(cudaMalloc(&flags,   b));
        CUDA_OK(cudaMalloc(&verdict, b));
        CUDA_OK(cudaMalloc(&todo,    b));
        CUDA_OK(cudaMalloc(&ctr,     b * (size_t)g1sc::HARD_CTR_STRIDE * sizeof(int)));
        cap_B = B; ++n_alloc;
        return true;
    }
    // `diagnostics` decides whether the kernel sees the counter array at all: in the fast path
    // ws.ctr is null and every counter site compiles down to a null test that never stores.
    g1sc::HardWorkspace view(bool diagnostics) const {
        g1sc::HardWorkspace w{};
        w.Tf = Tf; w.Td = Td; w.qc = qc; w.qfree = qfree; w.flags = flags;
        w.ctr = diagnostics ? ctr : nullptr;
        return w;
    }
    void release() {
        for (void* p : {(void*)Tf,(void*)Td,(void*)qc,(void*)qfree,(void*)q0,
                        (void*)flags,(void*)verdict,(void*)todo,(void*)ctr})
            if (p) cudaFree(p);
        Tf=nullptr; Td=nullptr; qc=nullptr; qfree=nullptr; q0=nullptr;
        flags=nullptr; verdict=nullptr; todo=nullptr; ctr=nullptr; cap_B=0;
        release_reseed();
    }
};
static Owner g_ws;

// One warp per seed: full FK (f32 + f64) into the committed transforms, then the FULL 351-pair
// check. This is the SAME geometry the batched full checker runs, reached through the same device
// functions -- Stage 3D does not get its own collision model.
template<typename CT>
__global__ void hard_init_kernel(const CT* __restrict__ x, g1sc::HardWorkspace ws,
                                 float* __restrict__ q0, unsigned char* __restrict__ verdict,
                                 const unsigned char* __restrict__ todo,
                                 int B, float margin, int store_q0)
{
    const int b = blockIdx.x;
    if (b >= B) return;
    if (todo && !todo[b]) return;          // uniform across the block: safe before any __syncwarp
    const int lane = threadIdx.x & 31;

    float*  T  = &ws.Tf[(size_t)b * g1s::N_LINKS * 16];
    double* Td = &ws.Td[(size_t)b * g1s::N_LINKS * 16];
    float*  q  = &ws.qc[(size_t)b * g1s::N_JOINTS];
    for (int i = lane; i < g1s::N_JOINTS && i < N; i += 32) q[i] = (float)x[(size_t)b * N + i];
    __syncwarp();
    if (store_q0)
        for (int i = lane; i < g1s::N_JOINTS; i += 32) q0[(size_t)b * g1s::N_JOINTS + i] = q[i];
    if (lane == 0) { g1sc::sidecar_fk(q, T); g1sc::sidecar_fk_d(q, Td); }
    __syncwarp();

    int hit = 0;
    for (int g = lane; g < g1s::N_CHECKED_PAIRS; g += 32) {
        if (g1s::PAIR_TYPE[g] == g1s::PAIR_CONVEX_GJK) continue;
        if (g1sc::linkpair_colliding_nongjk(g, T, margin)) hit = 1;
    }
    hit = __ballot_sync(0xffffffffu, hit) ? 1 : 0;      // warp-uniform before the cooperative GJK
    if (!hit) {
        for (int g = 0; g < g1s::N_CHECKED_PAIRS; ++g) {
            if (g1s::PAIR_TYPE[g] != g1s::PAIR_CONVEX_GJK) continue;
            if (g1sc::linkpair_colliding_gjk(g, Td, margin, lane)) { hit = 1; break; }
        }
    }
    if (lane == 0) verdict[b] = (unsigned char)hit;
    if (!hit) {                            // free: this q IS a collision-free coarse state already
        for (int i = lane; i < g1s::N_JOINTS; i += 32)
            ws.qfree[(size_t)b * g1s::N_JOINTS + i] = q[i];
        if (lane == 0) ws.flags[b] = g1sc::HARD_FLAG_STATE_VALID | g1sc::HARD_FLAG_HAS_FREE_Q;
    } else if (lane == 0) {
        ws.flags[b] = 0;                   // no valid committed state -> excluded from the search
    }
}

// Re-draw a colliding seed with the EXISTING kick policy (coarse_search_mt_kernel's stall
// perturbation, formula for formula), always measured from the ORIGINAL seed so retries do not
// compound into an ever-larger jump.
template<typename CT>
__global__ void hard_reseed_kernel(CT* __restrict__ x, const float* __restrict__ q0,
                                   const unsigned char* __restrict__ todo,
                                   int B, uint64_t seed, int attempt)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = idx / N, j = idx % N;
    if (b >= B || !todo[b]) return;
    CT Llo, Lhi; joint_limit<CT>(j, &Llo, &Lhi);
    // CONTENT-addressed, not index-addressed. Keying the draw on the batch index would make a
    // seed's recovery depend on where it happens to sit in the batch, and a permuted batch would
    // return different answers for the same seeds (measured: 2/128 rows diverged before this
    // change). The seed's own bits are its stable identity. The perturbation formula, magnitude
    // and distribution are the existing stall-kick policy, unchanged -- only the stream key moved.
    uint32_t sh = 0x9E3779B9u;
    for (int i = 0; i < g1s::N_JOINTS; ++i)
        sh = wanghash(sh ^ __float_as_uint(q0[(size_t)b * g1s::N_JOINTS + i]));
    const uint32_t h = wanghash((uint32_t)(seed ^ (uint64_t)(
        sh + (uint32_t)j * 131u + (uint32_t)attempt * 0x9E3779B9u)));
    const CT u = (CT)((h & 0xFFFFFFu) / (CT)0x1000000u) - (CT)0.5;
    const CT span = Lhi - Llo;
    const CT base = (j < g1s::N_JOINTS) ? (CT)q0[(size_t)b * g1s::N_JOINTS + j] : (CT)0;
    x[(size_t)b * N + j] = fmin(fmax(base + (CT)0.1 * span * u, Llo), Lhi);
}

// =================================================================================================
// CHECKPOINT 3D.1 -- DEDICATED COLLISION-FREE SEED GENERATOR.
//
// WHY THE OLD RESEED FAILED. Stage 3D originally re-drew a colliding seed with the coarse search's
// own stall kick: a +-5%-of-joint-span jitter around the original seed. Measured at B=2000 that
// recovered 0 of 132 colliding seeds, and 147-253 of 256 on the harder problems. A seed that is
// deep inside a self-collision is not 5% of a span away from a free one; the kick was never a
// recovery mechanism, only an escape-from-a-local-minimum mechanism, and reusing it as one was the
// mistake. This stage replaces it with a broad, explicitly-parameterised candidate MIXTURE that is
// generated, checked and selected entirely on device.
//
// The full checker stays where it belongs -- its own batched kernel over the candidate array. It is
// never inlined into the coarse-search kernel, so the hard coarse kernel's 226/233-register frame
// is untouched by anything here.
// =================================================================================================
namespace reseed {

// Distribution components. Recorded per candidate so the benchmark can report WHICH kind of
// candidate actually rescued each seed, rather than just that something did.
enum : unsigned char { COMP_PERTURB = 0, COMP_NOMINAL = 1, COMP_BROAD = 2 };

static constexpr int MAX_SCALES = 8;
struct Config {
    int   candidates = 16;                       // R per failed seed per round
    int   rounds = 2;                            // bounded retry rounds
    int   n_scales = 4;
    float scales[MAX_SCALES] = {0.10f, 0.20f, 0.35f, 0.50f};   // fractions of joint span
    float nominal_jitter = 0.15f;                // component B jitter, fraction of span
    float round_broaden = 2.0f;                  // scales multiplier per extra round
};

// Split R into the three components. Every component gets at least one candidate at any R >= 3,
// so a small pool still samples all three rather than degenerating to perturbations only.
__host__ __device__ __forceinline__ void split_counts(int R, int* nA, int* nB, int* nC) {
    int a = R * 6 / 10, b = R * 2 / 10;
    if (a < 1) a = 1;
    if (b < 1) b = 1;
    int c = R - a - b;
    if (c < 1) { c = 1; if (a + b + c > R) { a = R - b - c; if (a < 1) { a = 1; b = R - a - c; } } }
    *nA = a; *nB = b; *nC = (R - a - b) > 0 ? (R - a - b) : 0;
}

// The G1's nominal crouch, joint-for-joint the configuration the sidecar corpus calls "crouch"
// (hip_pitch -0.6, knee +1.2, ankle_pitch -0.6, both legs; everything else zero). Together with
// the all-zero neutral pose these are the two configurations independently verified collision-free,
// which is exactly what makes them useful anchors for a recovery draw.
__device__ __forceinline__ float nominal_value(int j, int which) {
    if (which == 0) return 0.0f;                                   // neutral
    if (j == 0  || j == 6)  return -0.6f;                          // {l,r}_hip_pitch
    if (j == 3  || j == 9)  return  1.2f;                          // {l,r}_knee
    if (j == 4  || j == 10) return -0.6f;                          // {l,r}_ankle_pitch
    return 0.0f;
}

// A seed's CONTENT fingerprint. This -- not its row -- is its logical identity for RNG purposes.
// Keying on the batch position is what made a permuted batch return different answers for the same
// seeds; a content key makes permutation a pure relabelling of outputs.
__global__ void fingerprint_kernel(const float* __restrict__ q0, const int* __restrict__ fail_idx,
                                   unsigned int* __restrict__ fp, int F) {
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= F) return;
    const int b = fail_idx[f];
    unsigned int h = 0x9E3779B9u;
    for (int i = 0; i < g1s::N_JOINTS; ++i)
        h = wanghash(h ^ __float_as_uint(q0[(size_t)b * g1s::N_JOINTS + i]));
    fp[f] = h;
}

// One thread per (failed seed, candidate, joint). Every value is projected exactly into the joint
// limits, and the per-candidate normalised distance to the original seed is reduced by joint 0.
__global__ void generate_kernel(const float* __restrict__ q0, const int* __restrict__ fail_idx,
                                const unsigned int* __restrict__ fp,
                                float* __restrict__ cand_q, float* __restrict__ cand_dist,
                                unsigned char* __restrict__ cand_comp,
                                int F, int R, int round, Config cfg, unsigned long long seed) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)F * R * g1s::N_JOINTS;
    if (tid >= total) return;
    const int j = (int)(tid % g1s::N_JOINTS);
    const int r = (int)((tid / g1s::N_JOINTS) % R);
    const int f = (int)(tid / ((long long)R * g1s::N_JOINTS));
    const int b = fail_idx[f];

    float lo, hi; joint_limit<float>(j, &lo, &hi);
    const float span = hi - lo;
    const float base_q = q0[(size_t)b * g1s::N_JOINTS + j];

    int nA, nB, nC; split_counts(R, &nA, &nB, &nC); (void)nC;
    // RNG key: logical seed identity (content fingerprint) x round x candidate x joint. No term
    // depends on where the seed sits in the batch.
    const unsigned int h = wanghash((unsigned int)(seed) ^ wanghash(
        fp[f] ^ ((unsigned int)round * 0x85EBCA6Bu)
              ^ ((unsigned int)r * 0xC2B2AE35u)
              ^ ((unsigned int)j * 0x27D4EB2Du)));
    const float u = (float)(h & 0xFFFFFFu) / (float)0x1000000u;    // [0,1)
    const float u2 = u - 0.5f;                                     // [-0.5,0.5)

    // Broaden with the round: a second round is only worth running if it searches somewhere the
    // first did not.
    float broaden = 1.0f;
    for (int i = 0; i < round; ++i) broaden *= cfg.round_broaden;

    float v;
    unsigned char comp;
    if (r < nA) {                                   // A: perturb the original seed at several scales
        const float sc = cfg.scales[(r % (cfg.n_scales < 1 ? 1 : cfg.n_scales))] * broaden;
        v = base_q + sc * span * u2;
        comp = COMP_PERTURB;
    } else if (r < nA + nB) {                       // B: neutral / crouch anchored, moderate jitter
        const int which = (r - nA) & 1;
        v = nominal_value(j, which) + cfg.nominal_jitter * broaden * span * u2;
        comp = COMP_NOMINAL;
    } else {                                        // C: broad, uniform across the whole limit range
        v = lo + u * span;
        comp = COMP_BROAD;
    }
    v = fminf(fmaxf(v, lo), hi);                    // exact projection into the joint limits
    const long long ci = (long long)f * R + r;
    cand_q[ci * g1s::N_JOINTS + j] = v;
    if (j == 0) cand_comp[ci] = comp;

    // Normalised distance to the original seed, reduced across the 29 joints by joint 0. The
    // per-joint terms are written first and read back after a grid-wide ordering guarantee we do
    // NOT have, so the reduction is done in the selection kernel instead -- see select_kernel.
    (void)cand_dist;
}

// One warp per candidate: full FK + the complete 351-pair check, in the SAME device functions the
// batched full checker uses. Shared-memory transforms, so no per-candidate global scratch (which
// would be 7.7 KB/candidate, ~1 GB at F=2000, R=64).
__global__ void check_kernel(const float* __restrict__ cand_q, unsigned char* __restrict__ cand_free,
                             long long FR, float margin) {
    __shared__ float  shT[g1s::N_LINKS * 16];
    __shared__ double shTd[g1s::N_LINKS * 16];
    const long long ci = blockIdx.x;
    if (ci >= FR) return;
    const int lane = threadIdx.x & 31;
    const float* q = &cand_q[ci * g1s::N_JOINTS];
    if (lane == 0) { g1sc::sidecar_fk(q, shT); g1sc::sidecar_fk_d(q, shTd); }
    __syncwarp();
    int hit = 0;
    for (int g = lane; g < g1s::N_CHECKED_PAIRS; g += 32) {
        if (g1s::PAIR_TYPE[g] == g1s::PAIR_CONVEX_GJK) continue;
        if (g1sc::linkpair_colliding_nongjk(g, shT, margin)) hit = 1;
    }
    hit = __ballot_sync(0xffffffffu, hit) ? 1 : 0;
    if (!hit) {
        for (int g = 0; g < g1s::N_CHECKED_PAIRS; ++g) {
            if (g1s::PAIR_TYPE[g] != g1s::PAIR_CONVEX_GJK) continue;
            if (g1sc::linkpair_colliding_gjk(g, shTd, margin, lane)) { hit = 1; break; }
        }
    }
    if (lane == 0) cand_free[ci] = (unsigned char)(hit ? 0 : 1);
}

// One thread per failed seed. Among the COLLISION-FREE candidates pick the one closest to the
// original seed in normalised joint space, ties broken by candidate ordinal -- a total order, so
// the choice is deterministic and independent of scheduling.
template<typename CT>
__global__ void select_kernel(CT* __restrict__ x, const float* __restrict__ q0,
                              const int* __restrict__ fail_idx,
                              const float* __restrict__ cand_q,
                              const unsigned char* __restrict__ cand_free,
                              float* __restrict__ cand_dist, int* __restrict__ sel,
                              int F, int R) {
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= F) return;
    const int b = fail_idx[f];
    int best = -1;
    float best_d = 3.4e38f;
    for (int r = 0; r < R; ++r) {
        const long long ci = (long long)f * R + r;
        if (!cand_free[ci]) continue;
        float d2 = 0.0f;
        for (int j = 0; j < g1s::N_JOINTS; ++j) {
            float lo, hi; joint_limit<float>(j, &lo, &hi);
            const float t = (cand_q[ci * g1s::N_JOINTS + j] - q0[(size_t)b * g1s::N_JOINTS + j])
                          / (hi - lo);
            d2 += t * t;
        }
        cand_dist[ci] = sqrtf(d2);
        if (d2 < best_d) { best_d = d2; best = r; }    // strict <: earliest ordinal wins a tie
    }
    sel[f] = best;
    if (best >= 0) {
        const long long ci = (long long)f * R + best;
        for (int j = 0; j < g1s::N_JOINTS && j < N; ++j)
            x[(size_t)b * N + j] = (CT)cand_q[ci * g1s::N_JOINTS + j];
    }
}

}  // namespace reseed

struct InitReport {
    int initially_free = 0, initially_colliding = 0, reseed_attempts = 0;
    int recovered = 0, failures = 0;
    double ms = 0.0;
    // Checkpoint 3D.1 breakdown
    int rounds_run = 0, candidates_checked = 0;
    int sel_perturb = 0, sel_nominal = 0, sel_broad = 0;
    double gen_ms = 0.0, check_ms = 0.0, select_ms = 0.0, verify_ms = 0.0;
};

// Bind the solver TU's copies of the sidecar model pointers to the sidecar TU's allocations. Once
// per process; the allocations are themselves uploaded once (hjcdik._ensure_self_collision_sidecar).
static bool bind_model_once() {
    static int state = 0;                  // 0 unbound, 1 bound, -1 model not uploaded
    if (state) return state == 1;
    const void* sdf[8] = {nullptr};
    for (int c = 0; c < g1s::N_CLUSTERS && c < 8; ++c) sdf[c] = sidecar_device_sdf_ptr(c);
    const void* cv = sidecar_device_convex_ptr();
    if (!cv || !sdf[0]) { state = -1; return false; }
    hjcd_hard_bind_model(sdf, g1s::N_CLUSTERS, cv);
    state = 1;
    return true;
}

// Full Stage-3D pass: check every seed, reseed the colliding ones (bounded), report.
template<typename CT>
static InitReport prepare(CT* d_x, int B, float margin, int max_reseed, uint64_t seed,
                          bool diagnostics, int rs_mode, reseed::Config rs_cfg)
{
    InitReport rep;
    cudaEvent_t t0, t1; CUDA_OK(cudaEventCreate(&t0)); CUDA_OK(cudaEventCreate(&t1));
    CUDA_OK(cudaEventRecord(t0));

    g_ws.ensure(B);
    g1sc::HardWorkspace w = g_ws.view(diagnostics);
    CUDA_OK(cudaMemset(g_ws.ctr, 0, (size_t)B * g1sc::HARD_CTR_STRIDE * sizeof(int)));
    CUDA_OK(cudaMemset(g_ws.todo, 1, (size_t)B));
    CUDA_OK(cudaMemset(g_ws.flags, 0, (size_t)B));

    hard_init_kernel<CT><<<B, 32>>>(d_x, w, g_ws.q0, g_ws.verdict, nullptr, B, margin, 1);
    CUDA_OK(cudaPeekAtLastError());

    std::vector<unsigned char> v(B);
    CUDA_OK(cudaMemcpy(v.data(), g_ws.verdict, B, cudaMemcpyDeviceToHost));
    for (int b = 0; b < B; ++b) if (v[b]) ++rep.initially_colliding;
    rep.initially_free = B - rep.initially_colliding;

    int remaining = rep.initially_colliding;

    if (rs_mode == 0) {
        // ---- LEGACY (Checkpoint 3D) reseed: the coarse search's own +-5%-of-span stall kick.
        // Retained ONLY so the 3D.1 benchmark can measure the old policy against the new one in
        // the same binary. Measured recovery: 0/132 at B=2000. Not the default.
        for (int att = 1; att <= max_reseed && remaining > 0; ++att) {
            CUDA_OK(cudaMemcpy(g_ws.todo, v.data(), B, cudaMemcpyHostToDevice));
            rep.reseed_attempts += remaining;
            const int TPB = 256, nthr = B * N;
            hard_reseed_kernel<CT><<<(nthr + TPB - 1) / TPB, TPB>>>(d_x, g_ws.q0, g_ws.todo, B,
                                                                    seed + (uint64_t)att, att);
            hard_init_kernel<CT><<<B, 32>>>(d_x, w, g_ws.q0, g_ws.verdict, g_ws.todo, B, margin, 0);
            CUDA_OK(cudaPeekAtLastError());
            CUDA_OK(cudaMemcpy(v.data(), g_ws.verdict, B, cudaMemcpyDeviceToHost));
            int still = 0;
            for (int b = 0; b < B; ++b) if (v[b]) ++still;
            remaining = still;
        }
    } else {
        // ---- CHECKPOINT 3D.1: dedicated batched collision-free seed generator.
        // Per round: compact the still-colliding rows -> one generation kernel -> ONE batched
        // full-check over all F*R candidates -> one deterministic selection kernel -> re-verify
        // the selected replacements through the same Stage-3D init the free seeds went through.
        const int R = rs_cfg.candidates < 1 ? 1 : rs_cfg.candidates;
        std::vector<int> fail_h(B);
        std::vector<unsigned char> comp_h;
        cudaEvent_t a0, a1; CUDA_OK(cudaEventCreate(&a0)); CUDA_OK(cudaEventCreate(&a1));
        auto tick = [&](double* acc) {
            CUDA_OK(cudaEventRecord(a1)); CUDA_OK(cudaEventSynchronize(a1));
            float ms = 0.f; CUDA_OK(cudaEventElapsedTime(&ms, a0, a1)); *acc += ms;
            CUDA_OK(cudaEventRecord(a0));
        };

        for (int round = 0; round < rs_cfg.rounds && remaining > 0; ++round) {
            int F = 0;
            for (int b = 0; b < B; ++b) if (v[b]) fail_h[F++] = b;
            if (F == 0) break;
            ++rep.rounds_run;
            rep.reseed_attempts += F;
            rep.candidates_checked += F * R;

            g_ws.ensure_reseed(B * R, B);
            CUDA_OK(cudaMemcpy(g_ws.fail_idx, fail_h.data(), sizeof(int) * F, cudaMemcpyHostToDevice));
            CUDA_OK(cudaEventRecord(a0));

            reseed::fingerprint_kernel<<<(F + 127) / 128, 128>>>(g_ws.q0, g_ws.fail_idx,
                                                                 g_ws.fail_fp, F);
            const long long nthr = (long long)F * R * g1s::N_JOINTS;
            reseed::generate_kernel<<<(int)((nthr + 255) / 256), 256>>>(
                g_ws.q0, g_ws.fail_idx, g_ws.fail_fp, g_ws.cand_q, g_ws.cand_dist,
                g_ws.cand_comp, F, R, round, rs_cfg, seed);
            CUDA_OK(cudaPeekAtLastError());
            tick(&rep.gen_ms);

            reseed::check_kernel<<<(int)((long long)F * R), 32>>>(
                g_ws.cand_q, g_ws.cand_free, (long long)F * R, margin);
            CUDA_OK(cudaPeekAtLastError());
            tick(&rep.check_ms);

            reseed::select_kernel<CT><<<(F + 127) / 128, 128>>>(
                d_x, g_ws.q0, g_ws.fail_idx, g_ws.cand_q, g_ws.cand_free, g_ws.cand_dist,
                g_ws.sel, F, R);
            CUDA_OK(cudaPeekAtLastError());
            tick(&rep.select_ms);

            // Re-verify: a replacement enters coordinate search only after passing the SAME full
            // Stage-3D check every other seed passed. The candidate check above is not taken on
            // trust -- it also has to leave a consistent committed sidecar state behind.
            CUDA_OK(cudaMemcpy(g_ws.todo, v.data(), B, cudaMemcpyHostToDevice));
            hard_init_kernel<CT><<<B, 32>>>(d_x, w, g_ws.q0, g_ws.verdict, g_ws.todo, B, margin, 0);
            CUDA_OK(cudaPeekAtLastError());
            tick(&rep.verify_ms);

            // Which distribution component rescued each seed (diagnostics only).
            std::vector<int> sel_h(F);
            comp_h.assign((size_t)F * R, 0);
            CUDA_OK(cudaMemcpy(sel_h.data(), g_ws.sel, sizeof(int) * F, cudaMemcpyDeviceToHost));
            CUDA_OK(cudaMemcpy(comp_h.data(), g_ws.cand_comp, (size_t)F * R, cudaMemcpyDeviceToHost));
            for (int f = 0; f < F; ++f) {
                if (sel_h[f] < 0) continue;
                switch (comp_h[(size_t)f * R + sel_h[f]]) {
                    case reseed::COMP_PERTURB: ++rep.sel_perturb; break;
                    case reseed::COMP_NOMINAL: ++rep.sel_nominal; break;
                    default:                   ++rep.sel_broad;   break;
                }
            }
            CUDA_OK(cudaMemcpy(v.data(), g_ws.verdict, B, cudaMemcpyDeviceToHost));
            int still = 0;
            for (int b = 0; b < B; ++b) if (v[b]) ++still;
            remaining = still;
        }
        CUDA_OK(cudaEventDestroy(a0)); CUDA_OK(cudaEventDestroy(a1));
    }
    rep.failures  = remaining;
    rep.recovered = rep.initially_colliding - remaining;

    CUDA_OK(cudaEventRecord(t1)); CUDA_OK(cudaEventSynchronize(t1));
    { float ms = 0.f; CUDA_OK(cudaEventElapsedTime(&ms, t0, t1)); rep.ms = (double)ms; }
    CUDA_OK(cudaEventDestroy(t0)); CUDA_OK(cudaEventDestroy(t1));
    return rep;
}

}  // namespace hjcd_hard

// Test/inspection hooks for the persistent hard workspace (spec section 3: expose allocation
// counters and workspace dimensions).
extern "C" int hjcd_hard_ws_nalloc()  { return hjcd_hard::g_ws.n_alloc; }
extern "C" int hjcd_hard_ws_capacity(){ return hjcd_hard::g_ws.cap_B; }
extern "C" int hjcd_hard_reseed_ws_capacity(){ return hjcd_hard::g_ws.cap_FR; }
extern "C" int hjcd_hard_reseed_ws_nalloc(){ return hjcd_hard::g_ws.n_alloc_rs; }

// Copy the LAST reseed round's candidate arena out for tests: the joint-limit projection, the
// distribution mixture and the selection rule are only checkable if the candidates are observable.
extern "C" int hjcd_hard_reseed_dump(float* cand_q, unsigned char* cand_free,
                                     unsigned char* cand_comp, float* cand_dist,
                                     int* fail_idx, int* sel, int FR, int F) {
    auto& w = hjcd_hard::g_ws;
    if (w.cap_FR <= 0 || FR <= 0) return 0;
    const int n = FR < w.cap_FR ? FR : w.cap_FR;
    if (cand_q)    CUDA_OK(cudaMemcpy(cand_q, w.cand_q,
                                      sizeof(float)*(size_t)n*g1s::N_JOINTS, cudaMemcpyDeviceToHost));
    if (cand_free) CUDA_OK(cudaMemcpy(cand_free, w.cand_free, (size_t)n, cudaMemcpyDeviceToHost));
    if (cand_comp) CUDA_OK(cudaMemcpy(cand_comp, w.cand_comp, (size_t)n, cudaMemcpyDeviceToHost));
    if (cand_dist) CUDA_OK(cudaMemcpy(cand_dist, w.cand_dist,
                                      sizeof(float)*(size_t)n, cudaMemcpyDeviceToHost));
    if (fail_idx && F > 0) CUDA_OK(cudaMemcpy(fail_idx, w.fail_idx,
                                              sizeof(int)*(size_t)F, cudaMemcpyDeviceToHost));
    if (sel && F > 0) CUDA_OK(cudaMemcpy(sel, w.sel, sizeof(int)*(size_t)F, cudaMemcpyDeviceToHost));
    return n;
}
extern "C" int hjcd_hard_ctr_stride() { return g1sc::HARD_CTR_STRIDE; }
extern "C" int hjcd_hard_max_top_k()  { return g1sc::HARD_MAX_K; }
extern "C" void hjcd_hard_ws_release(){ hjcd_hard::g_ws.release(); }

// Copy the persistent committed state out for tests (spec section 10: the committed-state
// invariant is only testable if the committed state is observable). Returns the number of seeds
// copied; 0 when hard mode has never run. Any out-pointer may be null.
extern "C" int hjcd_hard_dump(float* qc, float* qfree, unsigned char* flags,
                              float* Tf, double* Td, int B) {
    auto& w = hjcd_hard::g_ws;
    if (w.cap_B <= 0 || B <= 0) return 0;
    const int n = B < w.cap_B ? B : w.cap_B;
    const size_t J = (size_t)g1s::N_JOINTS, L = (size_t)g1s::N_LINKS * 16;
    if (qc)    CUDA_OK(cudaMemcpy(qc,    w.qc,    sizeof(float)*(size_t)n*J, cudaMemcpyDeviceToHost));
    if (qfree) CUDA_OK(cudaMemcpy(qfree, w.qfree, sizeof(float)*(size_t)n*J, cudaMemcpyDeviceToHost));
    if (flags) CUDA_OK(cudaMemcpy(flags, w.flags, (size_t)n,                 cudaMemcpyDeviceToHost));
    if (Tf)    CUDA_OK(cudaMemcpy(Tf,    w.Tf,    sizeof(float)*(size_t)n*L, cudaMemcpyDeviceToHost));
    if (Td)    CUDA_OK(cudaMemcpy(Td,    w.Td,    sizeof(double)*(size_t)n*L,cudaMemcpyDeviceToHost));
    return n;
}

template<typename CT>
static CoarseOutputs launch_coarse_mt(
    const SolveInputs& in,
    int B, double eps_pos, double eps_ori, double lambda_coord, double h_min, double max_step,
    int max_iters, int stall_lim, int use_incremental, unsigned long long seed,
    const grid::robotModel<double>* d_robotModel, bool diagnostics,
    const void* cc_model, const void* cc_env_ptr, int max_pert_attempts,
    HjcdWorkspace* ws, void* out_q_ct)
{
    if (max_pert_attempts < 1) max_pert_attempts = 1;
    const int K = hjcd::NT;
    CoarseOutputs R;
    R.num_targets = K;
    R.q.assign((size_t)B * N, 0.0);
    R.pos_err.assign((size_t)B * K, 0.0);
    R.ori_err.assign((size_t)B * K, 0.0);
    R.cost.assign(B, 0.0);
    R.success.assign(B, 0);
    R.fp32 = std::is_same<CT, float>::value;
    R.trace_cols = CTRACE_COLS;
    R.trace_cap = diagnostics ? max_iters : 0;
    R.trace.assign((size_t)B * R.trace_cap * CTRACE_COLS, 0.0);
    if (!in.q || B <= 0 || !d_robotModel || !ws || !out_q_ct) return R;
    init_joint_limits_from_grid();

    const grid::robotModel<CT>* d_rm = robot_model_for<CT>(d_robotModel);
    std::vector<CT> stage;                       // reused per-call staging (only for a dtype mismatch)

    // Problem count P and seeds-per-problem S. Legacy callers leave num_problems == 0 -> P = B, S = 1
    // (every candidate its own problem). Targets/weights/mask are stored ONCE per problem.
    const int S = (in.seeds_per_problem >= 1) ? in.seeds_per_problem : 1;
    const int P = (in.num_problems >= 1) ? in.num_problems : B;

    ws->ensure(P, B, std::is_same<CT,float>::value ? 1 : 0, R.trace_cap);
    ws->rewind();
    CT* d_q  = ws->take<CT>((size_t)B*N);
    CT* d_tp = ws->take<CT>((size_t)P*K*3);
    CT* d_tq = ws->take<CT>((size_t)P*K*4);
    CT* d_wp = ws->take<CT>((size_t)P*K);
    CT* d_wo = ws->take<CT>((size_t)P*K);
    CT* d_pn = ws->take<CT>((size_t)B*K);
    CT* d_on = ws->take<CT>((size_t)B*K);
    CT* d_c  = ws->take<CT>((size_t)B);
    unsigned int*  d_act = ws->take<unsigned int>((size_t)P);
    unsigned char* d_s   = ws->take<unsigned char>((size_t)B);
    CT* d_tr = (R.trace_cap > 0) ? ws->take<CT>((size_t)B * R.trace_cap * CTRACE_COLS) : nullptr;

    upload_in<CT>(d_q,  in.q,     in.f32, (size_t)B*N,   stage);
    upload_in<CT>(d_tp, in.tgt_p, in.f32, (size_t)P*K*3, stage);
    upload_in<CT>(d_tq, in.tgt_q, in.f32, (size_t)P*K*4, stage);
    upload_in<CT>(d_wp, in.wp,    in.f32, (size_t)P*K,   stage);
    upload_in<CT>(d_wo, in.wo,    in.f32, (size_t)P*K,   stage);
    CUDA_OK(cudaMemcpy(d_act, in.active, sizeof(unsigned int)*P, cudaMemcpyHostToDevice));
    if (d_tr) CUDA_OK(cudaMemset(d_tr, 0, sizeof(CT) * (size_t)B * R.trace_cap * CTRACE_COLS));

    // Dynamic shared is reserved ENTIRELY for grid_collision::config_free's internal arena; the
    // coarse scratch is static shared. Zero when the collision gate is off.
    const int cc_enabled = (cc_model && cc_env_ptr) ? 1 : 0;
    size_t cc_smem = 0;
    // ---- Stage 3D/3E: hard self-collision mode -------------------------------------------------
    // The <CT,true> instantiation is launched ONLY here. Everything else keeps running <CT,false>,
    // which contains no sidecar code at all.
    const bool hard = (in.hard_self_collision != 0) && hjcd_hard_available();
    if (hard) {
        if (!hjcd_hard::bind_model_once())
            throw std::runtime_error("self_collision_mode='hard': sidecar model not uploaded "
                                     "(call hjcdik._ensure_self_collision_sidecar() first)");
        hjcd_hard::reseed::Config rs_cfg;
        rs_cfg.candidates = in.hard_reseed_candidates;
        rs_cfg.rounds = in.hard_reseed_rounds;
        rs_cfg.n_scales = in.hard_reseed_n_scales;
        for (int i = 0; i < hjcd_hard::reseed::MAX_SCALES; ++i)
            rs_cfg.scales[i] = in.hard_reseed_scales[i];
        const auto rep = hjcd_hard::prepare<CT>(d_q, B, in.hard_margin, in.hard_max_reseed,
                                                seed, in.hard_diagnostics != 0,
                                                in.hard_reseed_mode, rs_cfg);
        R.hard_ran = true;
        R.hard_initial_free = rep.initially_free;
        R.hard_initial_colliding = rep.initially_colliding;
        R.hard_reseed_attempts = rep.reseed_attempts;
        R.hard_recovered = rep.recovered;
        R.hard_seed_failures = rep.failures;
        R.hard_init_ms = rep.ms;
        R.hard_reseed_rounds_run = rep.rounds_run;
        R.hard_candidates_checked = rep.candidates_checked;
        R.hard_sel_perturb = rep.sel_perturb;
        R.hard_sel_nominal = rep.sel_nominal;
        R.hard_sel_broad = rep.sel_broad;
        R.hard_gen_ms = rep.gen_ms;
        R.hard_check_ms = rep.check_ms;
        R.hard_select_ms = rep.select_ms;
        R.hard_verify_ms = rep.verify_ms;

        g1sc::HardWorkspace hw = hjcd_hard::g_ws.view(in.hard_diagnostics != 0);
        cudaEvent_t h0, h1; CUDA_OK(cudaEventCreate(&h0)); CUDA_OK(cudaEventCreate(&h1));
        CUDA_OK(cudaEventRecord(h0));
        // Two distinct instantiations, selected once, per the resource contract above.
        auto* hard_kernel = (in.hard_oracle_every > 0)
                          ? coarse_search_mt_kernel<CT, true, true>
                          : coarse_search_mt_kernel<CT, true, false>;
        hard_kernel<<<B, 32, 0>>>(
            d_q, d_tp, d_tq, d_act, d_wp, d_wo, /*base_p=*/nullptr, /*base_q=*/nullptr,
            d_pn, d_on, d_c, d_s, d_tr, R.trace_cap,
            d_rm, (CT)eps_pos, (CT)eps_ori, (CT)lambda_coord, (CT)h_min, (CT)max_step,
            max_iters, stall_lim, B, S, use_incremental, seed,
            /*problem_seeds=*/nullptr,   // 5D.14c: standalone probe path; not the planner path
            max_pert_attempts,
            1, in.hard_top_k, in.hard_oracle_every, in.hard_margin, hw,
            /*cc_enabled=*/0
#if defined(HJCD_HAS_COLLISION)
            , reinterpret_cast<const grid::robotModel<float>*>(nullptr),
            grid_collision::Environment<float>{}
#endif
            );
        CUDA_OK(cudaEventRecord(h1));
        CUDA_OK(cudaPeekAtLastError());
        CUDA_OK(cudaDeviceSynchronize());
        { float ms = 0.f; CUDA_OK(cudaEventElapsedTime(&ms, h0, h1)); R.kernel_ms = (double)ms; }
        CUDA_OK(cudaEventDestroy(h0)); CUDA_OK(cudaEventDestroy(h1));

        // Per-seed hard-mode outputs the caller needs: the collision-free coarse fallback pose,
        // the seed-validity mask, and (when asked for) the diagnostic counters.
        R.hard_qfree.assign((size_t)B * g1s::N_JOINTS, 0.f);
        CUDA_OK(cudaMemcpy(R.hard_qfree.data(), hjcd_hard::g_ws.qfree,
                           sizeof(float) * (size_t)B * g1s::N_JOINTS, cudaMemcpyDeviceToHost));
        R.hard_flags.assign(B, 0);
        CUDA_OK(cudaMemcpy(R.hard_flags.data(), hjcd_hard::g_ws.flags, B, cudaMemcpyDeviceToHost));
        if (in.hard_diagnostics) {
            R.hard_ctr_stride = g1sc::HARD_CTR_STRIDE;
            R.hard_counters.assign((size_t)B * R.hard_ctr_stride, 0);
            CUDA_OK(cudaMemcpy(R.hard_counters.data(), hjcd_hard::g_ws.ctr,
                               sizeof(int) * R.hard_counters.size(), cudaMemcpyDeviceToHost));
        }
    }

    if (!hard) {
#if defined(HJCD_HAS_COLLISION)
    if (cc_enabled) {
        cc_smem = grid::MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<float>();
        CUDA_OK(cudaFuncSetAttribute(coarse_search_mt_kernel<CT, false, false>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cc_smem));
    }

    cudaEvent_t ev0, ev1;
    CUDA_OK(cudaEventCreate(&ev0)); CUDA_OK(cudaEventCreate(&ev1));
    CUDA_OK(cudaEventRecord(ev0));
    coarse_search_mt_kernel<CT, false, false><<<B, 32, cc_smem>>>(
        d_q, d_tp, d_tq, d_act, d_wp, d_wo, /*base_p=*/nullptr, /*base_q=*/nullptr,
        d_pn, d_on, d_c, d_s, d_tr, R.trace_cap,
        d_rm, (CT)eps_pos, (CT)eps_ori, (CT)lambda_coord, (CT)h_min, (CT)max_step,
        max_iters, stall_lim, B, S, use_incremental, seed,
        /*problem_seeds=*/nullptr,   // 5D.14c: standalone probe path; not the planner path
        max_pert_attempts,
        0 /*hard_enabled*/, 1 /*hard_top_k*/, 0 /*oracle*/, 0.0f /*hard_margin*/,
        g1sc::HardWorkspace{},
        cc_enabled,
        reinterpret_cast<const grid::robotModel<float>*>(cc_model),
        cc_env_ptr ? *reinterpret_cast<const grid_collision::Environment<float>*>(cc_env_ptr)
                   : grid_collision::Environment<float>{});
#else
    (void)cc_smem;
    cudaEvent_t ev0, ev1;
    CUDA_OK(cudaEventCreate(&ev0)); CUDA_OK(cudaEventCreate(&ev1));
    CUDA_OK(cudaEventRecord(ev0));
    coarse_search_mt_kernel<CT, false, false><<<B, 32, 0>>>(
        d_q, d_tp, d_tq, d_act, d_wp, d_wo, /*base_p=*/nullptr, /*base_q=*/nullptr,
        d_pn, d_on, d_c, d_s, d_tr, R.trace_cap,
        d_rm, (CT)eps_pos, (CT)eps_ori, (CT)lambda_coord, (CT)h_min, (CT)max_step,
        max_iters, stall_lim, B, S, use_incremental, seed,
        /*problem_seeds=*/nullptr,   // 5D.14c: standalone probe path; not the planner path
        max_pert_attempts,
        0 /*hard_enabled*/, 1 /*hard_top_k*/, 0 /*oracle*/, 0.0f /*hard_margin*/,
        g1sc::HardWorkspace{},
        cc_enabled);
#endif
    CUDA_OK(cudaEventRecord(ev1));
    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());
    { float ms = 0.f; CUDA_OK(cudaEventElapsedTime(&ms, ev0, ev1)); R.kernel_ms = (double)ms; }
    CUDA_OK(cudaEventDestroy(ev0)); CUDA_OK(cudaEventDestroy(ev1));
    }   // end !hard legacy launch

    // The B x N configuration goes STRAIGHT from the device into the caller's numpy buffer -- one
    // pass. It used to make four (D2H -> widen to double -> narrow back to float -> pybind copy).
    CUDA_OK(cudaMemcpy(out_q_ct, d_q, sizeof(CT)*(size_t)B*N, cudaMemcpyDeviceToHost));
    download_widen<CT>(d_pn, R.pos_err.data(), (size_t)B * K, stage);
    download_widen<CT>(d_on, R.ori_err.data(), (size_t)B * K, stage);
    download_widen<CT>(d_c,  R.cost.data(),    (size_t)B,     stage);
    CUDA_OK(cudaMemcpy(R.success.data(), d_s, sizeof(unsigned char)*B, cudaMemcpyDeviceToHost));

    if (d_tr) {
        download_widen<CT>(d_tr, R.trace.data(), R.trace.size(), stage);
        // Every public coarse counter is DERIVED FROM THE TRACE (Phase-3C rule). Row validity is
        // column 0 -- explicit, never inferred.
        R.iterations.assign(B, 0); R.accepted.assign(B, 0); R.rejected.assign(B, 0);
        R.perturbations.assign(B, 0); R.max_stall.assign(B, 0);
        R.pert_events.assign(B, 0); R.pert_attempts.assign(B, 0);
        R.pert_rejected.assign(B, 0); R.pert_exhausted.assign(B, 0);
        for (int b = 0; b < B; ++b) {
            int n = 0, acc = 0, pert = 0, mstall = 0;
            int pev = 0, patt = 0, prej = 0, pexh = 0;
            for (int i = 0; i < R.trace_cap; ++i) {
                const double* row = &R.trace[((size_t)b * R.trace_cap + i) * CTRACE_COLS];
                if (row[0] == 0.0) continue;
                ++n;
                acc  += (int)llrint(row[7]);
                pert += (int)llrint(row[9]);
                mstall = std::max(mstall, (int)llrint(row[8]));
                const int att = (int)llrint(row[10]);
                patt += att;
                if (att > 0) ++pev;                  // the kick fired this iteration
                prej += (int)llrint(row[11]);
                pexh += (int)llrint(row[12]);
            }
            R.iterations[b] = n;
            R.accepted[b] = acc;
            R.rejected[b] = n - acc;
            R.perturbations[b] = pert;
            R.max_stall[b] = mstall;
            R.pert_events[b] = pev;
            R.pert_attempts[b] = patt;
            R.pert_rejected[b] = prej;
            R.pert_exhausted[b] = pexh;
        }
    }
    return R;                    // every buffer above lives in the workspace: nothing to free
}

// Public entry point. precision: 0 = float64 (default), 1 = float32. Validated above this layer.
CoarseOutputs compute_coarse_search(
    const SolveInputs& in,
    int B, double eps_pos, double eps_ori, double lambda_coord, double h_min, double max_step,
    int max_iters, int stall_lim, int use_incremental, unsigned long long seed,
    const grid::robotModel<double>* d_robotModel, bool diagnostics,
    const void* cc_model, const void* cc_env_ptr, int max_pert_attempts, int precision,
    HjcdWorkspace* ws, void* out_q_ct)
{
    if (precision == 1)
        return launch_coarse_mt<float>(in, B, eps_pos, eps_ori, lambda_coord, h_min, max_step,
                                       max_iters, stall_lim, use_incremental, seed,
                                       d_robotModel, diagnostics, cc_model, cc_env_ptr,
                                       max_pert_attempts, ws, out_q_ct);
    return launch_coarse_mt<double>(in, B, eps_pos, eps_ori, lambda_coord, h_min, max_step,
                                    max_iters, stall_lim, use_incremental, seed,
                                    d_robotModel, diagnostics, cc_model, cc_env_ptr,
                                    max_pert_attempts, ws, out_q_ct);
}

template<typename CT>
static LMRefineOutputs launch_lm_mt(
    const SolveInputs& in,
    int B, double eps_pos, double eps_ori, double lambda_init, int max_iters,
    const grid::robotModel<double>* d_robotModel, bool diagnostics,
    int stag_patience, double stag_rel,
    HjcdWorkspace* ws, void* out_q_ct)
{
    const int K = hjcd::NT;
    LMRefineOutputs R;
    R.num_targets = K;
    R.q.assign((size_t)B * N, 0.0);
    R.pos_err.assign((size_t)B * K, 0.0);
    R.ori_err.assign((size_t)B * K, 0.0);
    R.cost.assign(B, 0.0);
    R.success.assign(B, 0);
    R.fp32 = std::is_same<CT, float>::value;
    R.trace_cols = TRACE_COLS;
    R.trace_cap = diagnostics ? max_iters : 0;
    R.trace.assign((size_t)B * R.trace_cap * TRACE_COLS, 0.0);
    if (!in.q || B <= 0 || !d_robotModel || !ws || !out_q_ct) return R;
    init_joint_limits_from_grid();

    const grid::robotModel<CT>* d_rm = robot_model_for<CT>(d_robotModel);
    std::vector<CT> stage;

    const int S = (in.seeds_per_problem >= 1) ? in.seeds_per_problem : 1;
    const int P = (in.num_problems >= 1) ? in.num_problems : B;

    ws->ensure(P, B, std::is_same<CT,float>::value ? 1 : 0, R.trace_cap);
    ws->rewind();
    CT* d_q  = ws->take<CT>((size_t)B*N);
    CT* d_tp = ws->take<CT>((size_t)P*K*3);
    CT* d_tq = ws->take<CT>((size_t)P*K*4);
    CT* d_wp = ws->take<CT>((size_t)P*K);
    CT* d_wo = ws->take<CT>((size_t)P*K);
    CT* d_pn = ws->take<CT>((size_t)B*K);
    CT* d_on = ws->take<CT>((size_t)B*K);
    CT* d_c  = ws->take<CT>((size_t)B);
    unsigned int*  d_act = ws->take<unsigned int>((size_t)P);
    unsigned char* d_s   = ws->take<unsigned char>((size_t)B);
    upload_in<CT>(d_q,  in.q,     in.f32, (size_t)B*N,   stage);
    upload_in<CT>(d_tp, in.tgt_p, in.f32, (size_t)P*K*3, stage);
    upload_in<CT>(d_tq, in.tgt_q, in.f32, (size_t)P*K*4, stage);
    upload_in<CT>(d_wp, in.wp,    in.f32, (size_t)P*K,   stage);
    upload_in<CT>(d_wo, in.wo,    in.f32, (size_t)P*K,   stage);
    CUDA_OK(cudaMemcpy(d_act, in.active, sizeof(unsigned int)*P, cudaMemcpyHostToDevice));

    const int W = 1;                       // one warp per candidate (measured fastest; see CLAUDE.md)
    const size_t smem = (size_t)W * sizeof(LMScratch<CT>);
    if (smem > 48u * 1024u)
        CUDA_OK(cudaFuncSetAttribute(lm_multi_target_kernel<CT>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    CT* d_tr = (R.trace_cap > 0) ? ws->take<CT>((size_t)B * R.trace_cap * TRACE_COLS) : nullptr;
    if (d_tr) CUDA_OK(cudaMemset(d_tr, 0, sizeof(CT) * (size_t)B * R.trace_cap * TRACE_COLS));
    // CUDA-event device time for THIS launch. Reported alongside the end-to-end time of the SAME
    // invocation, so the two are always commensurable (see the Phase-0C timing note).
    cudaEvent_t ev0, ev1;
    CUDA_OK(cudaEventCreate(&ev0)); CUDA_OK(cudaEventCreate(&ev1));
    CUDA_OK(cudaEventRecord(ev0));
    lm_multi_target_kernel<CT><<<(B + W - 1) / W, 32 * W, smem>>>(
        d_q, d_tp, d_tq, d_act, d_wp, d_wo, /*base_p=*/nullptr, /*base_q=*/nullptr,
        /*out_base_diag=*/nullptr,
        d_pn, d_on, d_c, d_s, /*out_pose=*/nullptr,
        d_tr, R.trace_cap,
        d_rm, (CT)eps_pos, (CT)eps_ori, (CT)lambda_init, max_iters, B, /*stop_on_first=*/0,
        stag_patience, (CT)stag_rel, S, BaseUpdateCfg<CT>{});   // fixed base: update disabled
    CUDA_OK(cudaEventRecord(ev1));
    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());
    { float ms = 0.f; CUDA_OK(cudaEventElapsedTime(&ms, ev0, ev1)); R.kernel_ms = (double)ms; }
    CUDA_OK(cudaEventDestroy(ev0)); CUDA_OK(cudaEventDestroy(ev1));

    CUDA_OK(cudaMemcpy(out_q_ct, d_q, sizeof(CT)*(size_t)B*N, cudaMemcpyDeviceToHost));
    download_widen<CT>(d_pn, R.pos_err.data(), (size_t)B * K, stage);
    download_widen<CT>(d_on, R.ori_err.data(), (size_t)B * K, stage);
    download_widen<CT>(d_c,  R.cost.data(),    (size_t)B,     stage);
    CUDA_OK(cudaMemcpy(R.success.data(), d_s, sizeof(unsigned char)*B, cudaMemcpyDeviceToHost));
    if (d_tr) {
        download_widen<CT>(d_tr, R.trace.data(), R.trace.size(), stage);

        // Derive every public LM diagnostic from the trace -- the authoritative source.
        //   lm_iterations   = number of rows whose explicit VALID flag is set
        //   lm_trials       = cumulative trial count on the last valid row
        //   accepted_steps  = cumulative accepted count on the last valid row
        //   rejected_steps  = lm_iterations - accepted_steps
        //   line_searches   = cumulative line-search count on the last valid row
        // A problem that converged on entry wrote no rows: everything is 0, by convention.
        R.lm_iterations.assign(B, 0);
        R.lm_trials.assign(B, 0);
        R.line_searches.assign(B, 0);
        R.accepted_steps.assign(B, 0);
        R.rejected_steps.assign(B, 0);
        for (int b = 0; b < B; ++b) {
            int n = 0, last = -1;
            for (int i = 0; i < R.trace_cap; ++i) {
                const double* row = &R.trace[((size_t)b * R.trace_cap + i) * TRACE_COLS];
                if (row[0] != 0.0) { ++n; last = i; }
            }
            R.lm_iterations[b] = n;
            if (last >= 0) {
                const double* row = &R.trace[((size_t)b * R.trace_cap + last) * TRACE_COLS];
                R.lm_trials[b]      = (int)llrint(row[2]);
                R.accepted_steps[b] = (int)llrint(row[4]);
                R.line_searches[b]  = (int)llrint(row[9]);
                R.rejected_steps[b] = n - R.accepted_steps[b];
            }
        }
    }
    return R;                    // workspace-owned buffers: nothing to free
}

// Public entry point. precision: 0 = float64 (default), 1 = float32.
LMRefineOutputs compute_lm_refine(
    const SolveInputs& in,
    int B, double eps_pos, double eps_ori, double lambda_init, int max_iters,
    const grid::robotModel<double>* d_robotModel, bool diagnostics, int precision,
    int stag_patience, double stag_rel, HjcdWorkspace* ws, void* out_q_ct)
{
    if (precision == 1)
        return launch_lm_mt<float>(in, B, eps_pos, eps_ori, lambda_init, max_iters,
                                   d_robotModel, diagnostics, stag_patience, stag_rel,
                                   ws, out_q_ct);
    return launch_lm_mt<double>(in, B, eps_pos, eps_ori, lambda_init, max_iters,
                                d_robotModel, diagnostics, stag_patience, stag_rel,
                                ws, out_q_ct);
}

// =============================================================================================
// MILESTONE 3: per-problem segmented top-1 selection, on device.
//
// After all B = P*S candidates finish (and, when collision is on, after the candidate-local coarse
// fallback), one block per problem scans its S candidates and picks the single winner by the
// deterministic three-class lexicographic key
//     R = (class, E_phys, seed)   lower is better
//     class 0 solved  <  class 1 valid-unsolved  <  class 2 invalid
// E_phys = sum_{k in active} [ |e_p|^2/eps_p^2 + |e_R|^2/eps_R^2 ] is the STABLE cross-candidate
// metric. The row-scaled LM cost is NEVER used for selection (carried only as cost_lm).
//
// Per-problem summaries (num_solved / num_valid / problem_success and, with collision, the collision
// counts) are accumulated INSIDE this scan -- no second pass over the candidates.
// =============================================================================================

// LM input = coarse output for coarse-dispatched candidates, raw seed otherwise (mixed masks).
template<typename CT>
__global__ void pick_lm_seed_kernel(CT* __restrict__ lm_in, const CT* __restrict__ coarse_q,
                                    const CT* __restrict__ seeds, const unsigned char* __restrict__ use_coarse,
                                    size_t BN, int N_) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= BN) return;
    const int b = (int)(i / N_);
    lm_in[i] = use_coarse[b] ? coarse_q[i] : seeds[i];
}

#if defined(HJCD_HAS_COLLISION)
// Batched exact collision check reading a CT (float/double) config buffer. valid[b] = config_free.
template<typename CT>
__global__ void mark_collisions_ct(const CT* __restrict__ q_in, int Bn,
                                   unsigned char* __restrict__ valid,
                                   const grid::robotModel<float>* d_robotModel,
                                   grid_collision::Environment<float> env) {
    namespace gc = grid_collision;
    constexpr int NS = gc::NUM_COLLISION_SPHERES;
    const int b = (int)blockIdx.x;
    if (b >= Bn) return;
    __shared__ float s_q[N];
    __shared__ float s_pos[3 * NS];
    __shared__ float s_r[NS];
    for (int j = threadIdx.x; j < N; j += blockDim.x) s_q[j] = (float)q_in[(size_t)b*N + j];
    __syncthreads();
    const bool ok = gc::config_free<float>(s_q, d_robotModel, env, s_pos, s_r, nullptr);
    if (threadIdx.x == 0) valid[b] = ok ? 1 : 0;
}
#endif

// A candidate's ranking key. class_id: 0 solved, 1 valid-unsolved, 2 invalid.
struct Cand {
    int class_id;
    double ephys;
    int seed;        // seed index within the problem (0..S-1); the tie-break
};
__device__ __forceinline__ bool cand_better(const Cand& a, const Cand& b) {
    if (a.class_id != b.class_id) return a.class_id < b.class_id;
    if (a.ephys != b.ephys)       return a.ephys < b.ephys;      // both finite here (invalid=+inf)
    return a.seed < b.seed;
}

// Copy lm[b] <- coarse[b] where the candidate needs the collision fallback, and set the final
// per-candidate collision/fallback flags. Open-world callers skip this (final_free = 1, fb = 0).
template<typename CT>
__global__ void apply_fallback_kernel(
    CT* __restrict__ lm_q, CT* __restrict__ lm_pe, CT* __restrict__ lm_oe,
    CT* __restrict__ lm_cost, unsigned char* __restrict__ lm_succ,
    const CT* __restrict__ coarse_q, const CT* __restrict__ coarse_pe, const CT* __restrict__ coarse_oe,
    const CT* __restrict__ coarse_cost, const unsigned char* __restrict__ coarse_succ,
    const unsigned char* __restrict__ lm_free, const unsigned char* __restrict__ coarse_free,
    unsigned char* __restrict__ final_free, unsigned char* __restrict__ fallback,
    int B, int K)
{
    const int b = blockIdx.x;
    if (b >= B) return;
    const bool lm_ok = lm_free[b] != 0;
    const bool co_ok = coarse_free[b] != 0;
    const bool fb = (!lm_ok) && co_ok;                 // LM collided, coarse is a feasible fallback
    if (threadIdx.x == 0) {
        fallback[b] = fb ? 1 : 0;
        final_free[b] = (lm_ok || fb) ? 1 : 0;         // 0 => infeasible (invalid)
    }
    if (fb) {
        for (int j = threadIdx.x; j < N; j += blockDim.x) lm_q[(size_t)b*N + j] = coarse_q[(size_t)b*N + j];
        for (int k = threadIdx.x; k < K; k += blockDim.x) {
            lm_pe[(size_t)b*K + k] = coarse_pe[(size_t)b*K + k];
            lm_oe[(size_t)b*K + k] = coarse_oe[(size_t)b*K + k];
        }
        if (threadIdx.x == 0) { lm_cost[b] = coarse_cost[b]; lm_succ[b] = coarse_succ[b]; }
    }
}

// One block per problem. threadIdx scans candidates s in [0,S) with a grid-stride, keeps the best
// under cand_better, then a shared-memory tree reduction picks the block winner. Counts are summed
// the same way. Thread 0 gathers the winner's state into the [P,...] selected buffers.
template<typename CT>
__global__ void segmented_top1_kernel(
    const CT* __restrict__ q,          // B x N   (post-fallback config)
    const CT* __restrict__ pe,         // B x K
    const CT* __restrict__ oe,         // B x K
    const CT* __restrict__ cost_lm,    // B
    const unsigned char* __restrict__ succ,       // B  (kernel's own tolerance success flag)
    const unsigned char* __restrict__ final_free, // B  (1 = collision-free / feasible)
    const unsigned char* __restrict__ fallback,   // B  (1 = used coarse fallback), may be null
    const unsigned char* __restrict__ lm_free,    // B  (1 = LM output was free), may be null
    const CT* __restrict__ seeds,      // B x N   (for the all-invalid fill)
    const unsigned int* __restrict__ active,      // P
    const CT eps_pos, const CT eps_ori, const int S, const int P, const int cc_enabled,
    // outputs, all [P] or [P,N]/[P,K]
    CT* __restrict__ sel_q, CT* __restrict__ sel_pe, CT* __restrict__ sel_oe,
    CT* __restrict__ sel_cost, double* __restrict__ sel_ephys, int* __restrict__ sel_seed,
    unsigned char* __restrict__ sel_succ, unsigned char* __restrict__ sel_valid,
    unsigned char* __restrict__ sel_cfree, unsigned char* __restrict__ sel_fb,
    int* __restrict__ num_solved, int* __restrict__ num_valid, unsigned char* __restrict__ prob_success,
    int* __restrict__ num_cfree, int* __restrict__ num_lm_coll, int* __restrict__ num_fb,
    int* __restrict__ num_infeas)
{
    const int p = blockIdx.x;
    if (p >= P) return;
    constexpr int K = hjcd::NT;
    const unsigned int mask = active[p];
    const int tid = threadIdx.x, nt = blockDim.x;

    extern __shared__ unsigned char s_raw[];
    Cand* s_best = reinterpret_cast<Cand*>(s_raw);
    int*  s_cnt  = reinterpret_cast<int*>(s_best + nt);   // 6 ints per thread: solved,valid,cfree,lmcoll,fb,infeas

    Cand best{2, INFINITY, 1 << 30};
    int c_solved=0, c_valid=0, c_cfree=0, c_lmcoll=0, c_fb=0, c_infeas=0;

    for (int s = tid; s < S; s += nt) {
        const int b = p * S + s;
        // finiteness of the config
        bool finite = true;
        #pragma unroll 1
        for (int j = 0; j < N; ++j) { const CT v = q[(size_t)b*N + j]; if (!isfinite((double)v)) { finite = false; break; } }

        // E_phys over ACTIVE targets, in double (matches the host reference exactly).
        double ephys = 0.0;
        bool within = true;
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            if (!((mask >> k) & 1u)) continue;
            const double ep = (double)pe[(size_t)b*K + k], eo = (double)oe[(size_t)b*K + k];
            if (!isfinite(ep) || !isfinite(eo)) { finite = false; continue; }
            const double rp = ep / (double)eps_pos, ro = eo / (double)eps_ori;
            ephys += rp*rp + ro*ro;
            if (ep > (double)eps_pos || eo > (double)eps_ori) within = false;
        }
        const bool feasible = cc_enabled ? (final_free[b] != 0) : true;

        int cls;
        if (!finite || !feasible) { cls = 2; ephys = INFINITY; }
        else if (within)         { cls = 0; }
        else                     { cls = 1; }

        // counts
        if (cls == 0) ++c_solved;
        if (cls <= 1) ++c_valid;
        if (cc_enabled) {
            if (feasible)          ++c_cfree;
            if (lm_free && lm_free[b] == 0) ++c_lmcoll;
            if (fallback && fallback[b])    ++c_fb;
            if (!feasible)         ++c_infeas;
        }

        Cand cur{cls, ephys, s};
        if (cand_better(cur, best)) best = cur;
    }

    s_best[tid] = best;
    s_cnt[tid*6+0]=c_solved; s_cnt[tid*6+1]=c_valid; s_cnt[tid*6+2]=c_cfree;
    s_cnt[tid*6+3]=c_lmcoll; s_cnt[tid*6+4]=c_fb;    s_cnt[tid*6+5]=c_infeas;
    __syncthreads();

    for (int stride = nt >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (cand_better(s_best[tid+stride], s_best[tid])) s_best[tid] = s_best[tid+stride];
            #pragma unroll
            for (int q6 = 0; q6 < 6; ++q6) s_cnt[tid*6+q6] += s_cnt[(tid+stride)*6+q6];
        }
        __syncthreads();
    }

    if (tid != 0) return;
    const Cand win = s_best[0];
    const int win_s = win.seed, win_b = p * S + win_s;
    const bool invalid = (win.class_id == 2);

    num_solved[p] = s_cnt[0]; num_valid[p] = s_cnt[1];
    prob_success[p] = (s_cnt[0] > 0) ? 1 : 0;
    if (cc_enabled) {
        num_cfree[p] = s_cnt[2]; num_lm_coll[p] = s_cnt[3];
        num_fb[p] = s_cnt[4]; num_infeas[p] = s_cnt[5];
    }

    if (!invalid) {
        for (int j = 0; j < N; ++j) sel_q[(size_t)p*N + j] = q[(size_t)win_b*N + j];
        for (int k = 0; k < K; ++k) { sel_pe[(size_t)p*K+k] = pe[(size_t)win_b*K+k];
                                      sel_oe[(size_t)p*K+k] = oe[(size_t)win_b*K+k]; }
        sel_cost[p]  = cost_lm[win_b];
        sel_ephys[p] = win.ephys;
        sel_seed[p]  = win_s;
        sel_succ[p]  = (win.class_id == 0) ? 1 : 0;
        sel_valid[p] = 1;
        sel_cfree[p] = cc_enabled ? final_free[win_b] : 1;
        sel_fb[p]    = (cc_enabled && fallback) ? fallback[win_b] : 0;
    } else {
        // Every candidate invalid. Deterministic fill: the problem's FIRST seed if finite, else zeros.
        bool s0_finite = true;
        for (int j = 0; j < N; ++j) if (!isfinite((double)seeds[(size_t)(p*S)*N + j])) { s0_finite = false; break; }
        for (int j = 0; j < N; ++j)
            sel_q[(size_t)p*N + j] = s0_finite ? seeds[(size_t)(p*S)*N + j] : (CT)0;
        for (int k = 0; k < K; ++k) { sel_pe[(size_t)p*K+k] = (CT)INFINITY; sel_oe[(size_t)p*K+k] = (CT)INFINITY; }
        sel_cost[p]  = (CT)INFINITY;
        sel_ephys[p] = INFINITY;
        sel_seed[p]  = -1;                 // documented: -1 == no valid candidate
        sel_succ[p]  = 0;
        sel_valid[p] = 0;
        sel_cfree[p] = 0;                  // never marked feasible
        sel_fb[p]    = 0;
    }
}

// Deterministic segmented top-M. One block per problem; M rounds of masked argmin. Round m finds the
// m-th best candidate under (class, E_phys, seed), skipping the m already-selected seeds, so the M
// winners are DISTINCT candidate IDs. When a problem has fewer than M valid candidates, the leftover
// slots are INVALID PADS (seed=-1, cost=+inf, valid=False) -- never duplicates of a real candidate.
// M == 1 reproduces segmented_top1_kernel exactly. Per-problem summaries are counted in round 0.
template<typename CT>
__global__ void segmented_topM_kernel(
    const CT* __restrict__ q, const CT* __restrict__ pe, const CT* __restrict__ oe,
    const CT* __restrict__ cost_lm, const unsigned char* __restrict__ succ,
    const unsigned char* __restrict__ final_free, const unsigned char* __restrict__ fallback,
    const unsigned char* __restrict__ lm_free, const CT* __restrict__ seeds,
    const unsigned int* __restrict__ active,
    const CT eps_pos, const CT eps_ori, const int S, const int P, const int M, const int cc_enabled,
    CT* __restrict__ sel_q, CT* __restrict__ sel_pe, CT* __restrict__ sel_oe,
    CT* __restrict__ sel_cost, double* __restrict__ sel_ephys, int* __restrict__ sel_seed,
    unsigned char* __restrict__ sel_succ, unsigned char* __restrict__ sel_valid,
    unsigned char* __restrict__ sel_cfree, unsigned char* __restrict__ sel_fb,
    int* __restrict__ num_solved, int* __restrict__ num_valid, unsigned char* __restrict__ prob_success,
    int* __restrict__ num_cfree, int* __restrict__ num_lm_coll, int* __restrict__ num_fb,
    int* __restrict__ num_infeas)
{
    const int p = blockIdx.x;
    if (p >= P) return;
    constexpr int K = hjcd::NT;
    const unsigned int mask = active[p];
    const int tid = threadIdx.x, nt = blockDim.x;

    extern __shared__ unsigned char s_raw[];
    Cand* s_best = reinterpret_cast<Cand*>(s_raw);
    int*  s_cnt  = reinterpret_cast<int*>(s_best + nt);   // 6 ints/thread (round 0)
    int*  s_taken = s_cnt + nt * 6;                       // M selected seed indices (-1 = pad)

    for (int m = 0; m < M; ++m) {
        Cand best{2, INFINITY, 1 << 30};
        int c0=0,c1=0,c2=0,c3=0,c4=0,c5=0;
        for (int s = tid; s < S; s += nt) {
            const int b = p * S + s;
            bool taken = false;
            for (int t = 0; t < m; ++t) if (s_taken[t] == s) { taken = true; break; }

            bool finite = true;
            #pragma unroll 1
            for (int j = 0; j < N; ++j) { if (!isfinite((double)q[(size_t)b*N+j])) { finite=false; break; } }
            double ephys = 0.0; bool within = true;
            #pragma unroll
            for (int k = 0; k < K; ++k) {
                if (!((mask >> k) & 1u)) continue;
                const double ep=(double)pe[(size_t)b*K+k], eo=(double)oe[(size_t)b*K+k];
                if (!isfinite(ep) || !isfinite(eo)) { finite=false; continue; }
                const double rp=ep/(double)eps_pos, ro=eo/(double)eps_ori; ephys += rp*rp+ro*ro;
                if (ep>(double)eps_pos || eo>(double)eps_ori) within=false;
            }
            const bool feas = cc_enabled ? (final_free[b]!=0) : true;
            int cls; if (!finite||!feas){cls=2;ephys=INFINITY;} else if(within)cls=0; else cls=1;
            if (m == 0) {
                if (cls==0)++c0; if (cls<=1)++c1;
                if (cc_enabled){ if(feas)++c2; if(lm_free&&lm_free[b]==0)++c3;
                                 if(fallback&&fallback[b])++c4; if(!feas)++c5; }
            }
            if (taken) continue;
            Cand cur{cls, ephys, s};
            if (cand_better(cur, best)) best = cur;
        }
        s_best[tid] = best;
        if (m == 0) { s_cnt[tid*6+0]=c0; s_cnt[tid*6+1]=c1; s_cnt[tid*6+2]=c2;
                      s_cnt[tid*6+3]=c3; s_cnt[tid*6+4]=c4; s_cnt[tid*6+5]=c5; }
        __syncthreads();
        for (int stride = nt >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                if (cand_better(s_best[tid+stride], s_best[tid])) s_best[tid] = s_best[tid+stride];
                if (m == 0) {
                    #pragma unroll
                    for (int q6=0;q6<6;++q6) s_cnt[tid*6+q6] += s_cnt[(tid+stride)*6+q6];
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            const Cand win = s_best[0];
            const bool invalid = (win.class_id == 2);
            if (m == 0) {
                num_solved[p]=s_cnt[0]; num_valid[p]=s_cnt[1]; prob_success[p]=(s_cnt[0]>0)?1:0;
                if (cc_enabled){ num_cfree[p]=s_cnt[2]; num_lm_coll[p]=s_cnt[3];
                                 num_fb[p]=s_cnt[4]; num_infeas[p]=s_cnt[5]; }
            }
            s_taken[m] = invalid ? -1 : win.seed;
            const size_t slot = (size_t)p*M + m, wb = (size_t)(p*S + win.seed);
            if (!invalid) {
                for (int j=0;j<N;++j) sel_q[slot*N+j] = q[wb*N+j];
                for (int k=0;k<K;++k){ sel_pe[slot*K+k]=pe[wb*K+k]; sel_oe[slot*K+k]=oe[wb*K+k]; }
                sel_cost[slot]=cost_lm[wb]; sel_ephys[slot]=win.ephys; sel_seed[slot]=win.seed;
                sel_succ[slot]=(win.class_id==0)?1:0; sel_valid[slot]=1;
                sel_cfree[slot]=cc_enabled?final_free[wb]:1;
                sel_fb[slot]=(cc_enabled&&fallback)?fallback[wb]:0;
            } else {
                bool s0f=true; for(int j=0;j<N;++j) if(!isfinite((double)seeds[(size_t)(p*S)*N+j])){s0f=false;break;}
                for(int j=0;j<N;++j) sel_q[slot*N+j] = s0f ? seeds[(size_t)(p*S)*N+j] : (CT)0;
                for(int k=0;k<K;++k){ sel_pe[slot*K+k]=(CT)INFINITY; sel_oe[slot*K+k]=(CT)INFINITY; }
                sel_cost[slot]=(CT)INFINITY; sel_ephys[slot]=INFINITY; sel_seed[slot]=-1;
                sel_succ[slot]=0; sel_valid[slot]=0; sel_cfree[slot]=0; sel_fb[slot]=0;
            }
        }
        __syncthreads();
    }
}

static int seg_block_size(int S) {
    if (S <= 64)  return 64;
    if (S <= 128) return 128;
    return 256;                 // production default for larger S (benchmarked in the M3 report)
}

template<typename CT>
static SolveProblemsOutputs launch_solve_problems(
    const SolveInputs& in, int B, int num_solutions,
    double eps_pos, double eps_ori,
    double lambda_coord, double h_min, double max_step, int coarse_iters, int coarse_stall_lim,
    int use_incremental, unsigned long long seed, int max_pert_attempts,
    double lambda_init, int lm_iters, int stag_patience, double stag_rel,
    const grid::robotModel<double>* d_robotModel, const unsigned char* use_coarse_host, bool run_coarse,
    const void* cc_model, const void* cc_env_ptr,
    bool return_all, HjcdWorkspace* ws, void* out_sel_q_ct, void* out_all_q_ct)
{
    constexpr int K = hjcd::NT;
    const int S = in.seeds_per_problem, P = in.num_problems;
    const int M = (num_solutions >= 1) ? num_solutions : 1;   // top-M per problem
    SolveProblemsOutputs R;
    R.P = P; R.S = S; R.K = K; R.M = M; R.fp32 = std::is_same<CT,float>::value;
    R.cc_enabled = (cc_model && cc_env_ptr);
    if (B <= 0 || P <= 0 || S <= 0 || !d_robotModel || !ws) return R;
    init_joint_limits_from_grid();
    const grid::robotModel<CT>* d_rm = robot_model_for<CT>(d_robotModel);
    std::vector<CT> stage;

    auto al = [](size_t n){ return (n + 255) & ~size_t(255); };
    const size_t ct = sizeof(CT);
    const size_t bn=(size_t)B*N, bk=(size_t)B*K, pk=(size_t)P*K;
    const size_t pm=(size_t)P*M, pmn=pm*N, pmk=pm*K;      // selected buffers scale with M
    size_t need = 0;
    for (size_t s : {al(bn*ct), al(pk*3*ct), al(pk*4*ct), al(pk*ct), al(pk*ct), al((size_t)P*4),
                     al((size_t)B),                                        // use_coarse
                     al(bn*ct), al(bk*ct), al(bk*ct), al((size_t)B*ct), al((size_t)B),   // coarse
                     al(bn*ct), al(bk*ct), al(bk*ct), al((size_t)B*ct), al((size_t)B),   // lm
                     al((size_t)B)*4,                                      // lm_free,coarse_free,final_free,fb
                     al(pmn*ct), al(pmk*ct), al(pmk*ct), al(pm*ct), al(pm*8), al(pm*4), // sel [P,M,..]
                     al(pm)*4,                                             // sel succ/valid/cfree/fb
                     al((size_t)P*4)*6, al((size_t)P)})                    // summaries [P]
        need += s;
    // Floating base [B,3] + [B,4]. MUST be accounted here: take<>() returns nullptr on an
    // exhausted arena rather than throwing, so an unbudgeted take silently becomes a
    // cudaMemcpy(nullptr) -> "invalid argument" deep in the launch. Unconditional (a few KB)
    // so the arena size does not depend on the base flag, which would defeat its reuse across
    // calls -- ensure_raw only grows, and a fixed-base call after a floating-base one would
    // otherwise keep the larger arena anyway.
    need += al((size_t)B*3*ct) + al((size_t)B*4*ct) + al((size_t)B*3*sizeof(int));  // + base diag
    ws->ensure_raw(need, R.fp32 ? 1 : 0);
    ws->rewind();

    CT* d_seeds  = ws->take<CT>(bn);
    // 5D.14c: [P] semantic per-problem RNG roots. Reserved HERE with every other arena block --
    // `take` past the sizing phase overruns the pre-planned workspace (observed: CUDA
    // "invalid argument" on the H2D copy). Uploaded below, once the arena exists.
    unsigned int* d_pseeds = (in.problem_seeds != nullptr) ? ws->take<unsigned int>(P) : nullptr;
    CT* d_tp = ws->take<CT>(pk*3);  CT* d_tq = ws->take<CT>(pk*4);
    CT* d_wp = ws->take<CT>(pk);    CT* d_wo = ws->take<CT>(pk);
    unsigned int* d_act = ws->take<unsigned int>(P);
    unsigned char* d_usec = ws->take<unsigned char>(B);
    CT* d_cq = ws->take<CT>(bn); CT* d_cpe = ws->take<CT>(bk); CT* d_coe = ws->take<CT>(bk);
    CT* d_cc = ws->take<CT>(B);  unsigned char* d_cs = ws->take<unsigned char>(B);
    CT* d_lq = ws->take<CT>(bn); CT* d_lpe = ws->take<CT>(bk); CT* d_loe = ws->take<CT>(bk);
    CT* d_lc = ws->take<CT>(B);  unsigned char* d_ls = ws->take<unsigned char>(B);
    unsigned char* d_lmfree = ws->take<unsigned char>(B);
    unsigned char* d_cofree = ws->take<unsigned char>(B);
    unsigned char* d_final  = ws->take<unsigned char>(B);
    unsigned char* d_fb     = ws->take<unsigned char>(B);
    CT* d_sq = ws->take<CT>(pmn); CT* d_spe = ws->take<CT>(pmk); CT* d_soe = ws->take<CT>(pmk);
    CT* d_sc = ws->take<CT>(pm);  double* d_sephys = ws->take<double>(pm); int* d_sseed = ws->take<int>(pm);
    unsigned char* d_ssucc = ws->take<unsigned char>(pm); unsigned char* d_svalid = ws->take<unsigned char>(pm);
    unsigned char* d_scfree = ws->take<unsigned char>(pm); unsigned char* d_sfb = ws->take<unsigned char>(pm);
    int* d_nsolved = ws->take<int>(P); int* d_nvalid = ws->take<int>(P);
    int* d_ncfree = ws->take<int>(P); int* d_nlmcoll = ws->take<int>(P);
    int* d_nfb = ws->take<int>(P); int* d_ninfeas = ws->take<int>(P);
    unsigned char* d_psucc = ws->take<unsigned char>(P);

    // Alternating base update (refinement only). Disabled unless the caller asked AND a base was
    // supplied: an update with no base to move is meaningless.
    BaseUpdateCfg<CT> bcfg;
    bcfg.enabled         = (in.base_update_enabled && in.base_p != nullptr) ? 1 : 0;
    bcfg.interval        = in.base_update_interval > 0 ? in.base_update_interval : 1;
    bcfg.damping         = (CT)in.base_damping;
    bcfg.damping_scale_p = (CT)in.base_damping_scale_p;
    bcfg.damping_scale_R = (CT)in.base_damping_scale_R;
    bcfg.step_scale      = (CT)in.base_step_scale;
    bcfg.max_translation = (CT)in.base_max_translation_step;
    bcfg.max_rotation    = (CT)in.base_max_rotation_step;
    for (int i = 0; i < 3; ++i) {
        bcfg.lo[i] = (CT)in.base_position_lower[i];
        bcfg.hi[i] = (CT)in.base_position_upper[i];
        if (!(bcfg.lo[i] <= bcfg.hi[i]))
            throw std::invalid_argument("base_position_lower must be <= base_position_upper");
    }

    // Floating base: candidate-level [B,3] / [B,4]. Both stay NULL for a fixed-base solve, and
    // the kernels then take their verbatim-copy branch -- no allocation, no upload, no cost.
    const bool floating_base = (in.base_p != nullptr && in.base_q != nullptr);
    CT* d_bp = floating_base ? ws->take<CT>((size_t)B * 3) : nullptr;
    CT* d_bq = floating_base ? ws->take<CT>((size_t)B * 4) : nullptr;
    int* d_bdiag = (floating_base && in.base_diag) ? ws->take<int>((size_t)B * 3) : nullptr;

    upload_in<CT>(d_seeds, in.q,     in.f32, bn, stage);
    upload_in<CT>(d_tp,    in.tgt_p, in.f32, pk*3, stage);
    upload_in<CT>(d_tq,    in.tgt_q, in.f32, pk*4, stage);
    upload_in<CT>(d_wp,    in.wp,    in.f32, pk, stage);
    upload_in<CT>(d_wo,    in.wo,    in.f32, pk, stage);
    if (floating_base) {
        upload_in<CT>(d_bp, in.base_p, in.f32, (size_t)B * 3, stage);
        upload_in<CT>(d_bq, in.base_q, in.f32, (size_t)B * 4, stage);
    }
    CUDA_OK(cudaMemcpy(d_act, in.active, sizeof(unsigned int)*P, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_usec, use_coarse_host, B, cudaMemcpyHostToDevice));

    cudaEvent_t e0,e1,e2,e3; for (auto e:{&e0,&e1,&e2,&e3}) CUDA_OK(cudaEventCreate(e));
    const int cc_enabled = R.cc_enabled ? 1 : 0;

    if (d_pseeds != nullptr) {
        CUDA_OK(cudaMemcpy(d_pseeds, in.problem_seeds, sizeof(unsigned int) * (size_t)P,
                           cudaMemcpyHostToDevice));
    }

    // ---- coarse ----
    CUDA_OK(cudaEventRecord(e0));
    if (run_coarse) {
        CUDA_OK(cudaMemcpy(d_cq, d_seeds, sizeof(CT)*bn, cudaMemcpyDeviceToDevice));
        size_t cc_smem = 0;
#if defined(HJCD_HAS_COLLISION)
        if (cc_enabled) {
            cc_smem = grid::MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<float>();
            CUDA_OK(cudaFuncSetAttribute(coarse_search_mt_kernel<CT, false, false>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cc_smem));
        }
        coarse_search_mt_kernel<CT, false, false><<<B,32,cc_smem>>>(
            d_cq, d_tp, d_tq, d_act, d_wp, d_wo, d_bp, d_bq, d_cpe, d_coe, d_cc, d_cs, nullptr, 0,
            d_rm, (CT)eps_pos, (CT)eps_ori, (CT)lambda_coord, (CT)h_min, (CT)max_step,
            coarse_iters, coarse_stall_lim, B, S, use_incremental, seed, d_pseeds, max_pert_attempts,
            0 /*hard_enabled*/, 1 /*hard_top_k*/, 0 /*oracle*/, 0.0f /*hard_margin*/,
        g1sc::HardWorkspace{},
        cc_enabled, reinterpret_cast<const grid::robotModel<float>*>(cc_model),
            cc_env_ptr ? *reinterpret_cast<const grid_collision::Environment<float>*>(cc_env_ptr)
                       : grid_collision::Environment<float>{});
#else
        (void)cc_smem;
        coarse_search_mt_kernel<CT, false, false><<<B,32,0>>>(
            d_cq, d_tp, d_tq, d_act, d_wp, d_wo, d_bp, d_bq, d_cpe, d_coe, d_cc, d_cs, nullptr, 0,
            d_rm, (CT)eps_pos, (CT)eps_ori, (CT)lambda_coord, (CT)h_min, (CT)max_step,
            coarse_iters, coarse_stall_lim, B, S, use_incremental, seed, d_pseeds, max_pert_attempts,
            0 /*hard_enabled*/, 1 /*hard_top_k*/, 0 /*oracle*/, 0.0f /*hard_margin*/,
        g1sc::HardWorkspace{},
        cc_enabled);
#endif
        CUDA_OK(cudaPeekAtLastError());
        const int NB = (int)((bn + 255) / 256);
        pick_lm_seed_kernel<CT><<<NB,256>>>(d_lq, d_cq, d_seeds, d_usec, bn, N);
    } else {
        CUDA_OK(cudaMemcpy(d_lq, d_seeds, sizeof(CT)*bn, cudaMemcpyDeviceToDevice));
    }
    CUDA_OK(cudaEventRecord(e1));

    // ---- LM ----
    const size_t smem = sizeof(LMScratch<CT>);
    if (smem > 48u*1024u)
        CUDA_OK(cudaFuncSetAttribute(lm_multi_target_kernel<CT>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    lm_multi_target_kernel<CT><<<B,32,smem>>>(
        d_lq, d_tp, d_tq, d_act, d_wp, d_wo, d_bp, d_bq, d_bdiag,
        d_lpe, d_loe, d_lc, d_ls, nullptr, nullptr, 0,
        d_rm, (CT)eps_pos, (CT)eps_ori, (CT)lambda_init, lm_iters, B, 0, stag_patience, (CT)stag_rel, S,
        bcfg);
    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaEventRecord(e2));
    // The base is IN/OUT: the LM wrote each candidate's OPTIMIZED base back into d_bp/d_bq, so
    // copy it home over the caller's seed base. Without this the base update is unobservable --
    // the kernel would optimize a base nobody can read.
    if (floating_base) {
        CUDA_OK(cudaDeviceSynchronize());
        download_out<CT>(const_cast<void*>(in.base_p), d_bp, in.f32, (size_t)B * 3, stage);
        download_out<CT>(const_cast<void*>(in.base_q), d_bq, in.f32, (size_t)B * 4, stage);
        if (d_bdiag)                       // ints, not CT: a plain copy, no precision conversion
            CUDA_OK(cudaMemcpy(const_cast<void*>(in.base_diag), d_bdiag,
                               (size_t)B * 3 * sizeof(int), cudaMemcpyDeviceToHost));
    }

    // ---- collision + candidate-local fallback ----
#if defined(HJCD_HAS_COLLISION)
    if (cc_enabled) {
        const auto env = *reinterpret_cast<const grid_collision::Environment<float>*>(cc_env_ptr);
        const auto* rmcc = reinterpret_cast<const grid::robotModel<float>*>(cc_model);
        // config_free's internal sphere-FK extractor uses the DYNAMIC shared arena, exactly like the
        // standalone mark_collisions / the coarse gate -- launch with the same smem and attribute.
        const size_t mc_smem = grid::MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<float>();
        CUDA_OK(cudaFuncSetAttribute(mark_collisions_ct<CT>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)mc_smem));
        mark_collisions_ct<CT><<<B,128,mc_smem>>>(d_lq, B, d_lmfree, rmcc, env);
        mark_collisions_ct<CT><<<B,128,mc_smem>>>(d_cq, B, d_cofree, rmcc, env);
        apply_fallback_kernel<CT><<<B,32>>>(d_lq,d_lpe,d_loe,d_lc,d_ls, d_cq,d_cpe,d_coe,d_cc,d_cs,
                                            d_lmfree,d_cofree,d_final,d_fb, B, K);
        CUDA_OK(cudaPeekAtLastError());
    }
#endif

    // ---- segmented top-M (M == 1 reproduces top-1) ----
    const int blk = seg_block_size(S);
    const size_t sel_smem = (size_t)blk * (sizeof(Cand) + 6*sizeof(int)) + (size_t)M*sizeof(int);
    segmented_topM_kernel<CT><<<P, blk, sel_smem>>>(
        d_lq, d_lpe, d_loe, d_lc, d_ls, d_final, d_fb, d_lmfree, d_seeds, d_act,
        (CT)eps_pos, (CT)eps_ori, S, P, M, cc_enabled,
        d_sq, d_spe, d_soe, d_sc, d_sephys, d_sseed, d_ssucc, d_svalid, d_scfree, d_sfb,
        d_nsolved, d_nvalid, d_psucc, d_ncfree, d_nlmcoll, d_nfb, d_ninfeas);
    CUDA_OK(cudaEventRecord(e3));
    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());
    { float ms; CUDA_OK(cudaEventElapsedTime(&ms,e0,e1)); R.coarse_ms=ms;
      CUDA_OK(cudaEventElapsedTime(&ms,e1,e2)); R.lm_ms=ms;
      CUDA_OK(cudaEventElapsedTime(&ms,e2,e3)); R.select_ms=ms; }
    for (auto e:{e0,e1,e2,e3}) CUDA_OK(cudaEventDestroy(e));

    // ---- D2H: selected only (config goes straight into the caller's [P,N] buffer) ----
    auto d2hb=[&](unsigned char* d, std::vector<unsigned char>& v, int n){ v.assign(n,0);
        CUDA_OK(cudaMemcpy(v.data(), d, n, cudaMemcpyDeviceToHost)); };
    auto d2hi=[&](int* d, std::vector<int>& v){ v.assign(P,0);
        CUDA_OK(cudaMemcpy(v.data(), d, sizeof(int)*P, cudaMemcpyDeviceToHost)); };

    CUDA_OK(cudaMemcpy(out_sel_q_ct, d_sq, sizeof(CT)*pmn, cudaMemcpyDeviceToHost));
    widen_d2h<CT>(d_spe, R.sel_pe, pmk);
    widen_d2h<CT>(d_soe, R.sel_oe, pmk);
    widen_d2h<CT>(d_sc,  R.sel_cost, pm);
    R.sel_ephys.assign(pm,0); CUDA_OK(cudaMemcpy(R.sel_ephys.data(), d_sephys, sizeof(double)*pm, cudaMemcpyDeviceToHost));
    R.sel_seed.assign(pm,0);  CUDA_OK(cudaMemcpy(R.sel_seed.data(),  d_sseed,  sizeof(int)*pm,    cudaMemcpyDeviceToHost));
    d2hb(d_ssucc,R.sel_succ,(int)pm); d2hb(d_svalid,R.sel_valid,(int)pm);
    d2hb(d_scfree,R.sel_cfree,(int)pm); d2hb(d_sfb,R.sel_fb,(int)pm);
    d2hb(d_psucc,R.prob_success,P);
    d2hi(d_nsolved,R.num_solved); d2hi(d_nvalid,R.num_valid);
    if (cc_enabled) { d2hi(d_ncfree,R.num_cfree); d2hi(d_nlmcoll,R.num_lm_coll);
                      d2hi(d_nfb,R.num_fb); d2hi(d_ninfeas,R.num_infeas); }

    if (return_all) {
        if (out_all_q_ct) CUDA_OK(cudaMemcpy(out_all_q_ct, d_lq, sizeof(CT)*bn, cudaMemcpyDeviceToHost));
        widen_d2h<CT>(d_lpe, R.all_pe, bk);
        widen_d2h<CT>(d_loe, R.all_oe, bk);
        widen_d2h<CT>(d_lc,  R.all_cost, (size_t)B);
        d2hb(d_ls, R.all_succ, B);
        R.all_cfree.assign(B, 1);
        R.all_fb.assign(B, 0);
        if (cc_enabled) {
            CUDA_OK(cudaMemcpy(R.all_cfree.data(), d_final, B, cudaMemcpyDeviceToHost));
            CUDA_OK(cudaMemcpy(R.all_fb.data(),    d_fb,    B, cudaMemcpyDeviceToHost));
        }
    }
    return R;
}

SolveProblemsOutputs compute_solve_problems(
    const SolveInputs& in, int B, int num_solutions,
    double eps_pos, double eps_ori,
    double lambda_coord, double h_min, double max_step, int coarse_iters, int coarse_stall_lim,
    int use_incremental, unsigned long long seed, int max_pert_attempts,
    double lambda_init, int lm_iters, int stag_patience, double stag_rel,
    const grid::robotModel<double>* d_robotModel, int precision,
    const unsigned char* use_coarse_host, bool run_coarse,
    const void* cc_model, const void* cc_env_ptr,
    bool return_all, HjcdWorkspace* ws, void* out_sel_q_ct, void* out_all_q_ct)
{
    if (precision == 1)
        return launch_solve_problems<float>(in, B, num_solutions, eps_pos, eps_ori, lambda_coord,
            h_min, max_step, coarse_iters, coarse_stall_lim, use_incremental, seed, max_pert_attempts,
            lambda_init, lm_iters, stag_patience, stag_rel, d_robotModel, use_coarse_host, run_coarse,
            cc_model, cc_env_ptr, return_all, ws, out_sel_q_ct, out_all_q_ct);
    return launch_solve_problems<double>(in, B, num_solutions, eps_pos, eps_ori, lambda_coord,
        h_min, max_step, coarse_iters, coarse_stall_lim, use_incremental, seed, max_pert_attempts,
        lambda_init, lm_iters, stag_patience, stag_rel, d_robotModel, use_coarse_host, run_coarse,
        cc_model, cc_env_ptr, return_all, ws, out_sel_q_ct, out_all_q_ct);
}

// Workspace factory (the solver object owns one).
HjcdWorkspace* hjcd_workspace_new() { return new HjcdWorkspace(); }
void hjcd_workspace_free(HjcdWorkspace* w) { delete w; }
void hjcd_workspace_stats(const HjcdWorkspace* w, size_t* n_malloc, size_t* n_free,
                          size_t* bytes, int* cap_B, int* device) {
    *n_malloc = w->n_malloc(); *n_free = w->n_free(); *bytes = w->bytes();
    *cap_B = w->cap_B(); *device = w->device();
}

IncrementalOutputs compute_incremental_probe(
    const double* h_q, const int* h_upd_j, const double* h_upd_v,
    const unsigned char* h_accept, int M,
    const double* h_tgt_p, const double* h_tgt_q, const unsigned int* h_active,
    const double* h_wp, const double* h_wo, int B,
    const grid::robotModel<double>* d_robotModel)
{
    const int K = hjcd::NT;
    IncrementalOutputs R;
    R.n = N; R.num_targets = K;
    R.q.assign((size_t)B*N, 0.0);
    R.joint_xform.assign((size_t)B*N*16, 0.0);
    R.target_xform.assign((size_t)B*K*16, 0.0);
    R.e_pos.assign((size_t)B*K*3, 0.0);
    R.e_ori.assign((size_t)B*K*3, 0.0);
    R.pos_norm.assign((size_t)B*K, 0.0);
    R.ori_norm.assign((size_t)B*K, 0.0);
    R.cost.assign((size_t)B*K, 0.0);
    R.total_cost.assign(B, 0.0);
    if (!h_q || B <= 0 || M < 0 || !d_robotModel) return R;

    double *d_q,*d_v,*d_tp,*d_tq,*d_wp,*d_wo,*d_oq,*d_jx,*d_tx,*d_ep,*d_eo,*d_pn,*d_on,*d_ck,*d_tot;
    int* d_j; unsigned char* d_ac; unsigned int* d_act;
    const size_t Mn = (size_t)B * (M > 0 ? M : 1);
    CUDA_OK(cudaMalloc(&d_q,  sizeof(double)*(size_t)B*N));
    CUDA_OK(cudaMalloc(&d_j,  sizeof(int)*Mn));
    CUDA_OK(cudaMalloc(&d_v,  sizeof(double)*Mn));
    CUDA_OK(cudaMalloc(&d_ac, sizeof(unsigned char)*Mn));
    CUDA_OK(cudaMalloc(&d_tp, sizeof(double)*(size_t)B*K*3));
    CUDA_OK(cudaMalloc(&d_tq, sizeof(double)*(size_t)B*K*4));
    CUDA_OK(cudaMalloc(&d_act, sizeof(unsigned int)*B));
    CUDA_OK(cudaMalloc(&d_wp, sizeof(double)*(size_t)B*K));
    CUDA_OK(cudaMalloc(&d_wo, sizeof(double)*(size_t)B*K));
    CUDA_OK(cudaMalloc(&d_oq, sizeof(double)*R.q.size()));
    CUDA_OK(cudaMalloc(&d_jx, sizeof(double)*R.joint_xform.size()));
    CUDA_OK(cudaMalloc(&d_tx, sizeof(double)*R.target_xform.size()));
    CUDA_OK(cudaMalloc(&d_ep, sizeof(double)*R.e_pos.size()));
    CUDA_OK(cudaMalloc(&d_eo, sizeof(double)*R.e_ori.size()));
    CUDA_OK(cudaMalloc(&d_pn, sizeof(double)*R.pos_norm.size()));
    CUDA_OK(cudaMalloc(&d_on, sizeof(double)*R.ori_norm.size()));
    CUDA_OK(cudaMalloc(&d_ck, sizeof(double)*R.cost.size()));
    CUDA_OK(cudaMalloc(&d_tot, sizeof(double)*B));
    CUDA_OK(cudaMemcpy(d_q, h_q, sizeof(double)*(size_t)B*N, cudaMemcpyHostToDevice));
    if (M > 0) {
        CUDA_OK(cudaMemcpy(d_j, h_upd_j, sizeof(int)*Mn, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_v, h_upd_v, sizeof(double)*Mn, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_ac, h_accept, sizeof(unsigned char)*Mn, cudaMemcpyHostToDevice));
    }
    CUDA_OK(cudaMemcpy(d_tp, h_tgt_p, sizeof(double)*(size_t)B*K*3, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_tq, h_tgt_q, sizeof(double)*(size_t)B*K*4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_act, h_active, sizeof(unsigned int)*B, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_wp, h_wp, sizeof(double)*(size_t)B*K, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_wo, h_wo, sizeof(double)*(size_t)B*K, cudaMemcpyHostToDevice));

    incremental_probe_kernel<double><<<B, 32>>>(
        d_q, d_j, d_v, d_ac, M, d_tp, d_tq, d_act, d_wp, d_wo,
        d_oq, d_jx, d_tx, d_ep, d_eo, d_pn, d_on, d_ck, d_tot, d_robotModel, B);
    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());

    auto D2H = [](void* h, const void* d, size_t n){ CUDA_OK(cudaMemcpy(h,d,n,cudaMemcpyDeviceToHost)); };
    D2H(R.q.data(), d_oq, sizeof(double)*R.q.size());
    D2H(R.joint_xform.data(), d_jx, sizeof(double)*R.joint_xform.size());
    D2H(R.target_xform.data(), d_tx, sizeof(double)*R.target_xform.size());
    D2H(R.e_pos.data(), d_ep, sizeof(double)*R.e_pos.size());
    D2H(R.e_ori.data(), d_eo, sizeof(double)*R.e_ori.size());
    D2H(R.pos_norm.data(), d_pn, sizeof(double)*R.pos_norm.size());
    D2H(R.ori_norm.data(), d_on, sizeof(double)*R.ori_norm.size());
    D2H(R.cost.data(), d_ck, sizeof(double)*R.cost.size());
    D2H(R.total_cost.data(), d_tot, sizeof(double)*B);
    for (void* p : {(void*)d_q,(void*)d_j,(void*)d_v,(void*)d_ac,(void*)d_tp,(void*)d_tq,(void*)d_act,
                    (void*)d_wp,(void*)d_wo,(void*)d_oq,(void*)d_jx,(void*)d_tx,(void*)d_ep,
                    (void*)d_eo,(void*)d_pn,(void*)d_on,(void*)d_ck,(void*)d_tot}) cudaFree(p);
    return R;
}

double bench_fk_mode(const double* h_q, int j, int iters, int mode,
                     const double* h_tgt_p, const double* h_tgt_q, const unsigned int* h_active,
                     const double* h_wp, const double* h_wo, int B,
                     const grid::robotModel<double>* d_robotModel)
{
    const int K = hjcd::NT;
    double *d_q,*d_tp,*d_tq,*d_wp,*d_wo,*d_sink; unsigned int* d_act;
    CUDA_OK(cudaMalloc(&d_q, sizeof(double)*(size_t)B*N));
    CUDA_OK(cudaMalloc(&d_tp, sizeof(double)*(size_t)B*K*3));
    CUDA_OK(cudaMalloc(&d_tq, sizeof(double)*(size_t)B*K*4));
    CUDA_OK(cudaMalloc(&d_act, sizeof(unsigned int)*B));
    CUDA_OK(cudaMalloc(&d_wp, sizeof(double)*(size_t)B*K));
    CUDA_OK(cudaMalloc(&d_wo, sizeof(double)*(size_t)B*K));
    CUDA_OK(cudaMalloc(&d_sink, sizeof(double)*B));
    CUDA_OK(cudaMemcpy(d_q, h_q, sizeof(double)*(size_t)B*N, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_tp, h_tgt_p, sizeof(double)*(size_t)B*K*3, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_tq, h_tgt_q, sizeof(double)*(size_t)B*K*4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_act, h_active, sizeof(unsigned int)*B, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_wp, h_wp, sizeof(double)*(size_t)B*K, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_wo, h_wo, sizeof(double)*(size_t)B*K, cudaMemcpyHostToDevice));

    for (int w = 0; w < 3; ++w) {          // warmup
        fk_bench_kernel<double><<<B, 32>>>(d_q, j, 32, mode, d_tp, d_tq, d_act, d_wp, d_wo,
                                           d_sink, d_robotModel, B);
    }
    CUDA_OK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    fk_bench_kernel<double><<<B, 32>>>(d_q, j, iters, mode, d_tp, d_tq, d_act, d_wp, d_wo,
                                       d_sink, d_robotModel, B);
    cudaEventRecord(e1);
    CUDA_OK(cudaEventSynchronize(e1));
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    CUDA_OK(cudaPeekAtLastError());
    for (void* p : {(void*)d_q,(void*)d_tp,(void*)d_tq,(void*)d_act,(void*)d_wp,(void*)d_wo,
                    (void*)d_sink}) cudaFree(p);
    return (double)ms;
}

NormalEquations compute_normal_equations(
    const double* h_q, const double* h_tgt_p, const double* h_tgt_q,
    const unsigned int* h_active, const double* h_wp, const double* h_wo, int B,
    const grid::robotModel<double>* d_robotModel)
{
    const int K = hjcd::NT;
    NormalEquations R;
    R.n = N;
    R.A.assign((size_t)B * N * N, 0.0);
    R.b.assign((size_t)B * N, 0.0);
    if (!h_q || B <= 0 || !d_robotModel) return R;

    double *d_q, *d_tp, *d_tq, *d_wp, *d_wo, *d_A, *d_b;
    unsigned int* d_act;
    CUDA_OK(cudaMalloc(&d_q,  sizeof(double)*(size_t)B*N));
    CUDA_OK(cudaMalloc(&d_tp, sizeof(double)*(size_t)B*K*3));
    CUDA_OK(cudaMalloc(&d_tq, sizeof(double)*(size_t)B*K*4));
    CUDA_OK(cudaMalloc(&d_act, sizeof(unsigned int)*B));
    CUDA_OK(cudaMalloc(&d_wp, sizeof(double)*(size_t)B*K));
    CUDA_OK(cudaMalloc(&d_wo, sizeof(double)*(size_t)B*K));
    CUDA_OK(cudaMalloc(&d_A,  sizeof(double)*R.A.size()));
    CUDA_OK(cudaMalloc(&d_b,  sizeof(double)*R.b.size()));
    CUDA_OK(cudaMemcpy(d_q,  h_q,  sizeof(double)*(size_t)B*N, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_tp, h_tgt_p, sizeof(double)*(size_t)B*K*3, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_tq, h_tgt_q, sizeof(double)*(size_t)B*K*4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_act, h_active, sizeof(unsigned int)*B, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_wp, h_wp, sizeof(double)*(size_t)B*K, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_wo, h_wo, sizeof(double)*(size_t)B*K, cudaMemcpyHostToDevice));

    normal_equations_kernel<double><<<B, 32>>>(d_q, d_tp, d_tq, d_act, d_wp, d_wo, d_A, d_b,
                                               d_robotModel, B);
    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(R.A.data(), d_A, sizeof(double)*R.A.size(), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(R.b.data(), d_b, sizeof(double)*R.b.size(), cudaMemcpyDeviceToHost));
    for (void* p : {(void*)d_q,(void*)d_tp,(void*)d_tq,(void*)d_act,(void*)d_wp,(void*)d_wo,
                    (void*)d_A,(void*)d_b}) cudaFree(p);
    return R;
}

ResidualOutputs compute_target_residuals(
    const double* h_q, const double* h_tgt_p, const double* h_tgt_q,
    const unsigned int* h_active, const double* h_wp, const double* h_wo,
    const double* h_eps_p, const double* h_eps_o, int B,
    const grid::robotModel<double>* d_robotModel)
{
    const int K = hjcd::NT;
    ResidualOutputs R;
    R.num_targets = K;
    R.e_pos.assign((size_t)B * K * 3, 0.0);
    R.e_ori.assign((size_t)B * K * 3, 0.0);
    R.pos_norm.assign((size_t)B * K, 0.0);
    R.ori_norm.assign((size_t)B * K, 0.0);
    R.cost.assign((size_t)B * K, 0.0);
    R.cost_raw.assign(B, 0.0);
    R.cost_norm.assign(B, 0.0);
    R.success.assign((size_t)B * K, 0);
    R.success_all.assign(B, 0);
    if (!h_q || B <= 0 || !d_robotModel) return R;

    double *d_q, *d_tp, *d_tq, *d_wp, *d_wo, *d_ep, *d_eo;
    unsigned int* d_act;
    double *d_epos, *d_eori, *d_pn, *d_on, *d_c, *d_craw, *d_cnorm;
    unsigned char *d_succ, *d_sall;
    auto M = [](void** p, size_t n) { CUDA_OK(cudaMalloc(p, n)); };
    M((void**)&d_q,   sizeof(double) * (size_t)B * N);
    M((void**)&d_tp,  sizeof(double) * (size_t)B * K * 3);
    M((void**)&d_tq,  sizeof(double) * (size_t)B * K * 4);
    M((void**)&d_act, sizeof(unsigned int) * B);
    M((void**)&d_wp,  sizeof(double) * (size_t)B * K);
    M((void**)&d_wo,  sizeof(double) * (size_t)B * K);
    M((void**)&d_ep,  sizeof(double) * K);
    M((void**)&d_eo,  sizeof(double) * K);
    M((void**)&d_epos, sizeof(double) * (size_t)B * K * 3);
    M((void**)&d_eori, sizeof(double) * (size_t)B * K * 3);
    M((void**)&d_pn,  sizeof(double) * (size_t)B * K);
    M((void**)&d_on,  sizeof(double) * (size_t)B * K);
    M((void**)&d_c,   sizeof(double) * (size_t)B * K);
    M((void**)&d_craw, sizeof(double) * B);
    M((void**)&d_cnorm, sizeof(double) * B);
    M((void**)&d_succ, sizeof(unsigned char) * (size_t)B * K);
    M((void**)&d_sall, sizeof(unsigned char) * B);

    auto H2D = [](void* d, const void* h, size_t n) {
        CUDA_OK(cudaMemcpy(d, h, n, cudaMemcpyHostToDevice));
    };
    H2D(d_q,   h_q,     sizeof(double) * (size_t)B * N);
    H2D(d_tp,  h_tgt_p, sizeof(double) * (size_t)B * K * 3);
    H2D(d_tq,  h_tgt_q, sizeof(double) * (size_t)B * K * 4);
    H2D(d_act, h_active, sizeof(unsigned int) * B);
    H2D(d_wp,  h_wp,    sizeof(double) * (size_t)B * K);
    H2D(d_wo,  h_wo,    sizeof(double) * (size_t)B * K);
    H2D(d_ep,  h_eps_p, sizeof(double) * K);
    H2D(d_eo,  h_eps_o, sizeof(double) * K);

    target_residual_kernel<double><<<B, 32>>>(
        d_q, d_tp, d_tq, d_act, d_wp, d_wo, d_ep, d_eo,
        d_epos, d_eori, d_pn, d_on, d_c, d_craw, d_cnorm, d_succ, d_sall, d_robotModel, B);
    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());

    auto D2H = [](void* h, const void* d, size_t n) {
        CUDA_OK(cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost));
    };
    D2H(R.e_pos.data(), d_epos, sizeof(double) * (size_t)B * K * 3);
    D2H(R.e_ori.data(), d_eori, sizeof(double) * (size_t)B * K * 3);
    D2H(R.pos_norm.data(), d_pn, sizeof(double) * (size_t)B * K);
    D2H(R.ori_norm.data(), d_on, sizeof(double) * (size_t)B * K);
    D2H(R.cost.data(), d_c, sizeof(double) * (size_t)B * K);
    D2H(R.cost_raw.data(), d_craw, sizeof(double) * B);
    D2H(R.cost_norm.data(), d_cnorm, sizeof(double) * B);
    D2H(R.success.data(), d_succ, sizeof(unsigned char) * (size_t)B * K);
    D2H(R.success_all.data(), d_sall, sizeof(unsigned char) * B);

    for (void* p : {(void*)d_q,(void*)d_tp,(void*)d_tq,(void*)d_act,(void*)d_wp,(void*)d_wo,
                    (void*)d_ep,(void*)d_eo,(void*)d_epos,(void*)d_eori,(void*)d_pn,(void*)d_on,
                    (void*)d_c,(void*)d_craw,(void*)d_cnorm,(void*)d_succ,(void*)d_sall})
        cudaFree(p);
    return R;
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
T* sample_ik_config_halton(const grid::robotModel<T>* d_robotModel,
                           int num_configs,
                           uint64_t seed,
                           int offset = 1,
                           int leap   = 1) {
    if (num_configs <= 0 || !d_robotModel) return nullptr;

    T* d_q = nullptr;
    cudaMalloc(&d_q, sizeof(T) * (size_t)num_configs * N);

    const int tpb = 256;
    const int gpb = (num_configs + tpb - 1) / tpb;

    sample_q_halton_kernel<T><<<gpb, tpb>>>(d_q, num_configs, seed, offset, leap);
    cudaGetLastError();
    cudaDeviceSynchronize();

    return d_q;
}

template<typename T>
std::vector<std::array<T,7>>
sample_random_target_poses(const grid::robotModel<T>* d_robotModel,
                           int num_configs, uint64_t seed) {
    std::vector<std::array<T,7>> out;
    if (num_configs <= 0 || !d_robotModel) return out;

    T* d_q = sample_ik_config_halton<T>(d_robotModel, num_configs, seed, /*offset=*/1, /*leap=*/1);
    if (!d_q) return out;

    T* d_pose7 = nullptr;
    cudaMalloc(&d_pose7, sizeof(T) * 7 * (size_t)num_configs);

    const int threads = 32;
    const int blocks  = num_configs;

    forward_kinematics_kernel<T><<<blocks, threads>>>(
        d_q, d_pose7, nullptr, d_robotModel, num_configs
    );
    cudaGetLastError();
    cudaDeviceSynchronize();

    std::vector<T> h_pose7((size_t)num_configs * 7);
    cudaMemcpy(h_pose7.data(), d_pose7,
               sizeof(T) * 7 * (size_t)num_configs, cudaMemcpyDeviceToHost);

    out.resize(num_configs);
    for (int i = 0; i < num_configs; ++i)
        for (int k = 0; k < 7; ++k)
            out[i][k] = h_pose7[(size_t)i * 7 + k];

    cudaFree(d_pose7);
    cudaFree(d_q);
    return out;
}

template<typename T>
__global__ void gather_rows_generic(
    const T* __restrict__ src,
    const int* __restrict__ idx,
    T* __restrict__ dst,
    int K, int C)
{
    int r = blockIdx.x;
    if (r >= K) return;
    int src_row = idx[r];
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        dst[r * C + j] = src[src_row * C + j];
    }
}

template<typename T>
__global__ void build_scores_kernel(const T* __restrict__ pos_err_mm,
    const T* __restrict__ ori_err_rad,
    T* __restrict__ scores,
    int B)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;

    const T ORI_TARGET_RAD = (T)1.1e-4;
    const T ORI_OUTLIER_W  = (T)1e4;
    const T ori_excess = fmax((T)0, ori_err_rad[i] - ORI_TARGET_RAD);
    scores[i] = pos_err_mm[i] + ORI_OUTLIER_W * ori_excess;
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

template<typename T>
__global__ void replicate_target7_kernel(const T* __restrict__ target7,
    T* __restrict__ out,
    int R)
{
    int r = blockIdx.x;
    if (r >= R) return;
    for (int k = threadIdx.x; k < 7; k += blockDim.x)
        out[r * 7 + k] = target7[k];
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

        float u1 = fmaxf(u01(sj), 1e-7f);
        float u2 = u01(sj);
        float g = sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);

        const double2 L = c_joint_limits[j];
        float range = (float)(L.y - L.x);
        float step = (float)sigma_frac * range * g;

        T v = X[r * N + j] + (T)step;

        if (v < (T)L.x) v = (T)L.x;
        if (v > (T)L.y) v = (T)L.y;
        X[r * N + j] = v;
    }
}

template <typename Dst, typename Src>
__global__ void cast_array(const Src* __restrict__ in,
                           Dst* __restrict__ out,
                           size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (Dst)in[i];
}

// Uniform random seeds inside the joint limits -- the restart set every mode starts from. The
// legacy coarse kernel generated these internally; the new pipeline makes them explicit so the
// LM-only and multi-target-coarse modes start from exactly the same seeds.
// Position (mm) and orientation (rad) error of a batch of EE poses against one 7-vector target.
template <typename T>
__global__ void pose_errors_kernel(const T* __restrict__ pose7, const T* __restrict__ target7,
                                   T* __restrict__ pos_mm, T* __restrict__ ori_rad, int B) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    const T* P = &pose7[(size_t)b * 7];
    const T dx = P[0]-target7[0], dy = P[1]-target7[1], dz = P[2]-target7[2];
    pos_mm[b] = sqrt(dx*dx + dy*dy + dz*dz) * (T)1000;
    T qc[4] = {P[3], P[4], P[5], P[6]};
    const T* qt = &target7[3];
    if (qc[0]*qt[0]+qc[1]*qt[1]+qc[2]*qt[2]+qc[3]*qt[3] < (T)0)
        { qc[0]=-qc[0]; qc[1]=-qc[1]; qc[2]=-qc[2]; qc[3]=-qc[3]; }
    T wv[3];
    quat_err_rotvec(qc, qt, wv);
    ori_rad[b] = sqrt(wv[0]*wv[0] + wv[1]*wv[1] + wv[2]*wv[2]);
}

template <typename T>
__global__ void random_seed_kernel(T* __restrict__ x, int B, uint64_t seed) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    for (int j = 0; j < N; ++j) {
        const uint32_t h = wanghash((uint32_t)(seed ^ (uint64_t)(b * 2654435761u + j * 40503u)));
        const T u = (T)((h & 0xFFFFFFu) / (T)0x1000000u);
        const double2 L = c_joint_limits[j];
        x[(size_t)b * N + j] = (T)L.x + u * (T)(L.y - L.x);
    }
}

template <typename T>
__global__ void scale_array(T* __restrict__ v, T s, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] *= s;
}

// Threads per block for the collision-scoring kernel (power of two for the reduction).
static constexpr int CC_TPB = 128;

// Soft environment-collision penetration cost (mm) for a batch of candidate configs, scored via
// grid_collision AFTER optimization (never on the hot solver path). One block per config,
// thread-count-invariant. Places the URDF-driven sphere model at q via the W1b batched extractor
// and reduces the per-sphere signed clearance to the same measure the old pRRTC path used:
//   cost_mm[i] = 1000 * sum_sphere max(0, -d_sphere)          (d = nearest signed clearance, m)
// Base-link spheres are already dropped at codegen (anchor < 0), matching the old pedestal skip.
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
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) red[tid] += red[tid + stride];
        __syncthreads();
    }
    if (tid == 0) cost_mm[i] = red[0];
}

// Hard collision constraint (comparison to the soft cost): per candidate config, one block, write
// valid[i] = grid_collision::config_free (SELF + environment; note the soft path is env-only). The
// boolean is thread-invariant, so lane 0 publishes it. Same dynamic-smem contract as
// score_environment_costs. Selected via HJCD_CC_MODE=hard|both (see generate_ik_solutions).
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

// Exact collision check for a batch of configs, using the SAME evaluator the solver gates on
// (grid_collision::config_free -- SELF + environment). Test/introspection entry point.
std::vector<unsigned char> check_collision_free(
    const double* h_q, int B, const char* json, const char* set_name, int idx)
{
    std::vector<unsigned char> out((size_t)B, 1);
#if defined(HJCD_HAS_COLLISION)
    if (!bind_collision_env(json, set_name, idx)) return out;
    double* d_q = nullptr; unsigned char* d_v = nullptr;
    CUDA_OK(cudaMalloc(&d_q, sizeof(double)*(size_t)B*N));
    CUDA_OK(cudaMalloc(&d_v, sizeof(unsigned char)*(size_t)B));
    CUDA_OK(cudaMemcpy(d_q, h_q, sizeof(double)*(size_t)B*N, cudaMemcpyHostToDevice));
    const size_t cc_smem = grid::MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<float>();
    CUDA_OK(cudaFuncSetAttribute((const void*)mark_collisions,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cc_smem));
    mark_collisions<<<B, CC_TPB, cc_smem>>>(
        d_q, B, d_v,
        reinterpret_cast<const grid::robotModel<float>*>(collision_model_ptr()),
        *reinterpret_cast<const grid_collision::Environment<float>*>(collision_env_ptr()));
    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(out.data(), d_v, sizeof(unsigned char)*(size_t)B, cudaMemcpyDeviceToHost));
    cudaFree(d_q); cudaFree(d_v);
#else
    (void)h_q; (void)json; (void)set_name; (void)idx;
#endif
    return out;
}

const double ENV_COLLISION_COST_W = 1.5;
const double CC_HARD_PENALTY      = 1e12;  // added to a colliding candidate's score in hard mode
const double ORI_TARGET_RAD = 1.1e-4;
const double ORI_OUTLIER_W  = 7000.0;
const float  CC_SPHERE_MARGIN_MM = 0.0f;   // env-collision margin (mm) for the coll-free tally

// RT = LM-refine compute precision (the user-facing speed/accuracy knob). RT=double is the
// full-fp64 default; RT=float runs FK/Jacobian/residual/line-search in fp32 (~2.4x cheaper FK,
// cf. coarse_search) while the normal-equations Cholesky stays fp64 inside build_ne_and_solve_warp.
// (Default RT=double is declared in the header; not repeated here.)
template<typename T, typename RT>
Result<T> generate_ik_solutions(
    T* target_pose,
    const grid::robotModel<T>* d_robotModel,
    int b_size,
    int num_solutions,
    bool collision_free,
    const char* problems_json_text,
    const char* problem_set_name,
    int problem_idx,
    bool write_stats,
    int coarse_mode,          // 0 auto | 1 none (LM-only) | 2 multi_target | 3 legacy
    int coarse_iters,
    int coarse_incremental
)
{
    init_joint_limits_from_grid();

    using std::chrono::high_resolution_clock;
    auto t0 = high_resolution_clock::now();
    CUDA_OK(cudaDeviceSynchronize());

    Result<T> result{};
    if (!d_robotModel || !target_pose || b_size <= 0) {
        const int S = 1;
        result.pos_errors   = new T[S]{ std::numeric_limits<T>::infinity() };
        result.ori_errors   = new T[S]{ std::numeric_limits<T>::infinity() };
        result.pose         = new T[7 * S]{};
        result.joint_config = new T[N * S]{};
        result.elapsed_time = 0.0;
        result.count = S;

        return result;
    }

    // Collision environment (grid_collision). Cache the uploaded obstacle set + an fp32 robot model
    // across calls with the same problem (both are constant per problem, keyed by set#idx).
    // The whole collision path is compiled in only when grid.cuh was generated with --collision
    // (HJCD_HAS_COLLISION); otherwise collision-free is disabled and the solver runs open-world.
    bool have_env = false;
    bool stop_on_first = 1;

#if defined(HJCD_HAS_COLLISION)
    static hjcd_env::DeviceEnv g_cc_env;
    static std::string g_cc_key;
    static bool g_cc_ready = false;
    static const grid::robotModel<float>* d_robotModel_cc = nullptr;

    if (collision_free) {
        if (!problems_json_text || !problem_set_name) {
            collision_free = false;
            printf("[grid_collision] Warning: collision-free requested but no problem JSON provided\n");
        } else {
            std::string key = std::string(problem_set_name) + "#" + std::to_string(problem_idx);

            if (!g_cc_ready || g_cc_key != key) {
                if (g_cc_ready) { hjcd_env::free_env(g_cc_env); g_cc_ready = false; }

                nlohmann::json all_data = nlohmann::json::parse(problems_json_text);
                nlohmann::json problems_root = all_data.at("problems");
                nlohmann::json data = hjcd_env::select_problem_instance(
                    problems_root, problem_set_name, problem_idx);

                if (data.contains("valid") && !bool(data["valid"])) {
                    collision_free = false;
                } else {
                    hjcd_env::HostEnv h = hjcd_env::problem_dict_to_env(data);
                    g_cc_env = hjcd_env::upload_env(h);
                    g_cc_key = key;
                    g_cc_ready = true;
                }
            }

            if (g_cc_ready) {
                if (!d_robotModel_cc) d_robotModel_cc = grid::init_robotModel<float>();
                have_env = true;
            }
        }
    }
#else
    if (collision_free) {
        collision_free = false;
        printf("[grid_collision] this build has no collision (regenerate grid.cuh with --collision); "
               "running open-world\n");
    }
#endif  // HJCD_HAS_COLLISION

    if (!collision_free) have_env = false;
    const bool do_cc = collision_free && have_env;

    // Coarse phase precision
    using TC = float;

    const int    B            = b_size;
    const size_t num_elems_x  = (size_t)B * N;
    const size_t num_elems_p7 = (size_t)B * 7;

    // Robot model is a process-lifetime constant (baked from the URDF). Cache it once instead of
    // malloc+H2D every call; the previous per-call init_robotModel was also never freed (leak).
    static const grid::robotModel<TC>* d_robotModel_f = grid::init_robotModel<TC>();

    TC *d_x_c=nullptr, *d_pose_c=nullptr, *d_pos_mm_c=nullptr, *d_ori_r_c=nullptr;
    TC *d_target7_c=nullptr, *d_targets_coarse_c=nullptr;

    CUDA_OK(cudaMalloc(&d_x_c,         sizeof(TC) * num_elems_x));
    CUDA_OK(cudaMalloc(&d_pose_c,      sizeof(TC) * num_elems_p7));
    CUDA_OK(cudaMalloc(&d_pos_mm_c,    sizeof(TC) * B));
    CUDA_OK(cudaMalloc(&d_ori_r_c,     sizeof(TC) * B));
    CUDA_OK(cudaMalloc(&d_target7_c,   sizeof(TC) * 7));

    // copy target pose -> float (for coarse phase only)
    {
        TC h_target7f[7];
        for (int i=0; i<7; ++i)
            h_target7f[i] = (TC)target_pose[i];
        CUDA_OK(cudaMemcpy(d_target7_c, h_target7f,
                           sizeof(TC) * 7,
                           cudaMemcpyHostToDevice));
    }

    // init errors to +inf
    {
        thrust::device_ptr<TC> p(d_pos_mm_c), o(d_ori_r_c);
        thrust::fill(p, p + B, std::numeric_limits<TC>::infinity());
        thrust::fill(o, o + B, std::numeric_limits<TC>::infinity());
    }

    // replicate target7 -> B (float coarse targets)
    CUDA_OK(cudaMalloc(&d_targets_coarse_c, sizeof(TC) * (size_t)B * 7));
    {
        const int blocks=B, tpb=32;
        replicate_target7_kernel<TC><<<blocks, tpb>>>(
            d_target7_c, d_targets_coarse_c, B);
        cudaGetLastError();
        CUDA_OK(cudaDeviceSynchronize());
    }

    // reset global stop flags
    {
        int zero=0, neg1=-1;
        CUDA_OK(cudaMemcpyToSymbol(g_stop,   &zero, sizeof(int)));
        CUDA_OK(cudaMemcpyToSymbol(g_winner, &neg1, sizeof(int)));
        cudaGetLastError();
    }

    // ---------------------------------------------------------------------------------------
    // SEEDS + COARSE STAGE (auto-dispatched)
    //
    // coarse_mode:  0 auto | 1 none (LM-only) | 2 multi_target (new) | 3 legacy (old Panda coarse)
    //
    // AUTO dispatches on the number of ACTIVE TARGET BITS, __popc(active_target_mask) -- never on
    // the generated MAX_TARGETS/NUM_TARGETS:
    //     popcount == 1  -> LM only        (measured: the coarse search WORSENS K=1 accuracy)
    //     popcount >= 2  -> new coarse -> LM (measured: LM alone converges 0% on G1 K=4)
    //
    // The public single-target API enters here as K=1, active_target_mask = 0b1, coarse_mode = auto,
    // so it takes the LM-only branch. The legacy Panda coarse search is reachable ONLY by asking for
    // it explicitly (coarse_mode = legacy); it is never an implicit default.
    // ---------------------------------------------------------------------------------------
    const unsigned int single_mask = 1u;                       // K = 1 for this public entry point
    const int popc = __builtin_popcount(single_mask);
    int mode = coarse_mode;
    if (mode == 0) mode = (popc >= 2) ? 2 : 1;                 // auto
    const bool run_coarse = (mode == 2 || mode == 3);

    // Explicit random restarts, shared by every mode.
    {
        const int tpb = 256, gpb = (B + tpb - 1) / tpb;
        random_seed_kernel<TC><<<gpb, tpb>>>(d_x_c, B, 0x5EEDull ^ (uint64_t)B);
        cudaGetLastError();
        CUDA_OK(cudaDeviceSynchronize());
    }

    if (mode == 3) {
        // ---- LEGACY: the old single-target coarse search (compatibility / ablation only) -------
        int TPB_req = std::min((int)(2 * N * WARP_SIZE), 256);
        int maxThreadsPerBlock = 0;
        CUDA_OK(cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0));
        TPB_req = std::min(TPB_req, maxThreadsPerBlock);
        TPB_req = (TPB_req + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
        TPB_req = std::max(TPB_req, WARP_SIZE);
        const size_t perWarpBytes = (size_t)(2 * NX * 16) * sizeof(TC);
        cudaFuncAttributes attr{};
        CUDA_OK(cudaFuncGetAttributes(&attr, (const void*)coarse_search<TC>));
        const size_t staticShmem = (size_t)attr.sharedSizeBytes;
        int maxOptIn=0, maxDefault=0;
        CUDA_OK(cudaDeviceGetAttribute(&maxOptIn, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
        CUDA_OK(cudaDeviceGetAttribute(&maxDefault, cudaDevAttrMaxSharedMemoryPerBlock, 0));
        size_t maxSharedAvail = (size_t)std::max(maxOptIn, maxDefault);
        size_t roomForDyn = (maxSharedAvail > staticShmem) ? (maxSharedAvail - staticShmem) : 0;
        int maxWarpsBySmem = (perWarpBytes > 0) ? (int)(roomForDyn / perWarpBytes) : 1;
        maxWarpsBySmem = std::max(1, maxWarpsBySmem);
        int warpsPerBlock = std::min(std::min(TPB_req / WARP_SIZE, maxWarpsBySmem), 4);
        warpsPerBlock = std::max(1, warpsPerBlock);
        int TPB = warpsPerBlock * WARP_SIZE;
        size_t scratchBytes = (size_t)warpsPerBlock * perWarpBytes;
        int ask = (int)std::min(maxSharedAvail, staticShmem + scratchBytes);
        CUDA_OK(cudaFuncSetAttribute((const void*)coarse_search<TC>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, ask));
        for (;;) {
            coarse_search<TC><<<B, TPB, scratchBytes>>>(
                d_x_c, d_pose_c, d_targets_coarse_c, d_pos_mm_c, d_ori_r_c, d_robotModel_f,
                stop_on_first);
            cudaError_t e = cudaPeekAtLastError();
            if (e == cudaSuccess) break;
            if (e != cudaErrorLaunchOutOfResources) CUDA_OK(e);
            if (warpsPerBlock > 1) {
                warpsPerBlock >>= 1; TPB = warpsPerBlock * WARP_SIZE;
                scratchBytes = (size_t)warpsPerBlock * perWarpBytes;
                continue;
            }
            CUDA_OK(e);
            break;
        }
        CUDA_OK(cudaDeviceSynchronize());
    } else if (mode == 2) {
        // ---- NEW multi-target coarse search (with the exact collision gate when bound) ---------
        std::vector<double> hq((size_t)B * N), hp((size_t)B * hjcd::NT * 3),
                            hqt((size_t)B * hjcd::NT * 4),
                            hwp((size_t)B * hjcd::NT, 1.0), hwo((size_t)B * hjcd::NT, 1.0);
        std::vector<unsigned int> hact((size_t)B, single_mask);
        {
            std::vector<TC> tmp((size_t)B * N);
            CUDA_OK(cudaMemcpy(tmp.data(), d_x_c, sizeof(TC)*tmp.size(), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < tmp.size(); ++i) hq[i] = (double)tmp[i];
        }
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < 3; ++c) hp[(size_t)b*hjcd::NT*3 + c] = (double)target_pose[c];
            for (int c = 0; c < 4; ++c) hqt[(size_t)b*hjcd::NT*4 + c] = (double)target_pose[3 + c];
        }
        const void* ccm = do_cc ? collision_model_ptr() : nullptr;
        const void* cce = do_cc ? collision_env_ptr()   : nullptr;
        static const grid::robotModel<double>* d_rm_d = grid::init_robotModel<double>();
        // Legacy single-target path: its own workspace, fp64 in/out, single-threaded by construction.
        static HjcdWorkspace* legacy_ws = hjcd_workspace_new();
        std::vector<double> co_q((size_t)B * N);
        SolveInputs si{hq.data(), hp.data(), hqt.data(), hwp.data(), hwo.data(), hact.data(),
                       /*f32=*/false};
        CoarseOutputs co = compute_coarse_search(
            si, B,
            /*eps_pos=*/1e-8, /*eps_ori=*/1e-8, /*lambda_coord=*/1e-6, /*h_min=*/1e-9,
            /*max_step=*/0.35, /*max_iters=*/coarse_iters, /*stall_lim=*/5,
            /*use_incremental=*/coarse_incremental, /*seed=*/0xC0A45Eull, d_rm_d,
            /*diagnostics=*/false, ccm, cce, /*max_pert_attempts=*/4, /*precision=*/0,
            legacy_ws, co_q.data());
        (void)co;
        {
            std::vector<TC> tmp((size_t)B * N);
            for (size_t i = 0; i < tmp.size(); ++i) tmp[i] = (TC)co_q[i];
            CUDA_OK(cudaMemcpy(d_x_c, tmp.data(), sizeof(TC)*tmp.size(), cudaMemcpyHostToDevice));
        }
    }
    // mode == 1 (LM only): the seeds go straight to LM, untouched.

    // Score whatever the coarse stage produced (or the raw seeds, for LM-only) so the downstream
    // top-K selection has errors to sort on.
    if (!run_coarse) {
        forward_kinematics_kernel<TC><<<B, 32>>>(d_x_c, d_pose_c, nullptr, d_robotModel_f, B);
        cudaGetLastError();
        CUDA_OK(cudaDeviceSynchronize());
        pose_errors_kernel<TC><<<(B + 255)/256, 256>>>(d_pose_c, d_target7_c, d_pos_mm_c,
                                                       d_ori_r_c, B);
        cudaGetLastError();
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
    const int top_k_req = sch.top_k * std::max(1, num_solutions / 2);
    const int repeats = sch.repeats;
    const double sigma_frac = sch.sigma_frac;
    const bool keep_one = sch.keep_one;

    // score: pos + ori error (float)
    TC* d_scores_c = nullptr;
    CUDA_OK(cudaMalloc(&d_scores_c, sizeof(TC) * B));
    {
        const int tpb = 256, gpb = (B + tpb - 1) / tpb;
        build_scores_kernel<TC><<<gpb, tpb>>>(
            d_pos_mm_c, d_ori_r_c, d_scores_c, B);
        cudaGetLastError();
    }

    // sort configs and gather top K
    thrust::device_vector<int> d_idx(B);
    thrust::sequence(d_idx.begin(), d_idx.end(), 0);

    {
        thrust::device_ptr<TC> s_ptr(d_scores_c);
        thrust::sort_by_key(s_ptr, s_ptr + B, d_idx.begin());
    }

    // K in [1, B]
    const int K = (top_k_req <= 0) ? B : std::min(B, std::max(1, top_k_req));

    thrust::device_vector<int> d_top_idx(K);
    thrust::copy(d_idx.begin(), d_idx.begin() + K, d_top_idx.begin());

    TC* d_x_top_c = nullptr;
    CUDA_OK(cudaMalloc(&d_x_top_c, sizeof(TC) * (size_t)K * N));
    {
        const int blocks = K, tpb = 128;
        gather_rows_kernel<TC><<<blocks, tpb>>>(
            d_x_c,
            thrust::raw_pointer_cast(d_top_idx.data()),
            d_x_top_c,
            K
        );
        cudaGetLastError();
    }

    const int Krep = K * repeats;
    TC* d_x_rep_c = nullptr;
    CUDA_OK(cudaMalloc(&d_x_rep_c, sizeof(TC) * (size_t)Krep * N));
    {
        const int blocks = K, tpb = 128;
        replicate_rows_kernel<TC><<<blocks, tpb>>>(
            d_x_top_c, d_x_rep_c, K, N, repeats);
        cudaGetLastError();
    }
    {
        const int blocks = Krep, tpb = 128;
        perturb_rows_kernel<TC><<<blocks, tpb>>>(
            d_x_rep_c, Krep, (TC)sigma_frac, 0xC0FFEEull, repeats, keep_one);
        cudaGetLastError();
    }

    CUDA_OK(cudaDeviceSynchronize());

    // JACOBIAN LM TUNER — runs in the refine precision RT (double by default; float = the
    // fp32 speed knob). The Cholesky solve inside build_ne_and_solve_warp stays fp64 regardless.
    // (Array names keep the "64" suffix for continuity; their element type is RT.)
    RT *dx64=nullptr, *dtgt64=nullptr, *dpose64=nullptr;
    RT *dposmm64=nullptr, *dori64=nullptr;
    const size_t KrepN = (size_t)Krep * N;
    const size_t Krep7 = (size_t)Krep * 7;

    CUDA_OK(cudaMalloc(&dx64,    sizeof(RT) * KrepN));
    CUDA_OK(cudaMalloc(&dtgt64,  sizeof(RT) * Krep7));
    CUDA_OK(cudaMalloc(&dpose64, sizeof(RT) * Krep7));
    CUDA_OK(cudaMalloc(&dposmm64,sizeof(RT) * Krep));
    CUDA_OK(cudaMalloc(&dori64,  sizeof(RT) * Krep));

    // cast coarse float -> RT
    {
        const int tpb = 256;
        int gpb = (int)((KrepN + tpb - 1) / tpb);
        cast_array<RT, TC><<<gpb, tpb>>>(d_x_rep_c, dx64, KrepN);
        cudaGetLastError();
        CUDA_OK(cudaDeviceSynchronize());
    }

    // build RT targets
    RT h_target7d[7];
    for (int i = 0; i < 7; ++i)
        h_target7d[i] = static_cast<RT>(target_pose[i]);

    RT* d_target7_d = nullptr;
    CUDA_OK(cudaMalloc(&d_target7_d, sizeof(RT) * 7));
    CUDA_OK(cudaMemcpy(d_target7_d, h_target7d,
                       sizeof(RT) * 7,
                       cudaMemcpyHostToDevice));

    {
        const int blocks = Krep;
        const int tpb    = 32;
        replicate_target7_kernel<RT><<<blocks, tpb>>>(
            d_target7_d, dtgt64, Krep);
        cudaGetLastError();
        CUDA_OK(cudaDeviceSynchronize());
    }

    static auto* d_robotModel_rt = grid::init_robotModel<RT>();  // cached once per RT instantiation
    {
        int zero = 0, neg1 = -1;
        CUDA_OK(cudaMemcpyToSymbol(g_stop,   &zero, sizeof(int)));
        CUDA_OK(cudaMemcpyToSymbol(g_winner, &neg1, sizeof(int)));
    }

    // The single-target public path runs the SAME multi-target LM with K=1 and mask bit 0 active --
    // there is no second LM implementation. On a build whose generated target set has K > 1 the
    // 7-vector target is ambiguous, so the caller must use the multi-target refine() entry instead.
    static_assert(hjcd::NT >= 1, "no generated targets");
    if (hjcd::NT != 1) {
        printf("[hjcd] generate_solutions takes ONE target pose, but this build generated %d target "
               "frames. Use the multi-target API (hjcdik.refine).\n", hjcd::NT);
        result.pos_errors   = new T[1]{ std::numeric_limits<T>::infinity() };
        result.ori_errors   = new T[1]{ std::numeric_limits<T>::infinity() };
        result.pose         = new T[7]{};
        result.joint_config = new T[N]{};
        result.count = 1;
        return result;
    }

    {
        const int max_iters = 40;

        // Convergence / early-stop tolerance (pos in m, ori in rad). Default 1e-8 m is far below the
        // fp32 representable floor at ~0.5 m coords, so fp32 can't early-stop and grinds all iters at
        // num_solutions=1 -> env knobs let us sweep a precision-appropriate looser tol.
        RT eps_pos = (RT)1e-8, eps_ori = (RT)1e-8;
        if (const char* e = std::getenv("HJCD_LM_EPS_POS")) { double v = std::atof(e); if (v > 0) eps_pos = (RT)v; }
        if (const char* e = std::getenv("HJCD_LM_EPS_ORI")) { double v = std::atof(e); if (v > 0) eps_ori = (RT)v; }

        // K=1 problem arrays: position + quaternion split out of the 7-vector, mask = bit 0, unit
        // weights. dtgt64 already holds the replicated 7-vectors, so split on the host once.
        std::vector<RT> h_p((size_t)Krep * 3), h_qt((size_t)Krep * 4), h_w((size_t)Krep, (RT)1);
        std::vector<unsigned int> h_act((size_t)Krep, 1u);
        for (int i = 0; i < Krep; ++i) {
            for (int c = 0; c < 3; ++c) h_p[(size_t)i*3 + c] = (RT)target_pose[c];
            for (int c = 0; c < 4; ++c) h_qt[(size_t)i*4 + c] = (RT)target_pose[3 + c];
        }
        RT *d_p=nullptr, *d_qt=nullptr, *d_wp=nullptr, *d_wo=nullptr, *d_cost=nullptr;
        unsigned int* d_act=nullptr; unsigned char* d_su=nullptr;
        CUDA_OK(cudaMalloc(&d_p,  sizeof(RT)*h_p.size()));
        CUDA_OK(cudaMalloc(&d_qt, sizeof(RT)*h_qt.size()));
        CUDA_OK(cudaMalloc(&d_wp, sizeof(RT)*h_w.size()));
        CUDA_OK(cudaMalloc(&d_wo, sizeof(RT)*h_w.size()));
        CUDA_OK(cudaMalloc(&d_act, sizeof(unsigned int)*h_act.size()));
        CUDA_OK(cudaMalloc(&d_cost, sizeof(RT)*Krep));
        CUDA_OK(cudaMalloc(&d_su, sizeof(unsigned char)*Krep));
        CUDA_OK(cudaMemcpy(d_p,  h_p.data(),  sizeof(RT)*h_p.size(),  cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_qt, h_qt.data(), sizeof(RT)*h_qt.size(), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_wp, h_w.data(),  sizeof(RT)*h_w.size(),  cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_wo, h_w.data(),  sizeof(RT)*h_w.size(),  cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_act, h_act.data(), sizeof(unsigned int)*h_act.size(), cudaMemcpyHostToDevice));

        const size_t lm_smem = sizeof(LMScratch<RT>);
        if (lm_smem > (size_t)48 * 1024)
            CUDA_OK(cudaFuncSetAttribute(lm_multi_target_kernel<RT>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)lm_smem));

        lm_multi_target_kernel<RT><<<Krep, 32, lm_smem>>>(
            dx64, d_p, d_qt, d_act, d_wp, d_wo, /*base_p=*/nullptr, /*base_q=*/nullptr,
            /*out_base_diag=*/nullptr,
            dposmm64, dori64, d_cost, d_su, dpose64,
            /*out_trace=*/nullptr, /*trace_cap=*/0,
            d_robotModel_rt, eps_pos, eps_ori, (RT)5e-3, max_iters, Krep,
            /*stop_on_first=*/(num_solutions > 1) ? 0 : 1,
            /*stag_patience=*/0, /*stag_rel=*/(RT)1e-4,     // legacy single-target path: Policy B off
            /*seeds_per_problem=*/1,                         // one problem per candidate (P = B)
            BaseUpdateCfg<RT>{});                            // legacy path: fixed base
        cudaGetLastError();
        CUDA_OK(cudaDeviceSynchronize());

        // The LM reports pos_err in METRES per target; downstream expects millimetres.
        {
            const int tpb = 256, gpb = (Krep + tpb - 1) / tpb;
            scale_array<RT><<<gpb, tpb>>>(dposmm64, (RT)1000.0, Krep);
            cudaGetLastError();
            CUDA_OK(cudaDeviceSynchronize());
        }
        for (void* p : {(void*)d_p,(void*)d_qt,(void*)d_wp,(void*)d_wo,(void*)d_act,
                        (void*)d_cost,(void*)d_su}) cudaFree(p);
    }

    // Collision scoring mode (comparison knob, env HJCD_CC_MODE): "soft" (default) = penetration cost
    // biases selection; "hard" = grid_collision::config_free filters colliding candidates outright
    // (self + env); "both" = soft cost + hard filter. Default preserves prior behavior.
    int cc_mode = 0;  // 0=soft, 1=hard, 2=both
    if (const char* e = std::getenv("HJCD_CC_MODE")) {
        std::string m(e);
        if (m == "hard") cc_mode = 1;
        else if (m == "both") cc_mode = 2;
    }
    const bool use_soft = do_cc && (cc_mode == 0 || cc_mode == 2);
    const bool use_hard = do_cc && (cc_mode == 1 || cc_mode == 2);

    std::vector<float> h_env_cost_refined(Krep, 0.0f);
    std::vector<float> h_env_cost_coarse(B, 0.0f);
    std::vector<unsigned char> h_valid_refined(Krep, 1);   // 1 = collision-free (hard mode)
    std::vector<unsigned char> h_valid_coarse(B, 1);
    float* d_env_cost_refined = nullptr;
    float* d_env_cost_coarse = nullptr;
    unsigned char* d_valid_refined = nullptr;
    unsigned char* d_valid_coarse = nullptr;
    double* dx_coarse64 = nullptr;
    int n_cc_in_refined = 0, n_cc_in_coarse = 0;

#if defined(HJCD_HAS_COLLISION)
    if (do_cc) {
        CUDA_OK(cudaMalloc(&dx_coarse64, sizeof(double) * num_elems_x));
        {
            const int tpb = 256;
            const int gpb = (int)((num_elems_x + tpb - 1) / tpb);
            cast_array<double, TC><<<gpb, tpb>>>(d_x_c, dx_coarse64, num_elems_x);
            cudaGetLastError();
            CUDA_OK(cudaDeviceSynchronize());
        }
        // Dynamic smem for the multi_target FK extractor (shared by both collision kernels).
        const size_t cc_smem = grid::MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<float>();

        // Refined q as double (both collision kernels read double). Reuse dx64 when RT==double.
        double* dq_ref = nullptr; bool dq_ref_owned = false;
        if constexpr (std::is_same_v<RT, double>) {
            dq_ref = dx64;
        } else {
            CUDA_OK(cudaMalloc(&dq_ref, sizeof(double) * KrepN));
            cast_array<double, RT><<<(int)((KrepN + 255) / 256), 256>>>(dx64, dq_ref, KrepN);
            CUDA_OK(cudaDeviceSynchronize());
            dq_ref_owned = true;
        }

        if (use_soft) {
            CUDA_OK(cudaMalloc(&d_env_cost_refined, sizeof(float) * (size_t)Krep));
            CUDA_OK(cudaMalloc(&d_env_cost_coarse, sizeof(float) * (size_t)B));
            CUDA_OK(cudaFuncSetAttribute((const void*)score_environment_costs,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cc_smem));
            score_environment_costs<<<Krep, CC_TPB, cc_smem>>>(
                dq_ref, Krep, d_env_cost_refined, d_robotModel_cc, g_cc_env.env);
            score_environment_costs<<<B, CC_TPB, cc_smem>>>(
                dx_coarse64, B, d_env_cost_coarse, d_robotModel_cc, g_cc_env.env);
            cudaGetLastError();
            CUDA_OK(cudaDeviceSynchronize());
            CUDA_OK(cudaMemcpy(h_env_cost_refined.data(), d_env_cost_refined,
                               sizeof(float) * (size_t)Krep, cudaMemcpyDeviceToHost));
            CUDA_OK(cudaMemcpy(h_env_cost_coarse.data(), d_env_cost_coarse,
                               sizeof(float) * (size_t)B, cudaMemcpyDeviceToHost));
            for (int i = 0; i < Krep; ++i)
                if (h_env_cost_refined[i] > CC_SPHERE_MARGIN_MM) ++n_cc_in_refined;
            for (int i = 0; i < B; ++i)
                if (h_env_cost_coarse[i] > CC_SPHERE_MARGIN_MM) ++n_cc_in_coarse;
        }

        if (use_hard) {
            CUDA_OK(cudaMalloc(&d_valid_refined, sizeof(unsigned char) * (size_t)Krep));
            CUDA_OK(cudaMalloc(&d_valid_coarse, sizeof(unsigned char) * (size_t)B));
            CUDA_OK(cudaFuncSetAttribute((const void*)mark_collisions,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cc_smem));
            mark_collisions<<<Krep, CC_TPB, cc_smem>>>(
                dq_ref, Krep, d_valid_refined, d_robotModel_cc, g_cc_env.env);
            mark_collisions<<<B, CC_TPB, cc_smem>>>(
                dx_coarse64, B, d_valid_coarse, d_robotModel_cc, g_cc_env.env);
            cudaGetLastError();
            CUDA_OK(cudaDeviceSynchronize());
            CUDA_OK(cudaMemcpy(h_valid_refined.data(), d_valid_refined,
                               sizeof(unsigned char) * (size_t)Krep, cudaMemcpyDeviceToHost));
            CUDA_OK(cudaMemcpy(h_valid_coarse.data(), d_valid_coarse,
                               sizeof(unsigned char) * (size_t)B, cudaMemcpyDeviceToHost));
        }

        if (dq_ref_owned) cudaFree(dq_ref);
    }
#endif  // HJCD_HAS_COLLISION

    // Read the RT device results back into double host buffers (downstream stays fp64).
    std::vector<double> h_posmm64(Krep), h_orir64(Krep);
    std::vector<double> h_pose64(Krep7), h_x64(KrepN);
    if constexpr (std::is_same_v<RT, double>) {
        CUDA_OK(cudaMemcpy(h_posmm64.data(), dposmm64, sizeof(double)*Krep,  cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(h_orir64 .data(), dori64,   sizeof(double)*Krep,  cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(h_pose64 .data(), dpose64,  sizeof(double)*Krep7, cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(h_x64    .data(), dx64,      sizeof(double)*KrepN, cudaMemcpyDeviceToHost));
    } else {
        std::vector<RT> t_posmm(Krep), t_orir(Krep), t_pose(Krep7), t_x(KrepN);
        CUDA_OK(cudaMemcpy(t_posmm.data(), dposmm64, sizeof(RT)*Krep,  cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(t_orir .data(), dori64,   sizeof(RT)*Krep,  cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(t_pose .data(), dpose64,  sizeof(RT)*Krep7, cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(t_x    .data(), dx64,      sizeof(RT)*KrepN, cudaMemcpyDeviceToHost));
        std::copy(t_posmm.begin(), t_posmm.end(), h_posmm64.begin());
        std::copy(t_orir .begin(), t_orir .end(), h_orir64 .begin());
        std::copy(t_pose .begin(), t_pose .end(), h_pose64 .begin());
        std::copy(t_x    .begin(), t_x    .end(), h_x64    .begin());
    }

    // IK accuracy and collision-filter interaction stats
    int n_ik_lost = 0, n_ik_good_ref = 0, n_coll_free_ref = 0, n_feasible_ref = 0;
    float env_cost_min = std::numeric_limits<float>::infinity();
    float env_cost_max = 0.0f;
    double env_cost_mean = 0.0;
    {
        constexpr double POS_THR_MM  = 5.0;
        constexpr double ORI_THR_RAD = 1e-3;
        for (int i = 0; i < Krep; ++i) {
            const bool ik_good   = h_posmm64[i] < POS_THR_MM && h_orir64[i] < ORI_THR_RAD;
            const bool coll_free = !do_cc
                || (use_hard ? (bool)h_valid_refined[i]
                             : (h_env_cost_refined[i] <= CC_SPHERE_MARGIN_MM));
            if (ik_good)               ++n_ik_good_ref;
            if (coll_free)             ++n_coll_free_ref;
            if (ik_good && coll_free)  ++n_feasible_ref;
            if (ik_good && !coll_free) ++n_ik_lost;
            if (do_cc) {
                const float c = h_env_cost_refined[i];
                if (c < env_cost_min) env_cost_min = c;
                if (c > env_cost_max) env_cost_max = c;
                env_cost_mean += c;
            }
        }
        if (do_cc && Krep > 0) env_cost_mean /= Krep;
    }

    // GET SOLUTIONS
    const int S_target = std::max(1, num_solutions);
    auto score_ref = [&](int i)->double {
        const double ori_excess = std::max(0.0, h_orir64[i] - ORI_TARGET_RAD);
        double s = h_posmm64[i] + ORI_OUTLIER_W * ori_excess;
        if (use_soft) s += ENV_COLLISION_COST_W * (double)h_env_cost_refined[i];
        if (use_hard && !h_valid_refined[i]) s += CC_HARD_PENALTY;
        return s;
    };

    std::vector<int> order(Krep);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b){ return score_ref(a) < score_ref(b); });

    const T DUP_TOL = (T)1e-7;
    auto is_dup = [&](int ia, int ib)->bool {
        const double* qa = &h_x64[(size_t)ia * N];
        const double* qb = &h_x64[(size_t)ib * N];
        for (int j = 0; j < N; ++j)
            if (std::fabs(qa[j] - qb[j]) > (double)DUP_TOL)
                return false;
        return true;
    };

    auto score_coarse = [&](int i)->double {
        const double ori_excess = std::max(0.0, (double)h_ori_rad_coarse_f[i] - ORI_TARGET_RAD);
        double s = (double)h_pos_mm_coarse_f[i] + ORI_OUTLIER_W * ori_excess;
        if (use_soft) s += ENV_COLLISION_COST_W * (double)h_env_cost_coarse[i];
        if (use_hard && !h_valid_coarse[i]) s += CC_HARD_PENALTY;
        return s;
    };

    std::vector<int> chosen;
    chosen.reserve(S_target);
    for (int idx : order) {
        bool dup = false;
        for (int c : chosen) {
            if (is_dup(idx, c)) { dup = true; break; }
        }
        if (!dup) {
            chosen.push_back(idx);
            if ((int)chosen.size() == S_target) break;
        }
    }

    if ((int)chosen.size() < S_target) {
        std::vector<int> order_coarse(B);
        std::iota(order_coarse.begin(), order_coarse.end(), 0);
        std::sort(order_coarse.begin(), order_coarse.end(),
                [&](int a, int b){ return score_coarse(a) < score_coarse(b); });

        for (int cidx : order_coarse) {
            chosen.push_back(-1 - cidx);
            if ((int)chosen.size() == S_target) break;
        }
    }

    // Tally quality of the returned solutions
    int n_out_ik = 0, n_out_cf = 0, n_out_feasible = 0;
    if (do_cc) {
        constexpr double POS_THR = 5.0, ORI_THR = 1e-3;
        for (int idx : chosen) {
            double pos_mm, ori_r; float env_mm; bool cfree;
            if (idx >= 0) {
                pos_mm = h_posmm64[idx]; ori_r = h_orir64[idx];
                env_mm = h_env_cost_refined[idx];
                cfree = use_hard ? (bool)h_valid_refined[idx] : (env_mm <= CC_SPHERE_MARGIN_MM);
            } else {
                int cidx = -1 - idx;
                pos_mm = h_pos_mm_coarse_f[cidx]; ori_r = h_ori_rad_coarse_f[cidx];
                env_mm = h_env_cost_coarse[cidx];
                cfree = use_hard ? (bool)h_valid_coarse[cidx] : (env_mm <= CC_SPHERE_MARGIN_MM);
            }
            if (pos_mm < POS_THR && ori_r < ORI_THR)                ++n_out_ik;
            if (cfree)                                              ++n_out_cf;
            if (pos_mm < POS_THR && ori_r < ORI_THR && cfree)       ++n_out_feasible;
        }
    }

    if (write_stats) {
        constexpr const char* CSV_PATH = "ik_stats.csv";
        static bool s_header_written = false;
        std::ofstream csv(CSV_PATH, s_header_written ? std::ios::app : std::ios::trunc);
        if (csv.is_open()) {
            if (!s_header_written) {
                s_header_written = true;
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
                << ',' << n_coll_free_ref
                << ',' << n_feasible_ref
                << ',' << n_ik_lost
                << ',' << (do_cc ? n_cc_in_refined : -1)
                << ',' << (do_cc ? n_cc_in_coarse  : -1)
                << ',' << (do_cc ? env_cost_min  : -1.f)
                << ',' << (do_cc ? env_cost_max  : -1.f)
                << ',' << (do_cc ? env_cost_mean : -1.0)
                << ',' << (int)chosen.size()
                << ',' << n_out_ik
                << ',' << n_out_cf
                << ',' << pct_cf
                << ',' << n_out_feasible
                << '\n';
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
        int idx = chosen[r];
        if (idx >= 0) {
            result.pos_errors[r] = (T)h_posmm64[idx];
            result.ori_errors[r] = (T)h_orir64[idx];
            for (int k = 0; k < 7; ++k)
                result.pose[r * 7 + k] =
                    (T)h_pose64[(size_t)idx * 7 + k];
            for (int j = 0; j < N; ++j)
                result.joint_config[(size_t)r * N + j] =
                    (T)h_x64[(size_t)idx * N + j];
        } else {
            int cidx = -1 - idx; // from coarse
            result.pos_errors[r] = (T)h_pos_mm_coarse_f[cidx];
            result.ori_errors[r] = (T)h_ori_rad_coarse_f[cidx];
            for (int k = 0; k < 7; ++k)
                result.pose[r * 7 + k] =
                    (T)h_pose_coarse_f[(size_t)cidx * 7 + k];
            for (int j = 0; j < N; ++j)
                result.joint_config[(size_t)r * N + j] =
                    (T)h_x_coarse_f[(size_t)cidx * N + j];
        }
    }

    // CLEAN-UP
    cudaFree(d_scores_c);
    cudaFree(d_x_top_c);
    cudaFree(d_x_rep_c);

    cudaFree(d_targets_coarse_c);
    cudaFree(d_x_c);
    cudaFree(d_pose_c);
    cudaFree(d_pos_mm_c);
    cudaFree(d_ori_r_c);
    cudaFree(d_target7_c);

    cudaFree(dx64);
    cudaFree(dtgt64);
    cudaFree(dpose64);
    cudaFree(dposmm64);
    cudaFree(dori64);
    cudaFree(d_target7_d);
    if (dx_coarse64) cudaFree(dx_coarse64);

    if (d_env_cost_refined) cudaFree(d_env_cost_refined);
    if (d_env_cost_coarse) cudaFree(d_env_cost_coarse);
    if (d_valid_refined) cudaFree(d_valid_refined);
    if (d_valid_coarse) cudaFree(d_valid_coarse);

    auto t1 = high_resolution_clock::now();
    result.elapsed_time =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    return result;
}

template Result<double> generate_ik_solutions<double>(   // RT=double (full fp64, default)
    double* target_pose,
    const grid::robotModel<double>* d_robotModel,
    int b_size,
    int num_solutions,
    bool collision_free,
    const char* problems_json_text,
    const char* problem_set_name,
    int problem_idx,
    bool write_stats,
    int coarse_mode,
    int coarse_iters,
    int coarse_incremental
);

template Result<double> generate_ik_solutions<double, float>(   // RT=float (fp32 refine knob)
    double* target_pose,
    const grid::robotModel<double>* d_robotModel,
    int b_size,
    int num_solutions,
    bool collision_free,
    const char* problems_json_text,
    const char* problem_set_name,
    int problem_idx,
    bool write_stats,
    int coarse_mode,
    int coarse_iters,
    int coarse_incremental
);

template Result<float> generate_ik_solutions<float>(
    float* target_pose,
    const grid::robotModel<float>* d_robotModel,
    int b_size,
    int num_solutions,
    bool collision_free,
    const char* problems_json_text,
    const char* problem_set_name,
    int problem_idx,
    bool write_stats,
    int coarse_mode,
    int coarse_iters,
    int coarse_incremental
);

template std::vector<double> compute_link_transforms<double>(
    const double* h_q, int B, const grid::robotModel<double>* d_robotModel);

template std::vector<double> compute_target_transforms<double>(
    const double* h_q, int B, const grid::robotModel<double>* d_robotModel);

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

template grid::robotModel<double>* grid::init_robotModel<double>();
template grid::robotModel<float>* grid::init_robotModel<float>();
