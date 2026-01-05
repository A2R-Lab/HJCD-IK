#include "include/hjcd_kernel.h"
#include "grid.cuh"

// Test .cuh files (uncomment)
//#include "include/test_cuh/panda_grid.cuh"
//#include "include/test_cuh/fetch_grid.cuh"

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

#ifndef CUDA_OK
#define CUDA_OK(stmt)                                                         \
    do {                                                                      \
        cudaError_t __err = (stmt);                                           \
        if (__err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(__err), __FILE__, __LINE__);           \
            std::abort();                                                     \
        }                                                                     \
    } while (0)
#endif

enum : int { N = grid::NUM_JOINTS };
extern "C" int grid_num_joints() { return N; }

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef UNREFINE
#define UNREFINE 0
#endif

#ifndef FULL_WARP_MASK
#define FULL_WARP_MASK 0xFFFFFFFFu
#endif

__constant__ double2 c_joint_limits[N];

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
}

// RNG HELPER FUNCTIONS
__device__ __forceinline__ uint32_t wanghash(uint32_t a) {
    a = (a ^ 61u) ^ (a >> 16); a *= 9u; a ^= (a >> 4);
    a *= 0x27d4eb2d; a ^= (a >> 15); return a;
}

__device__ __forceinline__ float u01(uint32_t& s) {
    s = wanghash(s);
    return (s & 0x00FFFFFFu) * (1.0f / 16777216.0f);   // [0,1)
}

__device__ __forceinline__ float u11(uint32_t& s) {
    return 2.0f * u01(s) - 1.0f;
}

__device__ __forceinline__ float gauss01(uint32_t& s) {
    float u1 = fmaxf(u01(s), 1e-7f);
    float u2 = u01(s);
    float r = sqrtf(-2.0f * logf(u1));
    float phi = 6.283185307179586f * u2;
    return r * cosf(phi);
}

__device__ __forceinline__ uint32_t make_seed(
    uint32_t base,
    int global_problem,
    int local_problem,
    int joint_or_dim
) {
    uint32_t t = (blockIdx.x << 20) ^ (blockIdx.y << 10) ^ (threadIdx.x);
    t ^= (uint32_t)global_problem * 0x9E3779B9u;
    t ^= (uint32_t)local_problem * 0x85EBCA6Bu;
    t ^= (uint32_t)joint_or_dim * 0xC2B2AE35u;
    return wanghash(base ^ t);
}

// MATH HELPERS
template<typename T>
__device__ __forceinline__ T clamp_dot(T dot) {
    if (dot > T(1.0)) return T(1.0);
    if (dot < T(-1.0)) return T(-1.0);
    return dot;
}

template<typename T>
__device__ void mat_to_quat(const T* s_XmatsHom, T* quat) {
    T t;
    T m00, m11, m22;

    m00 = s_XmatsHom[0];
    m11 = s_XmatsHom[5];
    m22 = s_XmatsHom[10];

    if (m22 < 0) {
        if (m00 > m11) {
            t = 1 + m00 - m11 - m22;
            quat[0] = t;
            quat[1] = s_XmatsHom[4] + s_XmatsHom[1];
            quat[2] = s_XmatsHom[2] + s_XmatsHom[8];
            quat[3] = s_XmatsHom[9] - s_XmatsHom[6];
        }
        else {
            t = 1 - m00 + m11 - m22;
            quat[0] = s_XmatsHom[4] + s_XmatsHom[1];
            quat[1] = t;
            quat[2] = s_XmatsHom[9] + s_XmatsHom[6];
            quat[3] = s_XmatsHom[2] - s_XmatsHom[8];
        }
    }
    else {
        if (m00 < -m11) {
            t = 1 - m00 - m11 + m22;
            quat[0] = s_XmatsHom[2] + s_XmatsHom[8];
            quat[1] = s_XmatsHom[9] + s_XmatsHom[6];
            quat[2] = t;
            quat[3] = s_XmatsHom[4] - s_XmatsHom[1];
        }
        else {
            t = 1 + m00 + m11 + m22;
            quat[0] = s_XmatsHom[9] - s_XmatsHom[6];
            quat[1] = s_XmatsHom[2] - s_XmatsHom[8];
            quat[2] = s_XmatsHom[4] - s_XmatsHom[1];
            quat[3] = t;
        }
    }
    quat[0] *= 0.5 / sqrt(t);
    quat[1] *= 0.5 / sqrt(t);
    quat[2] *= 0.5 / sqrt(t);
    quat[3] *= 0.5 / sqrt(t);
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
__device__ void sample_joint_config(T* s_x, int local_problem, int global_problem) {
    const int offset = local_problem * N;
    const uint32_t base = 1337u;
    uint32_t s = make_seed(base, global_problem, local_problem, threadIdx.x);

#pragma unroll
    for (int j = 0; j < N; ++j) {
        uint32_t sj = wanghash(s ^ (uint32_t)j * 0x27d4eb2du);

        float r = u01(sj);
        float low = (float)c_joint_limits[j].x;
        float hi = (float)c_joint_limits[j].y;

        float v = fmaf(r, (hi - low), low);
        s_x[offset + j] = (T)v;
    }
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

        float v = (float)s_x[j] + step;
        v = fminf(hi, fmaxf(low, v));
        s_x[j] = (T)v;
    }
}

template<typename T>
__device__ __forceinline__ T clamp_val(T v, T lo, T hi) {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
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
__device__ __forceinline__ T clamp_step_angle(T step_rad) {
    const T MAX_STEP = (T)(15.0 * PI / 180.0);
    if (step_rad > MAX_STEP) step_rad = MAX_STEP;
    if (step_rad < -MAX_STEP) step_rad = -MAX_STEP;
    return step_rad;
}

// SOLVE
template<typename T>
__device__ T solve_pos(const T* s_jointXforms, const T* pos, const T* target_pose_local, int joint, int k, int k_max, T delta_min = 0.35, T delta_max = 0.75) {
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

    T r[3] = {
        s_jointXforms[joint * 16 + 8],
        s_jointXforms[joint * 16 + 9],
        s_jointXforms[joint * 16 + 10]
    };
    normalize_vec3(r);

    T q_ee[4];
    mat_to_quat(&s_jointXforms[(N - 1) * 16], q_ee);
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

__device__ int g_stop = 0;
__device__ int g_winner = -1;

__device__ __forceinline__ int read_stop() {
    return atomicAdd(&g_stop, 0);
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
__device__ __forceinline__ T clamp_unit(T v) { return v > (T)1 ? (T)1 : (v < (T)-1 ? (T)-1 : v); }

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
__device__ __forceinline__
void mat_to_quat_colmajor(const T* __restrict__ C, T* __restrict__ q) {
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
__device__ __forceinline__ void mat_to_quat_colmajor3x3(const T* C4x4, T* q) {
    mat_to_quat_colmajor(C4x4, q);
}

template<typename T>
__device__ T compute_pos_err_colmajor(const T* C, const T* target_pose) {
    const T dx = C[(N - 1) * 16 + 12] - target_pose[0];
    const T dy = C[(N - 1) * 16 + 13] - target_pose[1];
    const T dz = C[(N - 1) * 16 + 14] - target_pose[2];
    return sqrt(dx * dx + dy * dy + dz * dz);
}
template<typename T>
__device__ T compute_ori_err_colmajor(const T* C, const T* q_goal) {
    T qee[4]; 
    mat_to_quat_colmajor3x3(&C[(N - 1) * 16], qee);
    if (qee[0]*q_goal[0] + qee[1]*q_goal[1] + qee[2]*q_goal[2] + qee[3]*q_goal[3] < (T)0) {
        qee[0] = -qee[0]; qee[1] = -qee[1]; qee[2] = -qee[2]; qee[3] = -qee[3];
    }
    T wv[3]; 
    quat_err_rotvec(qee, q_goal, wv);
    return sqrt(wv[0]*wv[0] + wv[1]*wv[1] + wv[2]*wv[2]);
}

template<typename T>
__device__ T compute_pos_err(const T* CjX, const T* target_pose) {
    T dx = CjX[(N - 1) * 16 + 12];
    T dy = CjX[(N - 1) * 16 + 13];
    T dz = CjX[(N - 1) * 16 + 14];

    return sqrt(
        (dx - target_pose[0]) * (dx - target_pose[0]) +
        (dy - target_pose[1]) * (dy - target_pose[1]) +
        (dz - target_pose[2]) * (dz - target_pose[2])
    );
}

template<typename T>
__device__ T compute_ori_err(const T* CjX, const T* q_goal) {
    T qee[4];
    mat_to_quat_colmajor3x3(&CjX[(N-1)*16], qee);
    if (qee[0]*q_goal[0]+qee[1]*q_goal[1]+qee[2]*q_goal[2]+qee[3]*q_goal[3] < (T)0) {
        qee[0]=-qee[0]; qee[1]=-qee[1]; qee[2]=-qee[2]; qee[3]=-qee[3];
    }
    T wv[3]; quat_err_rotvec(qee, q_goal, wv);
    return sqrt(wv[0]*wv[0] + wv[1]*wv[1] + wv[2]*wv[2]);
}

// JACOBIAN TUNER
template<typename T, int M>
__device__ bool chol_solve(T A[M * M], T b[M]) {
    for (int k = 0; k < M; ++k) {
        T s = A[k * M + k];
        for (int p = 0; p < k; ++p) { T Lkp = A[k * M + p]; s -= Lkp * Lkp; }
        if (s <= (T)0) return false;
        T Lkk = sqrt(s);
        A[k * M + k] = Lkk;
        for (int i = k + 1; i < M; ++i) {
            T t = A[i * M + k];
            for (int p = 0; p < k; ++p) t -= A[i * M + p] * A[k * M + p];
            A[i * M + k] = t / Lkk;
        }
        for (int j = k + 1; j < M; ++j) A[k * M + j] = (T)0;
    }
    T y[M];
    for (int i = 0; i < M; ++i) {
        T s = b[i];
        for (int p = 0; p < i; ++p) s -= A[i * M + p] * y[p];
        y[i] = s / A[i * M + i];
    }
    for (int i = M - 1; i >= 0; --i) {
        T s = y[i];
        for (int p = i + 1; p < M; ++p) s -= A[p * M + i] * b[p];
        b[i] = s / A[i * M + i];
    }
    return true;
}

__device__ __forceinline__ void upper_index_to_rc(int idx, int M, int& r, int& c) {
    int acc = 0;
    for (int rr = 0; rr < M; ++rr) {
        int rowCount = M - rr;
        if (idx < acc + rowCount) { r = rr; c = rr + (idx - acc); return; }
        acc += rowCount;
    }
    r = c = 0;
}

template<typename T>
__device__ __forceinline__ T safe_normN(const T* v, int n) {
    T s = (T)0; for (int i = 0; i < n; ++i) s += v[i] * v[i]; return sqrt(s);
}

template<typename T>
__device__ __forceinline__
void clamp_into_limits(const T* xbase, const T* step, T* xout, const double2* limits) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
        const double2 L = limits[i];
        const T xi = xbase[i] + step[i];
        xout[i] = fmin(fmax(xi, (T)L.x), (T)L.y);
    }
}

template<typename T>
__device__ __forceinline__
void recompute_cost_scaled(T* xcur,
    T* s_jointX, T* s_XmatsHom,
    const T* row_s,
    const T* tp, const T* q_goal,
    T& cost_sq, T& pos_err_m, T& ori_err_rad)
{
    grid::X_single_thread<T>(s_jointX, s_XmatsHom, xcur, N - 1);
    const T* Cn = &s_jointX[(N - 1) * 16];
    const T dx = tp[0] - Cn[12];
    const T dy = tp[1] - Cn[13];
    const T dz = tp[2] - Cn[14];
    T qee[4]; mat_to_quat_colmajor3x3(Cn, qee);
    if (qee[0] * q_goal[0] + qee[1] * q_goal[1] + qee[2] * q_goal[2] + qee[3] * q_goal[3] < (T)0) {
        qee[0] = -qee[0]; qee[1] = -qee[1]; qee[2] = -qee[2]; qee[3] = -qee[3];
    }
    T wv[3]; quat_err_rotvec(qee, q_goal, wv);
    const T rt0 = row_s[0] * dx, rt1 = row_s[1] * dy, rt2 = row_s[2] * dz;
    const T rt3 = row_s[3] * wv[0], rt4 = row_s[4] * wv[1], rt5 = row_s[5] * wv[2];
    cost_sq = (T)0.5 * (rt0 * rt0 + rt1 * rt1 + rt2 * rt2 + rt3 * rt3 + rt4 * rt4 + rt5 * rt5);
    pos_err_m = sqrt(dx * dx + dy * dy + dz * dz);
    ori_err_rad = sqrt(wv[0] * wv[0] + wv[1] * wv[1] + wv[2] * wv[2]);
}

template<typename T, int M>
__device__ __forceinline__
void build_solve_NE_warp(const T* __restrict__ J,
    const T* __restrict__ r_scaled,
    T lambda,
    T* __restrict__ dq,
    T* __restrict__ s_diagA,
    T* __restrict__ s_gvec,
    double* __restrict__ Ad_sh,
    double* __restrict__ rhsd_sh)
{
    const int lane = threadIdx.x & 31;
    if (threadIdx.x < 32) {
        for (int t = lane; t < M * (M + 1) / 2; t += 32) {
            int r, c; upper_index_to_rc(t, M, r, c);
            double acc = 0.0;
#pragma unroll
            for (int k = 0; k < 6; ++k) acc += (double)J[k * M + r] * (double)J[k * M + c];
            Ad_sh[r * M + c] = acc;
            Ad_sh[c * M + r] = acc;
        }
        for (int r = lane; r < M; r += 32) {
            double accb = 0.0;
#pragma unroll
            for (int k = 0; k < 6; ++k) accb += (double)J[k * M + r] * (double)r_scaled[k];
            rhsd_sh[r] = accb;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < M; ++i) {
            s_diagA[i] = (T)Ad_sh[i * M + i];
            s_gvec[i] = (T)rhsd_sh[i];
            Ad_sh[i * M + i] += (double)lambda * (double)s_diagA[i];
        }
        bool ok = chol_solve<double, M>(Ad_sh, rhsd_sh);
        for (int i = 0; i < M; ++i) dq[i] = ok ? (T)rhsd_sh[i] : (T)0;
    }
    __syncthreads();
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
    T p_try[N], p_gn_norm = safe_normN(dq_gn, N);
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

    const T clip =
        (pos_err_m_hint > (T)1e-2) ? (T)0.30 :
        (pos_err_m_hint > (T)1e-3) ? (T)0.15 :
        (pos_err_m_hint > (T)2e-4) ? (T)0.08 : (T)0.03;

    const T mag = clip;
    T best_cost = cost_sq, best_pos = pos_err_m, best_ori = ori_err_rad;
    T x_trial[N]; bool accepted = false;

    for (int sgn = -1; sgn <= 1; sgn += 2) {
#pragma unroll
        for (int j = 0; j < N; ++j) x_trial[j] = x_old[j];
        const double2 L = limits[i_star];
        const T xi = x_old[i_star] + (T)sgn * mag;
        x_trial[i_star] = fmin(fmax(xi, (T)L.x), (T)L.y);
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

// Factor A and solve A x = b
template<int M>
__device__ inline bool warp_cholesky_solve_inplace(double* __restrict__ A, double* __restrict__ b) {
    const unsigned mask = FULL_WARP_MASK;
    const int lane = threadIdx.x & 31;

    // Factorization
    #pragma unroll
    for (int k = 0; k < M; ++k) {
        double Lkk;
        if (lane == k) {
            // s = A[k,k] - sum_{p<k} L[k,p]^2
            double s = A[k*M + k];
            #pragma unroll
            for (int p = 0; p < k; ++p) {
                double Lkp = A[k*M + p];
                s -= Lkp * Lkp;
            }
            if (s <= 0.0) { Lkk = 0.0; }
            else          { Lkk = sqrt(s); }
            A[k*M + k] = Lkk;
        }
        // Broadcast Lkk to all lanes
        Lkk = __shfl_sync(mask, Lkk, k);
        if (Lkk <= 0.0) return false;

        // Compute column k below diagonal
        if (lane > k && lane < M) {
            // t = A[i,k] - sum_{p<k} L[i,p]*L[k,p]
            double t = A[lane*M + k];
            #pragma unroll
            for (int p = 0; p < k; ++p) {
                double Lip = A[lane*M + p];
                double Lkp = A[k*M    + p];
                t -= Lip * Lkp;
            }
            A[lane*M + k] = t / Lkk;
        }

        // Zero upper triangle entries in row k
        if (lane == k) {
            #pragma unroll
            for (int j = k+1; j < M; ++j) A[k*M + j] = 0.0;
        }
        __syncwarp(mask);
    }

    // Forward & back substitution
    if (lane == 0) {
        double y[M];
        // Forward: L y = b
        #pragma unroll
        for (int i = 0; i < M; ++i) {
            double s = b[i];
            #pragma unroll
            for (int p = 0; p < i; ++p) s -= A[i*M + p] * y[p];
            y[i] = s / A[i*M + i];
        }
        // Backward: L^T x = y
        for (int i = M-1; i >= 0; --i) {
            double s = y[i];
            #pragma unroll
            for (int p = i+1; p < M; ++p) s -= A[p*M + i] * b[p];
            b[i] = s / A[i*M + i];
        }
    }
    __syncwarp(mask);
    return true;
}

// Warp-cooperative NE build
template<typename T, int M>
__device__ inline void build_ne_and_solve_warp(
    const T* __restrict__ J,
    const T* __restrict__ r_scaled,
    T lambda,
    T* __restrict__ dq,
    T* __restrict__ diagA,
    T* __restrict__ gvec,
    double* __restrict__ A_sh,
    double* __restrict__ b_sh
) {
    const unsigned mask = FULL_WARP_MASK;
    const int lane = threadIdx.x & 31;

    // build A = J^T J
    if (lane < M) {
        const int r = lane;
        #pragma unroll
        for (int c = 0; c < M; ++c) {
            double acc = 0.0;
            #pragma unroll
            for (int k = 0; k < 6; ++k) acc += (double)J[k*M + r] * (double)J[k*M + c];
            A_sh[r*M + c] = acc;
        }
        // b = J^T r
        double accb = 0.0;
        #pragma unroll
        for (int k = 0; k < 6; ++k) accb += (double)J[k*M + r] * (double)r_scaled[k];
        b_sh[r] = accb;
    }
    __syncwarp(mask);

    // Damping: Ad = A + lambda*diag(A)
    if (lane < M) {
        const int i = lane;
        double di = A_sh[i*M + i];
        diagA[i] = (T)di;
        gvec[i]  = (T)b_sh[i];
        A_sh[i*M + i] = di + (double)lambda * di;
    }
    __syncwarp(mask);

    // warp Cholesky
    bool ok = warp_cholesky_solve_inplace<M>(A_sh, b_sh);

    // write
    if (lane < M) dq[lane] = ok ? (T)b_sh[lane] : (T)0;
    __syncwarp(mask);
}

__device__ __forceinline__ float warp_sum(float v){
#pragma unroll
    for (int off=16; off>0; off>>=1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}
__device__ __forceinline__ double warp_sum(double v){
#pragma unroll
    for (int off=16; off>0; off>>=1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}
template<typename T>
__device__ __forceinline__ T sqr(T x){ return x*x; }

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
    int stop_on_first)
{
    #define SYNC() do { if (blockDim.x <= warpSize) { __syncwarp(); } else { __syncthreads(); } } while (0)

    __shared__ T s_x[N], x_old[N];
    __shared__ T s_XmatsHom[N*16], s_jointX[N*16], s_tmp[N*2];
    __shared__ T J[6*N], r_scaled[6], row_s[6], q_goal[4];
    __shared__ T pos_err_m, ori_err_rad, cost_sq, prev_cost;
    __shared__ T dq[N], diagA[N], gvec[N];
    __shared__ T best_x_pos[N], best_pos_seen;
    __shared__ double Ad_sh[N*N], rhsd_sh[N];
    __shared__ int s_break, stall;

    const int warp_id = threadIdx.x >> 5;
    const int gp  = blockIdx.x;
    const int tid = threadIdx.x;
    if (gp >= B || !x || !pose || !target_poses || !pos_error || !ori_error || !d_robotModel) return;

    const T  lambda_min = (T)1e-12, lambda_max = (T)1e6;
    const int stall_lim = 5;

    const T* tp = &target_poses[gp*7];
    if (tid < N) s_x[tid] = x[gp*N + tid];
    if (tid == 0) {
        q_goal[0]=tp[3]; q_goal[1]=tp[4]; q_goal[2]=tp[5]; q_goal[3]=tp[6];
        T n = rsqrt(q_goal[0]*q_goal[0]+q_goal[1]*q_goal[1]+q_goal[2]*q_goal[2]+q_goal[3]*q_goal[3]);
        q_goal[0]*=n; q_goal[1]*=n; q_goal[2]*=n; q_goal[3]*=n;
        s_break=0; stall=0; prev_cost=(T)-1; cost_sq=(T)0;
    }
    SYNC();

    T lambda = lambda_init;

    grid::load_update_XmatsHom_helpers<T>(s_XmatsHom, s_x, d_robotModel, s_tmp);
    SYNC();

    if (warp_id == 0) {
        grid::X_warp<T>(s_jointX, s_XmatsHom, s_x, N-1);
    }
    SYNC();

    if (tid == 0) {
        pos_err_m   = compute_pos_err_colmajor<T>(s_jointX, tp);
        ori_err_rad = compute_ori_err_colmajor<T>(s_jointX, &tp[3]);
    }
    SYNC();

    best_pos_seen = pos_err_m;
    if (tid < N) best_x_pos[tid] = s_x[tid];
    if (tid == 0 && pos_err_m < eps_pos && ori_err_rad < eps_ori) s_break = 1;
    SYNC(); if (s_break) goto WRITE_OUT;

    for (int it = 0; it < k_max; ++it) {
        if (stop_on_first && threadIdx.x == 0 && ((it & 1) == 0)) {
            if (atomicAdd(&g_stop, 0)) s_break = 1;
        }
        SYNC(); if (s_break) break;

        // residual
        if (tid == 0) {
            const T* Cn = &s_jointX[(N-1)*16];
            T qee[4]; mat_to_quat_colmajor3x3(Cn, qee);
            if (qee[0]*q_goal[0] + qee[1]*q_goal[1] + qee[2]*q_goal[2] + qee[3]*q_goal[3] < (T)0) {
                qee[0]=-qee[0]; qee[1]=-qee[1]; qee[2]=-qee[2]; qee[3]=-qee[3];
            }
            T wv[3]; quat_err_rotvec(qee, q_goal, wv);
            const T dx = tp[0]-Cn[12], dy = tp[1]-Cn[13], dz = tp[2]-Cn[14];
            r_scaled[0]=dx; r_scaled[1]=dy; r_scaled[2]=dz;
            r_scaled[3]=wv[0]; r_scaled[4]=wv[1]; r_scaled[5]=wv[2];
        }
        SYNC();

        // build J and row-norms
        __shared__ T row_norm2[6];
        if (tid < 6) row_norm2[tid] = (T)0;
        SYNC();

        T p0 = (T)0, p1 = (T)0, p2 = (T)0, p3 = (T)0, p4 = (T)0, p5 = (T)0;

        if (tid < N) {
            const int i = tid;
            const T* Ci = &s_jointX[i*16];
            const T* Cn = &s_jointX[(N-1)*16];

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

        unsigned mask = __activemask();
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            p0 += __shfl_down_sync(mask, p0, off);
            p1 += __shfl_down_sync(mask, p1, off);
            p2 += __shfl_down_sync(mask, p2, off);
            p3 += __shfl_down_sync(mask, p3, off);
            p4 += __shfl_down_sync(mask, p4, off);
            p5 += __shfl_down_sync(mask, p5, off);
        }

        // lane per warp accumulate into shared row sums
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

            const T cp = fmax((T)1e-4, (T)5e-3 * (T)fmax((T)1, (T)1e3*pos_err_m));
            const T co = (T)0.5;
            #pragma unroll
            for (int k=0;k<3;++k){
                const T a=fabs(r_scaled[k]);
                const T w=(a<=cp)?(T)1:(cp/(a+(T)1e-30));
                const T s=sqrt(w);
                row_s[k]*=s; r_scaled[k]*=s;
            }
            #pragma unroll
            for (int k=3;k<6;++k){
                const T a=fabs(r_scaled[k]);
                const T w=(a<=co)?(T)1:(co/(a+(T)1e-30));
                const T s=sqrt(w);
                row_s[k]*=s; r_scaled[k]*=s;
            }

            T w_ori = (pos_err_m > (T)1e-3) ? (T)0.6 :
                      (pos_err_m > (T)2e-4) ? (T)2.2 : (T)5.5;
            const T s = sqrt(w_ori);
            row_s[3]*=s; row_s[4]*=s; row_s[5]*=s;
            r_scaled[3]*=s; r_scaled[4]*=s; r_scaled[5]*=s;

            cost_sq = (T)0.5*(
                r_scaled[0]*r_scaled[0]+r_scaled[1]*r_scaled[1]+r_scaled[2]*r_scaled[2]+
                r_scaled[3]*r_scaled[3]+r_scaled[4]*r_scaled[4]+r_scaled[5]*r_scaled[5]);
        }
        SYNC();

        // Apply row_s to J
        if (tid < N) {
            const int i = tid;
#pragma unroll
            for (int k=0;k<6;++k) J[k*N+i] *= row_s[k];
        }

        // Joint-limit column scaling + LM diag/prior
        if (tid < N) {
            const int i = tid;
            const double2 L = c_joint_limits[i];
            const T span=(T)(L.y - L.x);
            const T mid =(T)(0.5*(L.x + L.y));
            const T mlo = s_x[i]-(T)L.x, mhi=(T)L.y - s_x[i];

            // margin in [~0, 0.5]
            const T mar = fmax((T)1e-6, fmin(mlo,mhi))/span;
            // column scale
            const T col = fmax((T)0.2, (T)2.0*mar);

            #pragma unroll
            for (int k=0;k<6;++k) J[k*N+i] *= col;

            const T near = (T)clamp_unit((T)1 - (T)2*mar);
            diagA[i] = near*(T)1e-3;
            gvec[i]  = diagA[i]*(mid - s_x[i]);
        }
        SYNC();

        // Build normal equations and solve in warp 0
        if (warp_id == 0) {
            build_ne_and_solve_warp<T, N>(J, r_scaled, lambda, dq, diagA, gvec, Ad_sh, rhsd_sh);
        }
        SYNC();

        if (tid == 0) {
            T R;
            if      (pos_err_m > (T)1e-2 || ori_err_rad > (T)0.6)  R=(T)0.38;
            else if (pos_err_m > (T)1e-3 || ori_err_rad > (T)0.25) R=(T)0.22;
            else if (pos_err_m > (T)2e-4 || ori_err_rad > (T)0.08) R=(T)0.12;
            else                                                   R=(T)0.05;

            T nrm=(T)0; for (int i=0;i<N;++i) nrm += dq[i]*dq[i]; nrm = sqrt(nrm);
            if (nrm > R) { T s = R/(nrm + (T)1e-18); for (int i=0;i<N;++i) dq[i]*=s; }

            const T clip = (pos_err_m > (T)1e-2)?(T)0.30:
                           (pos_err_m > (T)1e-3)?(T)0.15:
                           (pos_err_m > (T)2e-4)?(T)0.08:(T)0.03;
            for (int i=0;i<N;++i){
                const double2 lim=c_joint_limits[i];
                dq[i]=fmin(fmax(dq[i],-clip),clip);
                if ((s_x[i] <= (T)lim.x + (T)1e-4 && dq[i] < (T)0) ||
                    (s_x[i] >= (T)lim.y - (T)1e-4 && dq[i] > (T)0)) dq[i]=(T)0;
                x_old[i]=s_x[i];
            }
        }
        SYNC();

        __shared__ int accepted;
        if (tid == 0) accepted = 0;
        SYNC();

        T best_cost=(T)1e38, best_pos=pos_err_m, best_ori=ori_err_rad, best_a=(T)0;

        // short backtracking schedule: 1.0, 0.5, 0.25, 0.125
        for (int tries=0; tries<4; ++tries) {
            const T a = (tries==0)?(T)1.0 : (T)0.5 * (T)pow((T)0.5, tries-1);
            if (tid == 0) {
                for (int i=0;i<N;++i){
                    const double2 lim=c_joint_limits[i];
                    const T xi= x_old[i] + a*dq[i];
                    s_x[i] = fmin(fmax(xi,(T)lim.x),(T)lim.y);
                }
            }
            SYNC();

            if (warp_id == 0) {
                grid::X_warp<T>(s_jointX, s_XmatsHom, s_x, N-1);
            }
            SYNC();

            if (tid == 0) {
                const T pos_new = compute_pos_err_colmajor<T>(s_jointX, tp);
                const T ori_new = compute_ori_err_colmajor<T>(s_jointX, &tp[3]);

                const T* Cn=&s_jointX[(N-1)*16];
                T qee[4]; mat_to_quat_colmajor3x3(Cn, qee);
                if (qee[0]*q_goal[0] + qee[1]*q_goal[1] + qee[2]*q_goal[2] + qee[3]*q_goal[3] < (T)0) {
                    qee[0]=-qee[0]; qee[1]=-qee[1]; qee[2]=-qee[2]; qee[3]=-qee[3];
                }
                T wv[3]; quat_err_rotvec(qee, q_goal, wv);
                T dx = tp[0]-Cn[12], dy = tp[1]-Cn[13], dz = tp[2]-Cn[14];

                T rr[6] = { dx,dy,dz, wv[0],wv[1],wv[2] };
#pragma unroll
                for (int k=0;k<6;++k) rr[k] *= row_s[k];

                const T cp2 = fmax((T)1e-4, (T)5e-3 * (T)fmax((T)1, (T)1e3*pos_new));
                const T co2 = (T)0.5;
#pragma unroll
                for (int k=0;k<3;++k){ 
                    const T a2=fabs(rr[k]); const T w=(a2<=cp2)?(T)1:(cp2/(a2+(T)1e-30)); rr[k]*=sqrt(w); 
                }
#pragma unroll
                for (int k=3;k<6;++k){ 
                    const T a2=fabs(rr[k]); const T w=(a2<=co2)?(T)1:(co2/(a2+(T)1e-30)); rr[k]*=sqrt(w); 
                }

                T trial=(T)0.5*(rr[0]*rr[0]+rr[1]*rr[1]+rr[2]*rr[2]+rr[3]*rr[3]+rr[4]*rr[4]+rr[5]*rr[5]);

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
                // dogleg then 1-D linesearch
                T R;
                if      (pos_err_m > (T)1e-2 || ori_err_rad > (T)0.6)  R=(T)0.45;
                else if (pos_err_m > (T)1e-3 || ori_err_rad > (T)0.25) R=(T)0.28;
                else if (pos_err_m > (T)2e-4 || ori_err_rad > (T)0.08) R=(T)0.10;
                else                                                   R=(T)0.04;

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
                    T kick = (T)0.015 * span * sgn;
                    T xi = s_x[i] + kick;
                    s_x[i] = fmin(fmax(xi,(T)L.x),(T)L.y);
                }
                grid::X_single_thread(s_jointX,s_XmatsHom,s_x,N-1);
                pos_err_m   = compute_pos_err_colmajor<T>(s_jointX,tp);
                ori_err_rad = compute_ori_err_colmajor<T>(s_jointX,&tp[3]);
                stall = 0;
            }

            if (pos_err_m < eps_pos && ori_err_rad < eps_ori) { atomicCAS(&g_stop, 0, 1); s_break = 1; }
        }
        SYNC();
        if (s_break) break;

        if (warp_id == 0) {
            grid::X_warp<T>(s_jointX, s_XmatsHom, s_x, N-1);
        }
        SYNC();

        if (tid == 0) {
            pos_err_m   = compute_pos_err_colmajor<T>(s_jointX, tp);
            ori_err_rad = compute_ori_err_colmajor<T>(s_jointX, &tp[3]);
        }
        SYNC();
    }

WRITE_OUT:

    // drift guard against noisy acceptances
    {
        const T MAX_DRIFT = (T)2e-3;
        if (pos_err_m > best_pos_seen + MAX_DRIFT) {
            for (int i=0; i<N; ++i) s_x[i] = best_x_pos[i];
            if (warp_id == 0) {
                grid::X_warp<T>(s_jointX, s_XmatsHom, s_x, N-1);
            }
            SYNC();
        }
    }

    if (tid == 0) {
        pos_err_m   = compute_pos_err_colmajor<T>(s_jointX, tp);
        ori_err_rad = compute_ori_err_colmajor<T>(s_jointX, &tp[3]);

        const T* Cn = &s_jointX[(N-1)*16];
        T q_out[4]; mat_to_quat_colmajor3x3(Cn, q_out);
        pose[gp*7+0]=Cn[12]; pose[gp*7+1]=Cn[13]; pose[gp*7+2]=Cn[14];
        pose[gp*7+3]=q_out[0]; pose[gp*7+4]=q_out[1]; pose[gp*7+5]=q_out[2]; pose[gp*7+6]=q_out[3];
        pos_error[gp] = pos_err_m * (T)1000.0;
        ori_error[gp] = ori_err_rad;
        for (int i=0;i<N;++i) x[gp*N+i] = s_x[i];
    }

    #undef SYNC
}

template<typename T>
__device__ __forceinline__ void warp_min_reduce_pair(T& e, int& j) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        T   e2 = __shfl_down_sync(FULL_WARP_MASK, e, off);
        int j2 = __shfl_down_sync(FULL_WARP_MASK, j, off);
        if (e2 < e) { e = e2; j = j2; }
    }
}

// COARSE SEARCH
template<typename T>
__global__ void coarse_search(
    T* __restrict__ x,
    T* __restrict__ pose,
    const T* __restrict__ targetsB,
    T* __restrict__ pos_errors,
    T* __restrict__ ori_errors,
    const grid::robotModel<T>* d_robotModel
) {
    const int gp   = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps_per_block = max(1, (int)(blockDim.x >> 5));

    if (!x || !pose || !targetsB || !pos_errors || !ori_errors || !d_robotModel) return;
    if (warps_per_block == 0) return;

    // Two per-warp scratch blocks only: l_tmp (copy buffer) and l_C (computed transforms)
    extern __shared__ __align__(16) unsigned char s_dyn_raw[];
    T* s_dyn = reinterpret_cast<T*>(s_dyn_raw);
    const size_t per_warp_elems = (size_t)(2 * N * 16);
    T* warp_base = s_dyn + (size_t)warp * per_warp_elems;
    T* l_tmp = warp_base;
    T* l_C   = warp_base + (size_t)(N * 16);

    const T  epsilon = (T)20e-3;   // 20 mm
    const T  nu      = (T)(90 * PI / 180.0);
    const int k_max  = 20;

    __shared__ int  s_stop;
    __shared__ int  s_allow_ori;
    __shared__ int  s_last_joint_o, s_last_joint_p;

    __shared__ T s_x[N];
    __shared__ T s_pose[7];
    __shared__ T s_glob_pos_err, s_glob_ori_err;

    __shared__ T s_pos_theta1[N], s_ori_theta1[N];
    __shared__ T s_pos_err[N],    s_ori_err[N];

    __shared__ T s_XmatsHom[N*16];
    __shared__ T s_jointXforms[N*16];
    __shared__ T s_temp[N*2];

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

    grid::load_update_XmatsHom_helpers<T>(s_XmatsHom, s_x, d_robotModel, s_temp);
    __syncthreads();

    if ((threadIdx.x >> 5) == 0) { // warp 0
        grid::X_warp<T>(s_jointXforms, s_XmatsHom, s_x, N - 1);
    }
    __syncthreads();

    if (tid == 0) {
        s_glob_pos_err = compute_pos_err<T>(s_jointXforms, target_pose_local);
        s_glob_ori_err = compute_ori_err<T>(s_jointXforms, q_t);

        T q_ee[4];
        mat_to_quat(&s_jointXforms[(N - 1) * 16], q_ee);
        normalize_quat(q_ee);
        s_pose[0] = s_jointXforms[(N - 1) * 16 + 12];
        s_pose[1] = s_jointXforms[(N - 1) * 16 + 13];
        s_pose[2] = s_jointXforms[(N - 1) * 16 + 14];
        s_pose[3] = q_ee[0]; s_pose[4] = q_ee[1]; s_pose[5] = q_ee[2]; s_pose[6] = q_ee[3];
    }
    __syncthreads();

    for (int k = 0; k < k_max; ++k) {
        if (tid == 0) s_stop = read_stop();
        __syncthreads();
        if (s_stop) break;

        if ((threadIdx.x >> 5) == 0) { // warp 0
            grid::X_warp<T>(s_jointXforms, s_XmatsHom, s_x, N - 1);
        }
        __syncthreads();

        if (tid == 0) {
            T q_ee[4];
            mat_to_quat(&s_jointXforms[(N - 1) * 16], q_ee);
            normalize_quat(q_ee);
            s_pose[0] = s_jointXforms[(N - 1) * 16 + 12];
            s_pose[1] = s_jointXforms[(N - 1) * 16 + 13];
            s_pose[2] = s_jointXforms[(N - 1) * 16 + 14];
            s_pose[3] = q_ee[0]; s_pose[4] = q_ee[1]; s_pose[5] = q_ee[2]; s_pose[6] = q_ee[3];
        }
        __syncthreads();

        if (tid == 0) {
            const T pos_gate = (T)10e-4;
            s_allow_ori = (s_glob_pos_err < pos_gate) ? 1 : 0;
            // s_allow_ori = 1; // always include orientation
        }
        __syncthreads();

        // compute per-joint theta1 for pos & ori
        for (int idx = warp; idx < 2 * N; idx += warps_per_block) {
            const int phase = idx / N;
            const int p     = idx % N;
            if (lane == 0) {
                if (phase == 0) {
                    s_pos_theta1[p] = solve_pos<T>(s_jointXforms, s_pose, target_pose_local, p, k, k_max);
                } else {
                    s_ori_theta1[p] = s_allow_ori ? solve_ori<T>(s_jointXforms, q_t, p, k, k_max) : (T)0;
                }
            }
        }
        __syncthreads();

        // evaluate greedy pairwise (p,j) with two scratch buffers (l_tmp, l_C)
        for (int idx = warp; idx < 2 * N; idx += warps_per_block) {
            const int phase = idx / N;
            const int p     = idx % N;
            const bool pos_phase = (phase == 0);

            T best_err_lane = pos_phase ? s_glob_pos_err : s_glob_ori_err;
            int best_j_lane = -1;

            if (lane == 0) {
                for (int j = 0; j < N; ++j) {
                    // Build candidate vector in registers
                    T cand[N];
                    #pragma unroll
                    for (int m = 0; m < N; ++m) cand[m] = s_x[m];

                    const T delta1 = pos_phase ? s_pos_theta1[p] : s_ori_theta1[p];
                    cand[p] = clamp_val<T>(cand[p] + delta1,
                                           (T)c_joint_limits[p].x, (T)c_joint_limits[p].y);

                    // C1: apply p on top of s_XmatsHom -> l_C
                    #pragma unroll
                    for (int m = 0; m < N * 16; ++m) l_tmp[m] = s_XmatsHom[m];
                    grid::X_single_thread(l_C, l_tmp, cand, N - 1);

                    // Compute theta2 using C1
                    T theta2 = (T)0;
                    if (pos_phase) {
                        const int ee = (N - 1) * 16;
                        T pos1[3] = { l_C[ee + 12], l_C[ee + 13], l_C[ee + 14] };
                        theta2 = solve_pos<T>(l_C, pos1, target_pose_local, j, k, k_max);
                    } else {
                        theta2 = s_allow_ori ? solve_ori<T>(l_C, q_t, j, k, k_max) : (T)0;
                    }

                    cand[j] = clamp_val<T>(cand[j] + theta2,
                                           (T)c_joint_limits[j].x, (T)c_joint_limits[j].y);

                    // C2: reapply full cand (p and j) on top of s_XmatsHom -> l_C
                    #pragma unroll
                    for (int m = 0; m < N * 16; ++m) l_tmp[m] = s_XmatsHom[m];
                    grid::X_single_thread(l_C, l_tmp, cand, N - 1);

                    const T err = pos_phase
                        ? compute_pos_err<T>(l_C, target_pose_local)
                        : compute_ori_err<T>(l_C, q_t);

                    if (err < best_err_lane) { best_err_lane = err; best_j_lane = j; }
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

        // choose best pos and ori joints
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
            grid::X_warp<T>(s_jointXforms, s_XmatsHom, s_x, N - 1);
        }
        __syncthreads();

        if (tid == 0) {
            s_glob_pos_err = compute_pos_err<T>(s_jointXforms, target_pose_local);
            s_glob_ori_err = compute_ori_err<T>(s_jointXforms, q_t);

            T q_ee[4];
            mat_to_quat(&s_jointXforms[(N - 1) * 16], q_ee);
            normalize_quat(q_ee);
            s_pose[0] = s_jointXforms[(N - 1) * 16 + 12];
            s_pose[1] = s_jointXforms[(N - 1) * 16 + 13];
            s_pose[2] = s_jointXforms[(N - 1) * 16 + 14];
            s_pose[3] = q_ee[0]; s_pose[4] = q_ee[1]; s_pose[5] = q_ee[2]; s_pose[6] = q_ee[3];

            for (int jj = 0; jj < N; ++jj) {
                s_pos_err[jj] = s_glob_pos_err;
                s_ori_err[jj] = s_glob_ori_err;
            }

            if (s_glob_pos_err < epsilon && s_glob_ori_err < nu) {
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
    int stop_on_first
) {
    const int B = gridDim.x;
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
        stop_on_first
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
    __shared__ T s_X[N * 16];
    __shared__ T s_tmp[N * 2];

    for (int j = threadIdx.x; j < N; j += blockDim.x)
        s_q[j] = q[b * N + j];
    __syncthreads();

    grid::load_update_XmatsHom_helpers<T>(s_X, s_q, RM, s_tmp);
    __syncthreads();

    if (threadIdx.x == 0) {
        grid::X_single_thread<T>(s_X, s_X, s_q, N - 1);

        if (ee_pose7) {
            const T* Cee = &s_X[(N - 1) * 16];
            T qee[4];
            mat_to_quat_colmajor3x3(Cee, qee);
            ee_pose7[b * 7 + 0] = Cee[12];
            ee_pose7[b * 7 + 1] = Cee[13];
            ee_pose7[b * 7 + 2] = Cee[14];
            ee_pose7[b * 7 + 3] = qee[0];
            ee_pose7[b * 7 + 4] = qee[1];
            ee_pose7[b * 7 + 5] = qee[2];
            ee_pose7[b * 7 + 6] = qee[3];
        }

        if (all_link_T) {
            T* out = &all_link_T[b * (N * 16)];
#pragma unroll
            for (int i = 0; i < N * 16; ++i) out[i] = s_X[i];
        }
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

    scores[i] = pos_err_mm[i] + ori_err_rad[i];
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

struct RefineSchedule {
    float  top_frac;
    int    repeats;
    double sigma_frac;
    bool   keep_one;
};

inline RefineSchedule schedule_for_B(int B) {
    RefineSchedule s; s.keep_one = true;

    if (B <= 8) {           
        s.repeats   = 24;
        s.sigma_frac= 0.25;
        s.top_frac = 1.0;
    } else if (B <= 16) {
        s.repeats   = 16;
        s.sigma_frac= 0.15;
        s.top_frac = 0.5;
    } else if (B <= 128) {
        s.repeats   = 5;
        s.sigma_frac= 0.15;
        s.top_frac = 0.2;
    } else if (B <= 1024) {
        s.repeats   = 5;
        s.sigma_frac= 0.15;
        s.top_frac = 0.02;
    } else if (B <= 2048) {
        s.repeats   = 5;
        s.sigma_frac= 0.15;
        s.top_frac = 0.01;
    } else {
        s.repeats   = 2;
        s.sigma_frac= 0.15;
        s.top_frac = 0.01;
    }
    return s;
}

template <typename Dst, typename Src>
__global__ void cast_array(const Src* __restrict__ in,
                           Dst* __restrict__ out,
                           size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (Dst)in[i];
}

template<typename T>
Result<T> generate_ik_solutions(
    T* target_pose,
    const grid::robotModel<T>* d_robotModel,
    int b_size,
    int num_solutions)
{
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
    #ifdef HAS_RESULT_COUNT_FIELD
        result.count = S;
    #endif
        return result;
    }

    // Coarse phase precision
    using TC = float;

    const int    B            = b_size;
    const size_t num_elems_x  = (size_t)B * N;
    const size_t num_elems_p7 = (size_t)B * 7;

    const grid::robotModel<TC>* d_robotModel_f = grid::init_robotModel<TC>();

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

    // COARSE SEARCH
    {
        // threads-per-block request ~ 2N warps
        int TPB_req = std::min((int)(2 * N * WARP_SIZE), 256);
        int maxThreadsPerBlock = 0;
        CUDA_OK(cudaDeviceGetAttribute(&maxThreadsPerBlock,
                                       cudaDevAttrMaxThreadsPerBlock, 0));
        TPB_req = std::min(TPB_req, maxThreadsPerBlock);
        TPB_req = (TPB_req + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
        TPB_req = std::max(TPB_req, WARP_SIZE);

        const size_t perWarpBytes = (size_t)(2 * N * 16) * sizeof(TC);

        cudaFuncAttributes attr{};
        CUDA_OK(cudaFuncGetAttributes(&attr, (const void*)coarse_search<TC>));
        const size_t staticShmem = (size_t)attr.sharedSizeBytes;

        int maxOptIn=0, maxDefault=0;
        CUDA_OK(cudaDeviceGetAttribute(&maxOptIn,
                                       cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
        CUDA_OK(cudaDeviceGetAttribute(&maxDefault,
                                       cudaDevAttrMaxSharedMemoryPerBlock, 0));
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
                d_pos_mm_c, d_ori_r_c, d_robotModel_f
            );
            cudaError_t e = cudaPeekAtLastError();
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

    const auto  sch         = schedule_for_B(B);
    const float top_frac    = sch.top_frac;
    const int   repeats     = sch.repeats;
    const double sigma_frac = sch.sigma_frac;
    const bool  keep_one    = true;

    // score: pos + ori error (float)
    TC* d_scores_c = nullptr;
    CUDA_OK(cudaMalloc(&d_scores_c, sizeof(TC) * B));
    {
        const int tpb = 256, gpb = (B + tpb - 1) / tpb;
        build_scores_kernel<TC><<<gpb, tpb>>>(
            d_pos_mm_c, d_ori_r_c, d_scores_c, B);
        cudaGetLastError();
    }

    // sort by score and gather top-K seeds
    thrust::device_vector<int> d_idx(B);
    thrust::sequence(d_idx.begin(), d_idx.end(), 0);
    {
        thrust::device_ptr<TC> s_ptr(d_scores_c);
        thrust::sort_by_key(s_ptr, s_ptr + B, d_idx.begin());
    }

    const int K = std::max(1, (int)std::ceil(top_frac * (float)B));
    thrust::device_vector<int> d_top_idx(K);
    thrust::copy(d_idx.begin(), d_idx.begin() + K, d_top_idx.begin());

    TC* d_x_top_c = nullptr;
    CUDA_OK(cudaMalloc(&d_x_top_c, sizeof(TC) * (size_t)K * N));
    {
        const int blocks = K, tpb = 128;
        gather_rows_kernel<TC><<<blocks, tpb>>>(
            d_x_c,
            thrust::raw_pointer_cast(d_top_idx.data()),
            d_x_top_c, K);
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

    // Refined targets in float
    TC* d_targets_refined_c = nullptr;
    CUDA_OK(cudaMalloc(&d_targets_refined_c, sizeof(TC) * 7 * (size_t)Krep));
    {
        const int blocks = Krep, tpb = 32;
        replicate_target7_kernel<TC><<<blocks, tpb>>>(
            d_target7_c, d_targets_refined_c, Krep);
        cudaGetLastError();
    }
    CUDA_OK(cudaDeviceSynchronize());

    // JACOBIAN LM TUNER
    double *dx64=nullptr, *dtgt64=nullptr, *dpose64=nullptr;
    double *dposmm64=nullptr, *dori64=nullptr;
    const size_t KrepN = (size_t)Krep * N;
    const size_t Krep7 = (size_t)Krep * 7;

    CUDA_OK(cudaMalloc(&dx64,    sizeof(double) * KrepN));
    CUDA_OK(cudaMalloc(&dtgt64,  sizeof(double) * Krep7));
    CUDA_OK(cudaMalloc(&dpose64, sizeof(double) * Krep7));
    CUDA_OK(cudaMalloc(&dposmm64,sizeof(double) * Krep));
    CUDA_OK(cudaMalloc(&dori64,  sizeof(double) * Krep));

    // cast float -> double
    {
        const int tpb = 256;
        int gpb = (int)((KrepN + tpb - 1) / tpb);
        cast_array<double, TC><<<gpb, tpb>>>(d_x_rep_c, dx64, KrepN);
        cudaGetLastError();
        CUDA_OK(cudaDeviceSynchronize());
    }

    // build double targets
    double h_target7d[7];
    for (int i = 0; i < 7; ++i)
        h_target7d[i] = static_cast<double>(target_pose[i]);

    double* d_target7_d = nullptr;
    CUDA_OK(cudaMalloc(&d_target7_d, sizeof(double) * 7));
    CUDA_OK(cudaMemcpy(d_target7_d, h_target7d,
                       sizeof(double) * 7,
                       cudaMemcpyHostToDevice));

    {
        const int blocks = Krep;
        const int tpb    = 32;
        replicate_target7_kernel<double><<<blocks, tpb>>>(
            d_target7_d, dtgt64, Krep);
        cudaGetLastError();
        CUDA_OK(cudaDeviceSynchronize());
    }

    auto* d_robotModel64 = grid::init_robotModel<double>();
    {
        int zero = 0, neg1 = -1;
        CUDA_OK(cudaMemcpyToSymbol(g_stop,   &zero, sizeof(int)));
        CUDA_OK(cudaMemcpyToSymbol(g_winner, &neg1, sizeof(int)));
    }

    {
        const int TPB_lm   = 32;
        const int max_iters = 40;
        int stop_on_first  = (num_solutions > 1) ? 0 : 1;

        lm_tuner<double><<<Krep, TPB_lm>>>(
            dx64, dpose64, dtgt64, dposmm64, dori64, d_robotModel64,
            1e-8, 1e-8, 5e-3, max_iters, stop_on_first
        );
        cudaGetLastError();
        CUDA_OK(cudaDeviceSynchronize());
    }

    std::vector<double> h_posmm64(Krep), h_orir64(Krep);
    std::vector<double> h_pose64(Krep7), h_x64(KrepN);
    CUDA_OK(cudaMemcpy(h_posmm64.data(), dposmm64,
                       sizeof(double)*Krep,  cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(h_orir64 .data(), dori64,
                       sizeof(double)*Krep,  cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(h_pose64 .data(), dpose64,
                       sizeof(double)*Krep7, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(h_x64    .data(), dx64,
                       sizeof(double)*KrepN, cudaMemcpyDeviceToHost));

    // GET SOLUTIONS
    const int S_target = std::max(1, num_solutions);
    auto score_ref = [&](int i)->double {
        return h_posmm64[i] + h_orir64[i];
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
        return (double)h_pos_mm_coarse_f[i] + (double)h_ori_rad_coarse_f[i];
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
        // fill with best coarse indexes
        std::vector<int> order_coarse(B);
        std::iota(order_coarse.begin(), order_coarse.end(), 0);
        std::sort(order_coarse.begin(), order_coarse.end(),
                  [&](int a, int b){ return score_coarse(a) < score_coarse(b); });
        for (int cidx : order_coarse) {
            chosen.push_back(-1 - cidx);
            if ((int)chosen.size() == S_target) break;
        }
    }

    // PACK OUTPUTS
    const int S = (int)chosen.size();
    result.pos_errors   = new T[S];
    result.ori_errors   = new T[S];
    result.pose         = new T[7 * S];
    result.joint_config = new T[N * S];
#ifdef HAS_RESULT_COUNT_FIELD
    result.count = S;
#endif

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
    cudaFree(d_targets_refined_c);

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

    auto t1 = high_resolution_clock::now();
    result.elapsed_time =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    return result;
}

template Result<double> generate_ik_solutions<double>(
    double* target_pose,
    const grid::robotModel<double>* d_robotModel,
    int b_size,
    int num_solutions
);

template Result<float> generate_ik_solutions<float>(
    float* target_pose,
    const grid::robotModel<float>* d_robotModel,
    int b_size,
    int num_solutions
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

template grid::robotModel<double>* grid::init_robotModel<double>();
template grid::robotModel<float>* grid::init_robotModel<float>();