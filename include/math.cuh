/*
    Additional Math Utilities
*/

#pragma once 
#include "hjcd_settings.h"

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
    mat_to_quat(&CjX[(hjcd::N-1)*16], qee);
    if (qee[0]*q_goal[0]+qee[1]*q_goal[1]+qee[2]*q_goal[2]+qee[3]*q_goal[3] < (T)0) {
        qee[0]=-qee[0]; qee[1]=-qee[1]; qee[2]=-qee[2]; qee[3]=-qee[3];
    }
    T wv[3]; quat_err_rotvec(qee, q_goal, wv);
    return sqrt(wv[0]*wv[0] + wv[1]*wv[1] + wv[2]*wv[2]);
}

template<typename T>
__device__ T compute_pos_err(const T* C, const T* target_pose) {
    const T dx = C[(hjcd::N - 1) * 16 + 12] - target_pose[0];
    const T dy = C[(hjcd::N - 1) * 16 + 13] - target_pose[1];
    const T dz = C[(hjcd::N - 1) * 16 + 14] - target_pose[2];
    return sqrt(dx * dx + dy * dy + dz * dz);
}