#pragma once
#include <cuda_runtime.h>

#include "Robots.hh"
#include "utils.cuh"
#include "environment.hh"
#include "robots/panda.cuh"

__device__ __forceinline__ void clear_link_CC(int* link_CC) {
    for (int i = threadIdx.x; i < 640; i += blockDim.x) link_CC[i] = 0;
    __syncthreads();
}

namespace pRRTC {

template <typename Robot>
__device__ __forceinline__
bool collision_free_cfg(
    const float* q,
    ppln::collision::Environment<float>* env,
    volatile float* sphere_pos,
    volatile float* sphere_pos_approx,
    volatile int*   link_CC,
    float*          Tbuf,
    int tid)
{
    // approx FK
    ppln::collision::fk_approx<Robot>(q, sphere_pos_approx, Tbuf, tid);
    __syncthreads();

    // Build a union mask of joints implicated by the approximate checks.
    // The Panda exact checks use this mask to narrow the expensive pass.
    clear_link_CC((int*)link_CC);
    bool env_ok_approx  = ppln::collision::env_collision_check_approx<Robot>(
        (float*)sphere_pos_approx, (int*)link_CC, env, tid);

    bool self_ok_approx = ppln::collision::self_collision_check_approx<Robot>(
        (float*)sphere_pos_approx, (int*)link_CC, tid);

    bool ok = env_ok_approx && self_ok_approx;
    __syncthreads();

    // confirm only if approx flagged collision
    if (!ok) {
        ppln::collision::fk<Robot>(q, sphere_pos, Tbuf, tid);
        __syncthreads();

        bool env_ok  = ppln::collision::env_collision_check<Robot>(
            (float*)sphere_pos, (int*)link_CC, env, tid);

        bool self_ok = ppln::collision::self_collision_check<Robot>(
            (float*)sphere_pos, (int*)link_CC, tid);

        ok = env_ok && self_ok;
        __syncthreads();
    }

    return ok;
}

__global__ void mark_collisions_panda(
    const double* __restrict__ q_in,
    int K,
    unsigned char* __restrict__ valid,
    ppln::collision::Environment<float>* env);

void setup_environment_on_device(
    ppln::collision::Environment<float>*& d_env,
    const ppln::collision::Environment<float>& h_env);

void cleanup_environment_on_device(
    ppln::collision::Environment<float>* d_env,
    const ppln::collision::Environment<float>& h_env);

}

