#include "hjcd_settings.h"

__device__ int g_stop = 0;
__device__ int g_winner = -1;

__device__ __forceinline__ int read_stop() {
    return atomicAdd(&g_stop, 0);
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

// ---------------------------------------------------------------------------
// SEMANTIC RNG (rng_policy_version = semantic_problem_rng_v2).
//
// Checkpoint 5D.14c. The legacy `make_seed` below mixes blockIdx/threadIdx and the caller's
// OUTER BATCH SLOT into random identity, so a problem's stochastic stream changed when it moved
// to a different slot or when the batch size P changed. Measured: reversing an 8-problem batch
// altered 6/8 results. That made execution scheduling semantically load-bearing.
//
// The contract now is: same semantic problem + same per-problem seed + same policy => same
// stream, independent of P, slot, ordering, partitioning and CUDA grid geometry. Physical
// indices may still LOCATE data; they may not IDENTIFY a random stream.
enum : uint32_t {
    RNG_SUB_INITIAL_JOINT_SAMPLE      = 0x1u,
    RNG_SUB_PO_CCD_STALL_PERTURBATION = 0x2u,
    RNG_SUB_PER_THREAD_INITIAL_CONFIG = 0x3u,
    RNG_SUB_HARD_MODE_RESEED          = 0x4u,
    RNG_SUB_HARD_MODE_PERTURBATION    = 0x5u,
    RNG_SUB_PJ_IK_RANDOM_FALLBACK     = 0x6u,
};

// Every argument is a SEMANTIC index. `sample` MUST be the index local to the problem (s), never
// the flattened p*S+s. Odd 32-bit multipliers keep neighbouring indices from colliding, and the
// final wanghash avalanches the mix.
__device__ __forceinline__ uint32_t semantic_rng(
    uint32_t problem_seed,
    uint32_t substream,
    uint32_t sample,
    uint32_t iteration,
    uint32_t joint_or_dim,
    uint32_t draw
) {
    uint32_t h = problem_seed;
    h = wanghash(h ^ (substream    * 0x9E3779B9u));
    h = wanghash(h ^ (sample       * 0x85EBCA6Bu));
    h = wanghash(h ^ (iteration    * 0xC2B2AE35u));
    h = wanghash(h ^ (joint_or_dim * 0x27D4EB2Du));
    h = wanghash(h ^ (draw         * 0x165667B1u));
    return h;
}

// Stable 64 -> 32 reduction for callers whose semantic seed is 64-bit. Explicit, never an
// implicit narrowing cast.
__device__ __host__ __forceinline__ uint32_t seed64_to_32(unsigned long long s) {
    return (uint32_t)(s ^ (s >> 32));
}

// LEGACY (rng_policy_version = legacy_slot_rng_v1). Slot- and launch-geometry-dependent.
// Retained only for paths not yet converted; must not be used by the multi-problem solver.
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

// MATH HELPER FUNCTIONS
// clamp-to-[-1,1] (acos/asin guards) now comes from GLASS: glass::clamp_unit
// (robotics-ops wave, lie/angle.cuh; tier-free host+device).

template<typename T>
__device__ __forceinline__ T clamp_val(T v, T lo, T hi) {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

template<typename T>
__device__ __forceinline__ T clamp_step_angle(T step_rad) {
    const T MAX_STEP = (T)(15.0 * PI / 180.0);
    if (step_rad > MAX_STEP) step_rad = MAX_STEP;
    if (step_rad < -MAX_STEP) step_rad = -MAX_STEP;
    return step_rad;
}

template<typename T>
__device__ __forceinline__
void clamp_into_limits(const T* xbase, const T* step, T* xout, const double2* limits) {
#pragma unroll
    for (int i = 0; i < hjcd::N; ++i) {
        const double2 L = limits[i];
        const T xi = xbase[i] + step[i];
        xout[i] = fmin(fmax(xi, (T)L.x), (T)L.y);
    }
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
__device__ __forceinline__ T sqr(T x){ 
    return x*x; 
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