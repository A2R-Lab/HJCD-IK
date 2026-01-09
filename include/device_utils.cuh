/*
    Device utility functions
*/

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
template<typename T>
__device__ __forceinline__ T clamp_dot(T dot) {
    if (dot > T(1.0)) return T(1.0);
    if (dot < T(-1.0)) return T(-1.0);
    return dot;
}

template<typename T>
__device__ __forceinline__ T clamp_unit(T v) { 
    return v > (T)1 ? (T)1 : (v < (T)-1 ? (T)-1 : v); 
}

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
    for (int i = 0; i < N; ++i) {
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
