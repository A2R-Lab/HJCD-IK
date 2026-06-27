#pragma once 

// Generated GRiD headers call this helper while defining template bodies.
template<typename T>
__device__ __forceinline__
void mat4_mul(const T* A, const T* B, T* C) {
    T tmp[16];

    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            tmp[c * 4 + r] =
                A[0 * 4 + r] * B[c * 4 + 0] +
                A[1 * 4 + r] * B[c * 4 + 1] +
                A[2 * 4 + r] * B[c * 4 + 2] +
                A[3 * 4 + r] * B[c * 4 + 3];
        }
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        C[i] = tmp[i];
    }
}

#include "grid.cuh"

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef UNREFINE
#define UNREFINE 0
#endif

#ifndef FULL_WARP_MASK
#define FULL_WARP_MASK 0xFFFFFFFFu
#endif

#ifndef PI
#define PI 3.14159265358979323846
#endif

namespace hjcd {
    static constexpr int N = grid::NUM_JOINTS;
}

struct RefineSchedule {
    int    top_k;
    int    repeats;
    double sigma_frac;
    bool   keep_one;
};

inline RefineSchedule schedule_for_B(int B) {
    RefineSchedule s;
    s.keep_one   = true;
    s.sigma_frac = 0.1;
    s.repeats    = 16;

    if (B <= 16) {
        s.top_k     = B;
        s.repeats   = 20;
        s.sigma_frac= 0.25;
    } else if (B >= 2000) {
        s.top_k  = 24;
        s.repeats = 16;
    } else if (B >= 1000) {
        s.top_k  = 20;
        s.repeats = 22;
    } else {
        s.top_k   = 16 + (int)((B - 1000)/1000 * 8);
        s.repeats = 18;
    }

    return s;
}

template<typename T>
struct HJCDSettings {
    // Coarse phase settings
    static constexpr T epsilon = static_cast<T>(20e-3);   // 20 mm
    static constexpr T nu = static_cast<T>(90 * PI / 180.0);
    static constexpr int k_max  = 20;

    // Refine phase settings
    static constexpr T lambda_init = static_cast<T>(5e-3);
};
