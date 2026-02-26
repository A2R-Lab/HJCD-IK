
#pragma once 
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
    s.repeats    = 5;

    if (B <= 12) {
        s.top_k     = B;
        s.repeats   = 12;
        s.sigma_frac= 0.25;
    } else {
        s.top_k = 32;
    }

    return s;
}

template<typename T>
struct HJCDSettings {
    // Coarse phase settings
    static constexpr T epsilon = static_cast<T>(20e-3);   // 20 mm
    static constexpr T  nu = static_cast<T>(90 * PI / 180.0);
    static constexpr int k_max  = 20;

    // Refine phase settings
    static constexpr T lambda_init = static_cast<T>(5e-3);
};
