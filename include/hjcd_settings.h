/*
    Settings for HJCD-IK algorithm
*/

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

template<typename T>
struct HJCDSettings {
    // Coarse phase settings
    static constexpr T epsilon = static_cast<T>(20e-3);   // 20 mm
    static constexpr T  nu = static_cast<T>(90 * PI / 180.0);
    static constexpr int k_max  = 20;

    // Refine phase settings
    static constexpr T lambda_init = static_cast<T>(5e-3);
};

