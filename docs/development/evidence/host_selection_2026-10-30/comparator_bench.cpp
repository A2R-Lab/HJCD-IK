// CPU-only microbenchmark of the host candidate sort comparator: reference (scalar score),
// candidate (pair key, csrc/kernel/selection_key.h), and precomputed keys.
#include "kernel/selection_key.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

template<typename T>
static inline T rank_score(T pos_err_mm, T ori_err_rad) {
    constexpr T ORI_TARGET_RAD = (T)1.1e-4;
    constexpr T ORI_OUTLIER_W  = (T)1e4;
    const T ori_excess = ori_err_rad - ORI_TARGET_RAD;
    return pos_err_mm + ORI_OUTLIER_W * (ori_excess > (T)0 ? ori_excess : (T)0);
}

int main(int argc, char** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 200;
    const double conv_frac = argc > 2 ? atof(argv[2]) : 0.6;
    std::mt19937_64 rng(42);
    for (int Krep : {512, 768, 2304, 8448}) {
        std::vector<double> h_posmm64(Krep), h_orir64(Krep);
        std::vector<float> env(Krep, 0.f);
        std::uniform_real_distribution<double> u(0, 1);
        std::lognormal_distribution<double> lp(-8, 2), lo(-12, 2), lpu(0.5, 1), lou(-5, 1);
        for (int i = 0; i < Krep; ++i) {
            if (u(rng) < conv_frac) { h_posmm64[i] = std::min(lp(rng), 0.99); h_orir64[i] = std::min(lo(rng), 9e-4); }
            else { h_posmm64[i] = lpu(rng); h_orir64[i] = lou(rng); }
        }
        volatile bool use_soft_v = false, warm_v = false;
        const bool use_soft = use_soft_v, warm = warm_v;
        constexpr double WARM_POS_MM = 1.0, WARM_ORI_RAD = 1e-3;
        auto pos_mm  = [&](int idx)->double { return h_posmm64[idx]; };
        auto ori_rad = [&](int idx)->double { return h_orir64[idx]; };
        auto env_cost = [&](int idx)->float { return env[idx]; };
        auto init_distance = [&](int)->double { return 0.; };
        auto score_old = [&](int idx)->double {
            double s = rank_score(pos_mm(idx), ori_rad(idx));
            if (use_soft) s += 1e3 * (double)env_cost(idx);
            if (!std::isfinite(s)) return std::numeric_limits<double>::infinity();
            if (warm) {
                const bool converged = pos_mm(idx) < WARM_POS_MM && ori_rad(idx) < WARM_ORI_RAD;
                return converged ? init_distance(idx) : 1e6 + s;
            }
            return s;
        };
        auto score_new = [&](int idx)->hjcd::SelectionKey {
            double s = rank_score(pos_mm(idx), ori_rad(idx));
            if (use_soft) s += 1e3 * (double)env_cost(idx);
            const bool converged = pos_mm(idx) < WARM_POS_MM && ori_rad(idx) < WARM_ORI_RAD;
            return hjcd::selection_key(warm, converged, s, warm && converged ? init_distance(idx) : 0.);
        };
        auto time = [&](auto&& sorter, const char* name) {
            std::vector<double> us;
            std::vector<int> pool(Krep);
            long long chk = 0;
            for (int r = 0; r < reps; ++r) {
                std::iota(pool.begin(), pool.end(), 0);
                std::shuffle(pool.begin(), pool.end(), rng);
                auto t0 = std::chrono::steady_clock::now();
                sorter(pool);
                auto t1 = std::chrono::steady_clock::now();
                us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
                chk += pool[0] + pool[Krep / 2];
            }
            std::sort(us.begin(), us.end());
            printf("Krep=%5d %-12s median %8.2f us  p10 %8.2f  p90 %8.2f  (chk %lld)\n", Krep, name,
                   us[us.size() / 2], us[us.size() / 10], us[us.size() * 9 / 10], chk);
            return us[us.size() / 2];
        };
        double a = time([&](std::vector<int>& p) { std::sort(p.begin(), p.end(), [&](int x, int y) { return score_old(x) < score_old(y); }); }, "reference");
        double b = time([&](std::vector<int>& p) { std::sort(p.begin(), p.end(), [&](int x, int y) { return score_new(x) < score_new(y); }); }, "candidate");
        std::vector<hjcd::SelectionKey> keys(Krep);
        double c = time([&](std::vector<int>& p) {
            for (int i : p) keys[i] = score_new(i);
            std::sort(p.begin(), p.end(), [&](int x, int y) { return keys[x] < keys[y]; }); }, "precomputed");
        printf("   candidate-reference %+.2f us, precomputed-reference %+.2f us\n", b - a, c - a);
        // Equivalence: same permutation from on-the-fly pair keys and precomputed keys; stable tie classes.
        std::vector<int> p1(Krep), p2(Krep);
        std::iota(p1.begin(), p1.end(), 0); std::shuffle(p1.begin(), p1.end(), rng); p2 = p1;
        std::sort(p1.begin(), p1.end(), [&](int x, int y) { return score_new(x) < score_new(y); });
        for (int i : p2) keys[i] = score_new(i);
        std::sort(p2.begin(), p2.end(), [&](int x, int y) { return keys[x] < keys[y]; });
        printf("   permutation identical: %s\n", p1 == p2 ? "yes" : "NO");
    }
}
