#include "kernel/selection_key.h"
#include <algorithm>
#include <cassert>
#include <limits>
#include <random>
#include <vector>

int main() {
    using hjcd::selection_key;
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const auto far = selection_key(true, true, .1, 1e300);
    const auto inaccurate = selection_key(true, false, 0., 0.);
    assert(far < inaccurate); // Valid prismatic distance must never override convergence class.
    assert(selection_key(true, true, .2, 0.) < far);
    assert(selection_key(true, false, 2., 0.) < selection_key(true, false, 3., 0.));
    assert(selection_key(false, false, 2., inf) < selection_key(false, true, 3., 0.));
    assert(inaccurate < selection_key(true, true, nan, 0.));
    assert(inaccurate < selection_key(true, true, 0., inf));
    auto tied = selection_key(true, true, .3, 1e300);
    assert(!(far < tied) && !(tied < far));
    std::vector<hjcd::SelectionKey> keys{inaccurate, far, tied,
        selection_key(true, true, inf, 0.), selection_key(true, true, nan, 1.)};
    std::sort(keys.begin(), keys.end());
    for (const auto& a : keys) for (const auto& b : keys) for (const auto& c : keys) {
        assert(!(a < a));
        if (a < b && b < c) assert(a < c);
        if (!(a < b) && !(b < a) && !(b < c) && !(c < b))
            assert(!(a < c) && !(c < a));
    }
    // sorted_positions must reproduce the per-comparison sort exactly, ties and nonfinite included,
    // because the host selects the first S distinct valid candidates in that order.
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u01(0., 1.);
    for (int trial = 0; trial < 200; ++trial) {
        const int n = 1 + (int)(u01(rng) * 600);
        const bool warm = u01(rng) < .5;
        std::vector<hjcd::SelectionKey> pool;
        for (int i = 0; i < n; ++i) {
            const double r = u01(rng);
            const double score = r < .05 ? nan : r < .1 ? inf : (double)(int)(u01(rng) * 8) * .25;
            const bool converged = u01(rng) < .6;
            const double distance = r < .15 ? inf : (double)(int)(u01(rng) * 4);
            pool.push_back(selection_key(warm, converged, score, distance));
        }
        std::vector<int> direct(n);
        for (int i = 0; i < n; ++i) direct[i] = i;
        std::shuffle(direct.begin(), direct.end(), rng);
        std::vector<hjcd::SelectionKey> shuffled(n);
        for (int i = 0; i < n; ++i) shuffled[i] = pool[direct[i]];
        std::vector<int> expect(n);
        for (int i = 0; i < n; ++i) expect[i] = i;
        std::sort(expect.begin(), expect.end(), [&](int a, int b) { return shuffled[a] < shuffled[b]; });
        assert(hjcd::sorted_positions(shuffled) == expect);
    }
}
