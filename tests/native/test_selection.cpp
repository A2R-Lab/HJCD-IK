#include "kernel/selection_key.h"
#include <algorithm>
#include <cassert>
#include <limits>
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
}
