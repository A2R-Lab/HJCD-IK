#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace hjcd {
using SelectionKey = std::pair<int, double>;

// Convergence is a category, not a finite offset that a joint-space distance can exceed.
// Invalid candidates sort last and are excluded by the caller. Equal keys remain equivalent.
inline SelectionKey selection_key(bool warm, bool converged, double score, double distance) {
    const double value = warm && converged ? distance : score;
    if (!std::isfinite(score) || !std::isfinite(value))
        return {2, std::numeric_limits<double>::infinity()};
    return {warm && !converged ? 1 : 0, value};
}

// Positions 0..n-1 ordered by their keys. std::sort decides only through the comparator, so for a
// pure key this is exactly the permutation std::sort(pool, score(a) < score(b)) produces while
// evaluating each key once instead of once per comparison.
template <typename Key>
inline std::vector<int> sorted_positions(const std::vector<Key>& keys) {
    std::vector<int> order(keys.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = (int)i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return keys[a] < keys[b]; });
    return order;
}
}  // namespace hjcd
