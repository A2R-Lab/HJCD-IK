#pragma once
#include <cmath>
#include <limits>
#include <utility>

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
}  // namespace hjcd
