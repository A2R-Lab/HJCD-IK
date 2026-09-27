#pragma once
// CPU-only, bounded cache for the last collision problem document.
// Callers serialize access and own any uploaded device environments separately.
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>

namespace hjcd_env {

class ProblemDocument {
public:
    // Exact bytes, not a pointer or hash identity. A hit neither allocates nor reparses.
    // Parse/allocation failures leave the previous document and its identity intact.
    bool update(std::string_view text) {
        if (ready_ && std::string_view(source_) == text) return false;
        auto parsed = nlohmann::json::parse(text.begin(), text.end());
        std::string source(text);
        source_.swap(source);
        root_.swap(parsed);
        ready_ = true;
        return true;
    }

    // The reference remains valid until the next successful document replacement.
    const nlohmann::json& select(const std::string& set, int index) const {
        const auto& problems = root_.at("problems").at(set);
        if (!problems.is_array()) throw std::runtime_error("problem set is not an array");
        if (index < 0 || static_cast<std::size_t>(index) >= problems.size())
            throw std::runtime_error("problem_idx out of range");
        return problems[static_cast<std::size_t>(index)];
    }

private:
    std::string source_;
    nlohmann::json root_;
    bool ready_ = false;
};

}  // namespace hjcd_env
