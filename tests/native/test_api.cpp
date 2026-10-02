#include "kernel/hjcd_kernel.h"

#include <future>
#include <iostream>
#include <type_traits>

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<typename T>
static void check_result(const Result<T>& result, const std::array<T, 7>& target, int requested) {
    require(result.count > 0 && result.count <= requested, "invalid result count");
    const int dof = grid_num_joints();
    for (int row = 0; row < result.count; ++row) {
        double squared_error = 0;
        for (int j = 0; j < dof; ++j)
            require(std::isfinite(result.joint_config[row * dof + j]), "nonfinite joint configuration");
        for (int j = 0; j < 7; ++j)
            require(std::isfinite(result.pose[row * 7 + j]), "nonfinite returned pose");
        for (int j = 0; j < 3; ++j) {
            const double delta = double(result.pose[row * 7 + j]) - target[j];
            squared_error += delta * delta;
        }
        require(std::isfinite(result.pos_errors[row]) && std::isfinite(result.ori_errors[row]),
                "nonfinite pose error");
        require(std::abs(1000 * std::sqrt(squared_error) - result.pos_errors[row]) < 0.01,
                "position error disagrees with returned pose (millimeters)");
    }
}

int main(int argc, char** argv) try {
    static_assert(!std::is_copy_constructible_v<Result<double>>);
    static_assert(std::is_nothrow_move_constructible_v<Result<double>>);
    static_assert(std::is_nothrow_move_assignable_v<Result<double>>);
    const int dof = grid_num_joints();
    require(dof > 0 && dof <= 32, "invalid compiled joint count");
    if (argc > 1) require(dof == std::stoi(argv[1]), "wrong robot header selected");

    // Exercise native validation and lazy initialization, with no wrapper or GIL.
    try {
        generate_ik_solutions<double>(nullptr, 32);
        throw std::runtime_error("null target accepted");
    } catch (const std::invalid_argument&) {}
    auto targets = sample_random_target_poses<double>(nullptr, 2, 17);
    require(targets.size() == 2, "sampling failed");

    auto result = generate_ik_solutions<double>(targets[0].data(), 2000, 2);
    check_result(result, targets[0], 2);
    auto* original = result.joint_config;
    Result<double> moved(std::move(result));
    require(result.count == 0 && result.joint_config == nullptr, "move source still owns buffers");
    require(moved.joint_config == original, "move copied or lost result buffers");
    result = std::move(moved);
    require(moved.count == 0 && moved.joint_config == nullptr, "move assignment retained ownership");
    check_result(result, targets[0], 2);

    // Different native template instantiations must share the same solver lock.
    auto fp32 = std::async(std::launch::async, [target = targets[0]]() mutable {
        return generate_ik_solutions<double, float>(target.data(), 512);
    });
    auto fp64 = std::async(std::launch::async, [target = targets[1]]() mutable {
        return generate_ik_solutions<double>(target.data(), 512);
    });
    check_result(fp32.get(), targets[0], 1);
    check_result(fp64.get(), targets[1], 1);

    auto float_targets = sample_random_target_poses<float>(nullptr, 1, 19);
    auto float_result = generate_ik_solutions<float>(float_targets[0].data(), 512);
    check_result(float_result, float_targets[0], 1);
    if (!grid_has_collision()) {
        bool rejected = false;
        try {
            generate_ik_solutions<double>(targets[0].data(), 32, 1, true,
                "{\"problems\":{\"empty\":[{\"obstacles\":{}}]}}", "empty");
        } catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "collision request accepted by a no-collision build");
    }
    std::cout << "native API passed: dof=" << dof
              << ", collision=" << grid_has_collision() << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
