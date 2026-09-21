#pragma once
#include <array>
#include <vector>
#include <string>
#include <cstdint>
#include <utility>

namespace grid {
    template<typename T> struct robotModel;
    template<typename T> robotModel<T>* init_robotModel();
}

#define PI 3.14159265358979323846

template<typename T>
struct Result {
    T* joint_config = nullptr;
    T* pose = nullptr;
    T* pos_errors = nullptr;
    T* ori_errors = nullptr;
    T elapsed_time{};
    int count = 0;

    Result() = default;
    ~Result() { reset(); }
    Result(const Result&) = delete;
    Result& operator=(const Result&) = delete;

    Result(Result&& other) noexcept { *this = std::move(other); }
    Result& operator=(Result&& other) noexcept {
        if (this != &other) {
            reset();
            joint_config = std::exchange(other.joint_config, nullptr);
            pose = std::exchange(other.pose, nullptr);
            pos_errors = std::exchange(other.pos_errors, nullptr);
            ori_errors = std::exchange(other.ori_errors, nullptr);
            elapsed_time = other.elapsed_time;
            count = other.count;
            other.elapsed_time = T{};
            other.count = 0;
        }
        return *this;
    }

    void reset() noexcept {
        delete[] joint_config;
        delete[] pose;
        delete[] pos_errors;
        delete[] ori_errors;
        joint_config = pose = pos_errors = ori_errors = nullptr;
        elapsed_time = T{};
        count = 0;
    }
};

// RT = LM-refine compute precision (speed/accuracy knob): RT=double (default) is full fp64;
// RT=float runs FK/Jacobian/residual/line-search in fp32 with the Cholesky solve still fp64.
template<typename T, typename RT = double>
Result<T> generate_ik_solutions(
    T* target_pose,
    const grid::robotModel<T>* d_robotModel,
    int b_size,
    int num_solutions = 1,
    bool collision_free = false,
    const char* problems_json_text = nullptr,
    const char* problem_set_name = nullptr,
    int problem_idx = 0,
    bool write_stats = false,
    int collision_mode = -1
);

template<typename T>
std::vector<std::array<T, 7>> sample_random_target_poses(
    const grid::robotModel<T>* d_robotModel,
    int num_configs,
    std::uint64_t seed
);

void init_joint_limits_constants();

void init_joint_limits_from_grid();

extern "C" int grid_num_joints();
extern "C" bool grid_has_collision();