#pragma once
#include <array>
#include <vector>
#include <string>
#include <cstdint>

namespace grid {
    template<typename T> struct robotModel;
    template<typename T> robotModel<T>* init_robotModel();
}

#define PI 3.14159265358979323846

template<typename T>
struct Result {
    T* joint_config;
    T* pose;
    T* pos_errors;
    T* ori_errors;
    T  elapsed_time;
};

template<typename T>
Result<T> generate_ik_solutions(
    T* target_pose,
    const grid::robotModel<T>* d_robotModel,
    int b_size
);

template<typename T>
std::vector<std::array<T, 7>> sample_random_target_poses(
    const grid::robotModel<T>* d_robotModel,
    int num_configs,
    std::uint64_t seed
);

void init_joint_limits_constants();

extern "C" int grid_num_joints();