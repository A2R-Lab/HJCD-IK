#include "include/globeik_kernel.h"
#include "include/util.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <thread>

#include <yaml-cpp/yaml.h>

#define N grid_num_joints()
#define NUM_TESTS 100

int main() {
    std::map<std::string, std::vector<float>> yaml_data;

    std::vector<int> num_seeds = { 1 };

    auto* d_robotModel = grid::init_robotModel<float>();
    init_joint_limits_constants();

    uint64_t seed = 0ull;
    auto targets = sample_random_target_poses<float>(
        d_robotModel, NUM_TESTS, seed
    );

    for (int i = 0; i < std::min<int>(NUM_TESTS, targets.size()); ++i) {
        float target_pose[7];
        for (int j = 0; j < 7; ++j) {
            target_pose[j] = targets[i][j];
        }

        for (auto& n_seed : num_seeds) {
            std::cout << "Running goal " << i << ", batch size " << n_seed << std::endl;
            for (int j = 0; j < 5; ++j) {
                generate_ik_solutions<float>(target_pose, d_robotModel, n_seed);
            }

            auto res = generate_ik_solutions<float>(
                target_pose, d_robotModel, n_seed);

            yaml_data["Batch-Size"].push_back(n_seed);
            yaml_data["IK-time(ms)"].push_back(res.elapsed_time);
            yaml_data["Pos-Error"].push_back(res.pos_errors[0]);
            yaml_data["Ori-Error"].push_back(res.ori_errors[0]);

            // cleanup
            delete[] res.joint_config;
            delete[] res.pose;
            delete[] res.pos_errors;
            delete[] res.ori_errors;
        }
    }

    YAML::Emitter out;
    out << YAML::BeginMap;

    for (const auto& kv : yaml_data) {
        out << YAML::Key << kv.first << YAML::Value << YAML::BeginSeq;
        for (const auto& val : kv.second) {
            out << val;
        }
        out << YAML::EndSeq;
    }

    out << YAML::EndMap;

    std::ofstream yaml_file("globeik_ik_per_batch.yml");
    yaml_file << out.c_str();
    yaml_file.close();

    return 0;
}