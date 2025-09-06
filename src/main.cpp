#include "include/globeik_kernel.h"
#include "include/util.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <thread>
#include <vector>

#include <yaml-cpp/yaml.h>

#define N grid_num_joints()
#define NUM_TESTS 100

int main(int argc, char** argv) {
    std::map<std::string, std::vector<double>> yaml_data;

    // default seeds if not passed
    int num_seeds = 2000;
    if (argc > 1) {
        num_seeds = std::stoi(argv[1]);
    }

    // default: do not print poses
    bool print_poses = false;
    if (argc > 2) {
        print_poses = (std::stoi(argv[2]) != 0);
    }

    auto* d_robotModel = grid::init_robotModel<double>();
    init_joint_limits_constants();

    uint64_t seed = 0ull;
    auto targets = sample_random_target_poses<double>(
        d_robotModel, NUM_TESTS, seed
    );

    for (int i = 0; i < std::min<int>(NUM_TESTS, targets.size()); ++i) {
        double target_pose[7];
        for (int j = 0; j < 7; ++j) {
            target_pose[j] = targets[i][j];
        }

        std::cout << "Running goal " << i << ", batch size " << num_seeds << std::endl;
        for (int j = 0; j < 3; ++j) {
            generate_ik_solutions<double>(target_pose, d_robotModel, num_seeds);
        }

        auto res = generate_ik_solutions<double>(
            target_pose, d_robotModel, num_seeds);

        if (print_poses) {
            std::cout << "Target Pose: [";
            for (int j = 0; j < 7; ++j) {
                std::cout << target_pose[j] << (j < 6 ? ", " : "");
            }
            std::cout << "]\n";

            std::cout << "Returned Pose: [";
            for (int j = 0; j < 7; ++j) {
                std::cout << res.pose[j] << (j < 6 ? ", " : "");
            }
            std::cout << "]\n";
        }

        yaml_data["Batch-Size"].push_back(num_seeds);
        yaml_data["IK-time(ms)"].push_back(res.elapsed_time);
        yaml_data["Pos-Error"].push_back(res.pos_errors[0]);
        yaml_data["Ori-Error"].push_back(res.ori_errors[0]);

        delete[] res.joint_config;
        delete[] res.pose;
        delete[] res.pos_errors;
        delete[] res.ori_errors;
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
