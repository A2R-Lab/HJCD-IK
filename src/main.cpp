#include "include/hjcd_kernel.h"
#include "include/util.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <yaml-cpp/yaml.h>

#define N grid_num_joints()
#define NUM_TESTS 100

static inline std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

static std::vector<int> parse_seeds_spec(const std::string& spec) {
    std::vector<int> seeds;

    if (spec.find(',') != std::string::npos) {
        for (const auto& tok : split(spec, ',')) seeds.push_back(std::stoi(tok));
        return seeds;
    }

    if (spec.find(':') != std::string::npos) {
        auto parts = split(spec, ':');
        if (parts.size() < 2 || parts.size() > 3) {
            throw std::runtime_error("Bad seeds range. Use start:end[:step]");
        }
        int start = std::stoi(parts[0]);
        int end   = std::stoi(parts[1]);
        int step  = (parts.size() == 3) ? std::stoi(parts[2]) : (end - start);
        if (step <= 0) throw std::runtime_error("Range step must be > 0");
        if (end < start) throw std::runtime_error("Range end must be >= start");
        for (int v = start; v <= end; v += step) seeds.push_back(v);
        return seeds;
    }

    // single integer
    seeds.push_back(std::stoi(spec));
    return seeds;
}

// ---- main ------------------------------------------------------------------

int main(int argc, char** argv) {
    std::map<std::string, std::vector<double>> yaml_data;

    std::vector<int> seeds_list = {1, 10, 100, 1000, 2000, 10000};
    if (argc > 1) {
        try {
            seeds_list = parse_seeds_spec(argv[1]);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing seeds spec: " << e.what() << "\n";
            return 1;
        }
    }

    bool print_poses = false;
    if (argc > 2) {
        print_poses = (std::stoi(argv[2]) != 0);
    }

    auto* d_robotModel = grid::init_robotModel<double>();
    init_joint_limits_constants();

    uint64_t seed = 0ull;
    auto targets = sample_random_target_poses<double>(d_robotModel, NUM_TESTS, seed);

    for (int i = 0; i < std::min<int>(NUM_TESTS, targets.size()); ++i) {
        double target_pose[7];
        for (int j = 0; j < 7; ++j) target_pose[j] = targets[i][j];

        for (int num_seeds : seeds_list) {
            std::cout << "Running goal " << i << ", batch size " << num_seeds << std::endl;

            for (int j = 0; j < 3; ++j) {
                generate_ik_solutions<double>(target_pose, d_robotModel, num_seeds);
            }

            auto res = generate_ik_solutions<double>(target_pose, d_robotModel, num_seeds);

            if (print_poses) {
                std::cout << "Target Pose: [";
                for (int j = 0; j < 7; ++j) std::cout << target_pose[j] << (j < 6 ? ", " : "");
                std::cout << "]\n";

                std::cout << "Returned Pose: [";
                for (int j = 0; j < 7; ++j) std::cout << res.pose[j] << (j < 6 ? ", " : "");
                std::cout << "]\n";
            }

            yaml_data["Batch-Size"].push_back(static_cast<double>(num_seeds));
            yaml_data["IK-time(ms)"].push_back(res.elapsed_time);
            yaml_data["Pos-Error"].push_back(res.pos_errors[0]);
            yaml_data["Ori-Error"].push_back(res.ori_errors[0]);

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
        for (const auto& val : kv.second) out << val;
        out << YAML::EndSeq;
    }
    out << YAML::EndMap;

    std::ofstream yaml_file("hjcd_ik_per_batch.yml");
    yaml_file << out.c_str();
    yaml_file.close();

    return 0;
}
