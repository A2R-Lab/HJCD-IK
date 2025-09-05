#include "util.h"
#include <iostream>
#include <yaml-cpp/yaml.h>

std::vector<std::array<double, 7>> load_dataset(const std::string& filename) {
    YAML::Node config = YAML::LoadFile(filename);
    YAML::Node goals = config["goals"];

    if (!goals || !goals.IsSequence()) {
        std::cerr << "Error: 'goals' not found or not a sequence\n";
        return {};
    }

    std::vector<std::array<double, 7>> goal_poses;
    for (std::size_t i = 0; i < goals.size(); ++i) {
        auto pos = goals[i]["position"];
        auto quat_xyzw = goals[i]["quaternion"];
        if (!pos || !quat_xyzw || pos.size() != 3 || quat_xyzw.size() != 4) {
            std::cerr << "Invalid goal at index " << i << "\n";
            continue;
        }
        std::array<double, 7> pose{};
        for (int j = 0; j < 3; ++j) pose[j] = pos[j].as<double>();
        pose[3] = quat_xyzw[3].as<double>();
        pose[4] = quat_xyzw[0].as<double>();
        pose[5] = quat_xyzw[1].as<double>();
        pose[6] = quat_xyzw[2].as<double>();
        goal_poses.push_back(pose);
    }

    if (goal_poses.empty()) {
        std::cerr << "No valid poses loaded from " << filename << "\n";
    }
    return goal_poses;
}
