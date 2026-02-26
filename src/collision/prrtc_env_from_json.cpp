#include "src/collision/prrtc_env_from_json.hpp"

#include <algorithm>
#include <vector>
#include "src/collision/factory.hh"

using json = nlohmann::json;
using namespace ppln::collision;

namespace pRRTC {

Environment<float> problem_dict_to_env(const json& problem, const std::string& name) {
    Environment<float> env{};

    std::vector<Sphere<float>> spheres;
    std::vector<Capsule<float>> capsules;
    std::vector<Cuboid<float>> cuboids;

    // spheres
    if (problem.contains("sphere")) {
        for (const auto& obj : problem["sphere"]) {
            const json& position = obj["position"];
            Sphere<float> sphere(position[0], position[1], position[2], obj["radius"]);
            sphere.name = obj["name"];
            spheres.push_back(sphere);
        }
    }

    // cylinders
    if (problem.contains("cylinder")) {
        if (name == "box") {
            for (const auto& obj : problem["cylinder"]) {
                const json& position = obj["position"];
                const json& orientation = obj["orientation_euler_xyz"];
                const float radius = obj["radius"];
                const std::array<float, 3> dims = {radius, radius, radius/2.0f};

                auto cuboid = factory::cuboid::array(position, orientation, dims);
                cuboid.name = obj["name"];
                cuboids.push_back(cuboid);
            }
        } else {
            for (const auto& obj : problem["cylinder"]) {
                const json& position = obj["position"];
                const json& orientation = obj["orientation_euler_xyz"];
                const float radius = obj["radius"];
                const float length = obj["length"];

                auto cylinder = factory::cylinder::center::array(position, orientation, radius, length);
                cylinder.name = obj["name"];
                capsules.push_back(cylinder);
            }
        }
    }

    // boxes
    if (problem.contains("box")) {
        for (const auto& obj : problem["box"]) {
            const json& position = obj["position"];
            const json& orientation = obj["orientation_euler_xyz"];
            const json& half_extents = obj["half_extents"];

            auto cuboid = factory::cuboid::array(position, orientation, half_extents);
            cuboid.name = obj["name"];
            cuboids.push_back(cuboid);
        }
    }

    // heap arrays
    if (!spheres.empty()) {
        env.spheres = new Sphere<float>[spheres.size()];
        std::copy(spheres.begin(), spheres.end(), env.spheres);
        env.num_spheres = (unsigned)spheres.size();
    }
    if (!capsules.empty()) {
        env.capsules = new Capsule<float>[capsules.size()];
        std::copy(capsules.begin(), capsules.end(), env.capsules);
        env.num_capsules = (unsigned)capsules.size();
    }
    if (!cuboids.empty()) {
        env.cuboids = new Cuboid<float>[cuboids.size()];
        std::copy(cuboids.begin(), cuboids.end(), env.cuboids);
        env.num_cuboids = (unsigned)cuboids.size();
    }

    return env;
}

void free_host_env(Environment<float>& env) {
    delete[] env.spheres;  env.spheres = nullptr;  env.num_spheres = 0;
    delete[] env.capsules; env.capsules = nullptr; env.num_capsules = 0;
    delete[] env.cuboids;  env.cuboids = nullptr;  env.num_cuboids = 0;
}

json select_problem_instance(const json& problems_root,
                            const std::string& problem_set_name,
                            int problem_idx)
{
    const auto& pset = problems_root.at(problem_set_name);
    if (!pset.is_array()) throw std::runtime_error("problem set is not an array");
    if (problem_idx < 0 || problem_idx >= (int)pset.size())
        throw std::runtime_error("problem_idx out of range");

    const json& data = pset[problem_idx];
    return data;
}

}
