#include "src/collision/prrtc_env_from_json.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>
#include "src/collision/factory.hh"

using json = nlohmann::json;
using namespace ppln::collision;

namespace pRRTC {

namespace {

std::array<float, 3> json_array3(const json& values) {
    return {
        values.at(0).get<float>(),
        values.at(1).get<float>(),
        values.at(2).get<float>(),
    };
}

Eigen::Vector3f json_vector3(const json& values) {
    return Eigen::Vector3f(
        values.at(0).get<float>(),
        values.at(1).get<float>(),
        values.at(2).get<float>());
}

Eigen::Quaternionf pose_quaternion_wxyz(const json& pose) {
    if (!pose.is_array() || pose.size() != 7) {
        throw std::runtime_error("expected pose as [x, y, z, qw, qx, qy, qz]");
    }

    Eigen::Quaternionf rotation(
        pose.at(3).get<float>(),
        pose.at(4).get<float>(),
        pose.at(5).get<float>(),
        pose.at(6).get<float>());
    rotation.normalize();
    return rotation;
}

Eigen::Vector3f pose_position(const json& pose) {
    if (!pose.is_array() || pose.size() != 7) {
        throw std::runtime_error("expected pose as [x, y, z, qw, qx, qy, qz]");
    }

    return Eigen::Vector3f(
        pose.at(0).get<float>(),
        pose.at(1).get<float>(),
        pose.at(2).get<float>());
}

std::string shape_name(const json& obj, const std::string& fallback_name) {
    if (obj.contains("name")) {
        return obj.at("name").get<std::string>();
    }
    return fallback_name;
}

template <typename Fn>
void for_each_shape(const json& collection, Fn&& fn) {
    if (!collection.is_array() && !collection.is_object()) {
        throw std::runtime_error("expected shape collection to be an array or object");
    }

    if (collection.is_array()) {
        for (const auto& obj : collection) {
            fn(obj, shape_name(obj, ""));
        }
        return;
    }

    for (auto it = collection.begin(); it != collection.end(); ++it) {
        fn(it.value(), shape_name(it.value(), it.key()));
    }
}

float cylinder_length(const json& obj) {
    if (obj.contains("height")) {
        return obj.at("height").get<float>();
    }
    if (obj.contains("length")) {
        return obj.at("length").get<float>();
    }
    throw std::runtime_error("cylinder obstacle is missing height/length");
}

}  // namespace

Environment<float> problem_dict_to_env(const json& problem, const std::string& name) {
    Environment<float> env{};

    std::vector<Sphere<float>> spheres;
    std::vector<Capsule<float>> capsules;
    std::vector<Cuboid<float>> cuboids;

    const json& obstacle_root = problem.contains("obstacles") ? problem.at("obstacles") : problem;

    // spheres
    if (obstacle_root.contains("sphere")) {
        for_each_shape(obstacle_root.at("sphere"), [&](const json& obj, const std::string& obj_name) {
            std::array<float, 3> position = obj.contains("pose")
                ? json_array3(obj.at("pose"))
                : json_array3(obj.at("position"));

            Sphere<float> sphere(position[0], position[1], position[2], obj.at("radius").get<float>());
            sphere.name = obj_name;
            spheres.push_back(sphere);
        });
    }

    // cuboids in the current JSON schema.
    if (obstacle_root.contains("cuboid")) {
        for_each_shape(obstacle_root.at("cuboid"), [&](const json& obj, const std::string& obj_name) {
            if (obj.contains("pose")) {
                Eigen::Vector3f center = pose_position(obj.at("pose"));
                Eigen::Quaternionf rotation = pose_quaternion_wxyz(obj.at("pose"));
                Eigen::Vector3f half_extents = 0.5f * json_vector3(obj.at("dims"));

                auto cuboid = factory::cuboid::eigen_rot(center, rotation, half_extents);
                cuboid.name = obj_name;
                cuboids.push_back(cuboid);
                return;
            }

            const std::array<float, 3> position = json_array3(obj.at("position"));
            const std::array<float, 3> orientation = json_array3(obj.at("orientation_euler_xyz"));
            const json& extents = obj.contains("half_extents") ? obj.at("half_extents") : obj.at("dims");
            std::array<float, 3> half_extents = json_array3(extents);

            if (obj.contains("dims")) {
                half_extents[0] *= 0.5f;
                half_extents[1] *= 0.5f;
                half_extents[2] *= 0.5f;
            }

            auto cuboid = factory::cuboid::array(position, orientation, half_extents);
            cuboid.name = obj_name;
            cuboids.push_back(cuboid);
        });
    }

    // cylinders in the current schema are modeled as capsules.
    if (obstacle_root.contains("cylinder")) {
        for_each_shape(obstacle_root.at("cylinder"), [&](const json& obj, const std::string& obj_name) {
            if (obj.contains("pose")) {
                Eigen::Vector3f center = pose_position(obj.at("pose"));
                Eigen::Quaternionf rotation = pose_quaternion_wxyz(obj.at("pose"));
                const float radius = obj.at("radius").get<float>();
                const float length = cylinder_length(obj);

                auto capsule = factory::capsule::center::eigen_rot(center, rotation, radius, length);
                capsule.name = obj_name;
                capsules.push_back(capsule);
                return;
            }

            const std::array<float, 3> position = json_array3(obj.at("position"));
            const std::array<float, 3> orientation = json_array3(obj.at("orientation_euler_xyz"));

            if (name == "box") {
                const float radius = obj.at("radius").get<float>();
                const std::array<float, 3> dims = {radius, radius, radius / 2.0f};

                auto cuboid = factory::cuboid::array(position, orientation, dims);
                cuboid.name = obj_name;
                cuboids.push_back(cuboid);
                return;
            }

            const float radius = obj.at("radius").get<float>();
            const float length = cylinder_length(obj);
            auto cylinder = factory::cylinder::center::array(position, orientation, radius, length);
            cylinder.name = obj_name;
            capsules.push_back(cylinder);
        });
    }

    // legacy box schema.
    if (obstacle_root.contains("box")) {
        for_each_shape(obstacle_root.at("box"), [&](const json& obj, const std::string& obj_name) {
            const std::array<float, 3> position = json_array3(obj.at("position"));
            const std::array<float, 3> orientation = json_array3(obj.at("orientation_euler_xyz"));
            const std::array<float, 3> half_extents = json_array3(obj.at("half_extents"));
            auto cuboid = factory::cuboid::array(position, orientation, half_extents);
            cuboid.name = obj_name;
            cuboids.push_back(cuboid);
        });
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
    env.reset();
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
