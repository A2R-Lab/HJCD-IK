#pragma once
// Runtime collision environment for the grid_collision path.
//
// Parses a MotionBenchMaker-style problem JSON into host arrays of grid_collision primitives
// (Sphere / Capsule / Cuboid), uploads them to the device, and hands back an Environment<float>
// carrying the DEVICE pointers so it can be passed by value straight into the scoring kernel
// (grid_collision::collision_distance). Replaces the bespoke pRRTC env-from-json + device upload.
//
// REQUIRES grid.cuh to be included FIRST (hjcd_settings.h does this) so grid_collision::{Sphere,
// Capsule,Cuboid,Environment} are visible. No Eigen dependency: pose -> body axes is plain math.
//
// Obstacle pose convention is MotionBenchMaker's: [x, y, z, qw, qx, qy, qz] (position + unit
// quaternion). cuboid = {dims, pose}; cylinder = {radius, height|length, pose} (modeled as a
// capsule, matching the old path); sphere = {radius, pose|position}. Legacy position +
// orientation_euler_xyz forms are also accepted.
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

// The grid_collision primitives (Sphere/Capsule/Cuboid/Environment) exist only when grid.cuh was
// generated with --collision (sentinel HJCD_HAS_COLLISION). Without it this header is empty, so a
// no-collision build (e.g. the DoF-scaling regens) still compiles; the kernel guards its uses too.
#if defined(HJCD_HAS_COLLISION)

namespace hjcd_env {

namespace gc = grid_collision;
using json = nlohmann::json;

struct HostEnv {
    std::vector<gc::Sphere<float>>  spheres;
    std::vector<gc::Capsule<float>> capsules;
    std::vector<gc::Cuboid<float>>  cuboids;
    bool empty() const { return spheres.empty() && capsules.empty() && cuboids.empty(); }
};

namespace detail {

// Unit quaternion (w,x,y,z) -> the three columns (body axes u,v,w) of the rotation matrix.
inline void quat_to_basis(float qw, float qx, float qy, float qz,
                          float u[3], float v[3], float w[3]) {
    const float n = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    if (n > 0.f) { qw /= n; qx /= n; qy /= n; qz /= n; }
    u[0] = 1.f - 2.f * (qy * qy + qz * qz); u[1] = 2.f * (qx * qy + qz * qw); u[2] = 2.f * (qx * qz - qy * qw);
    v[0] = 2.f * (qx * qy - qz * qw); v[1] = 1.f - 2.f * (qx * qx + qz * qz); v[2] = 2.f * (qy * qz + qx * qw);
    w[0] = 2.f * (qx * qz + qy * qw); w[1] = 2.f * (qy * qz - qx * qw); w[2] = 1.f - 2.f * (qx * qx + qy * qy);
}

// Intrinsic XYZ euler (rad) -> quaternion (w,x,y,z), then reuse quat_to_basis. Legacy path only.
inline void euler_xyz_to_basis(float rx, float ry, float rz,
                               float u[3], float v[3], float w[3]) {
    const float cx = std::cos(rx * 0.5f), sx = std::sin(rx * 0.5f);
    const float cy = std::cos(ry * 0.5f), sy = std::sin(ry * 0.5f);
    const float cz = std::cos(rz * 0.5f), sz = std::sin(rz * 0.5f);
    const float qw = cx * cy * cz + sx * sy * sz;
    const float qx = sx * cy * cz - cx * sy * sz;
    const float qy = cx * sy * cz + sx * cy * sz;
    const float qz = cx * cy * sz - sx * sy * cz;
    quat_to_basis(qw, qx, qy, qz, u, v, w);
}

inline std::array<float, 3> arr3(const json& a) {
    return {a.at(0).get<float>(), a.at(1).get<float>(), a.at(2).get<float>()};
}

// pose [x,y,z,qw,qx,qy,qz] -> center + body axes.
inline void pose_to_frame(const json& pose, float c[3], float u[3], float v[3], float w[3]) {
    if (!pose.is_array() || pose.size() != 7)
        throw std::runtime_error("expected pose as [x, y, z, qw, qx, qy, qz]");
    c[0] = pose.at(0).get<float>(); c[1] = pose.at(1).get<float>(); c[2] = pose.at(2).get<float>();
    quat_to_basis(pose.at(3).get<float>(), pose.at(4).get<float>(),
                  pose.at(5).get<float>(), pose.at(6).get<float>(), u, v, w);
}

inline gc::Cuboid<float> make_cuboid(const float c[3], const float u[3], const float v[3],
                                     const float w[3], const float half[3]) {
    return gc::Cuboid<float>{c[0], c[1], c[2],
                             u[0], u[1], u[2], half[0],
                             v[0], v[1], v[2], half[1],
                             w[0], w[1], w[2], half[2]};
}

// Cylinder (center, local-z axis w, radius, length) -> capsule with endpoints c +/- (L/2) w.
inline gc::Capsule<float> make_cylinder_capsule(const float c[3], const float w[3],
                                                float radius, float length) {
    const float h = 0.5f * length;
    return gc::Capsule<float>{c[0] - h * w[0], c[1] - h * w[1], c[2] - h * w[2],
                              c[0] + h * w[0], c[1] + h * w[1], c[2] + h * w[2], radius};
}

template <typename Fn>
inline void for_each_shape(const json& collection, Fn&& fn) {
    if (collection.is_array()) {
        for (const auto& obj : collection) fn(obj);
    } else if (collection.is_object()) {
        for (auto it = collection.begin(); it != collection.end(); ++it) fn(it.value());
    } else {
        throw std::runtime_error("expected shape collection to be an array or object");
    }
}

inline float cylinder_length(const json& obj) {
    if (obj.contains("height")) return obj.at("height").get<float>();
    if (obj.contains("length")) return obj.at("length").get<float>();
    throw std::runtime_error("cylinder obstacle is missing height/length");
}

}  // namespace detail

// Parse one problem instance's obstacles into host grid_collision primitives.
inline HostEnv problem_dict_to_env(const json& problem) {
    using namespace detail;
    HostEnv env;
    const json& root = problem.contains("obstacles") ? problem.at("obstacles") : problem;

    if (root.contains("sphere")) {
        for_each_shape(root.at("sphere"), [&](const json& o) {
            std::array<float, 3> p = o.contains("pose") ? arr3(o.at("pose")) : arr3(o.at("position"));
            env.spheres.push_back(gc::Sphere<float>{p[0], p[1], p[2], o.at("radius").get<float>()});
        });
    }

    if (root.contains("cuboid")) {
        for_each_shape(root.at("cuboid"), [&](const json& o) {
            float c[3], u[3], v[3], w[3];
            std::array<float, 3> half;
            if (o.contains("pose")) {
                pose_to_frame(o.at("pose"), c, u, v, w);
                auto dims = arr3(o.at("dims"));
                half = {0.5f * dims[0], 0.5f * dims[1], 0.5f * dims[2]};
            } else {
                auto pos = arr3(o.at("position"));
                auto rpy = arr3(o.at("orientation_euler_xyz"));
                c[0] = pos[0]; c[1] = pos[1]; c[2] = pos[2];
                euler_xyz_to_basis(rpy[0], rpy[1], rpy[2], u, v, w);
                const json& ext = o.contains("half_extents") ? o.at("half_extents") : o.at("dims");
                half = arr3(ext);
                if (o.contains("dims")) { half[0] *= 0.5f; half[1] *= 0.5f; half[2] *= 0.5f; }
            }
            env.cuboids.push_back(make_cuboid(c, u, v, w, half.data()));
        });
    }

    if (root.contains("cylinder")) {
        for_each_shape(root.at("cylinder"), [&](const json& o) {
            float c[3], u[3], v[3], w[3];
            if (o.contains("pose")) pose_to_frame(o.at("pose"), c, u, v, w);
            else {
                auto pos = arr3(o.at("position"));
                auto rpy = arr3(o.at("orientation_euler_xyz"));
                c[0] = pos[0]; c[1] = pos[1]; c[2] = pos[2];
                euler_xyz_to_basis(rpy[0], rpy[1], rpy[2], u, v, w);
            }
            env.capsules.push_back(
                make_cylinder_capsule(c, w, o.at("radius").get<float>(), cylinder_length(o)));
        });
    }

    if (root.contains("box")) {  // legacy euler-only box schema
        for_each_shape(root.at("box"), [&](const json& o) {
            auto pos = arr3(o.at("position"));
            auto rpy = arr3(o.at("orientation_euler_xyz"));
            auto half = arr3(o.at("half_extents"));
            float c[3] = {pos[0], pos[1], pos[2]}, u[3], v[3], w[3];
            euler_xyz_to_basis(rpy[0], rpy[1], rpy[2], u, v, w);
            env.cuboids.push_back(make_cuboid(c, u, v, w, half.data()));
        });
    }

    return env;
}

inline json select_problem_instance(const json& problems_root,
                                     const std::string& problem_set_name, int problem_idx) {
    const auto& pset = problems_root.at(problem_set_name);
    if (!pset.is_array()) throw std::runtime_error("problem set is not an array");
    if (problem_idx < 0 || problem_idx >= (int)pset.size())
        throw std::runtime_error("problem_idx out of range");
    return pset[problem_idx];
}

// Device-side handles for one uploaded environment (freed together).
struct DeviceEnv {
    gc::Sphere<float>*  d_spheres  = nullptr;
    gc::Capsule<float>* d_capsules = nullptr;
    gc::Cuboid<float>*  d_cuboids  = nullptr;
    gc::Environment<float> env{nullptr, 0, nullptr, 0, nullptr, 0};  // device pointers, pass by value
};

// Deep-copy each list to the device; the returned Environment holds device pointers.
inline DeviceEnv upload_env(const HostEnv& h) {
    DeviceEnv d;
    auto up = [](const auto& vec, auto*& dptr) {
        using Elem = typename std::decay<decltype(vec)>::type::value_type;
        if (vec.empty()) return 0;
        cudaMalloc(&dptr, sizeof(Elem) * vec.size());
        cudaMemcpy(dptr, vec.data(), sizeof(Elem) * vec.size(), cudaMemcpyHostToDevice);
        return (int)vec.size();
    };
    const int ns = up(h.spheres, d.d_spheres);
    const int nc = up(h.capsules, d.d_capsules);
    const int nb = up(h.cuboids, d.d_cuboids);
    d.env = gc::Environment<float>{d.d_spheres, ns, d.d_capsules, nc, d.d_cuboids, nb};
    return d;
}

inline void free_env(DeviceEnv& d) {
    if (d.d_spheres) cudaFree(d.d_spheres);
    if (d.d_capsules) cudaFree(d.d_capsules);
    if (d.d_cuboids) cudaFree(d.d_cuboids);
    d = DeviceEnv{};
}

}  // namespace hjcd_env

#endif  // HJCD_HAS_COLLISION
