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
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include "kernel/util.h"
#include "kernel/cuda_memory.h"
#include "kernel/problem_document.h"
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

inline float finite_number(const json& value) {
    if (!value.is_number())
        throw std::invalid_argument("obstacle geometry must contain finite numbers");
    const double number = value.get<double>();
    if (!std::isfinite(number) || std::abs(number) > std::numeric_limits<float>::max())
        throw std::invalid_argument("obstacle geometry must contain finite fp32 numbers");
    return static_cast<float>(number);
}

inline float positive_number(const json& value) {
    const float number = finite_number(value);
    if (!(number > 0))
        throw std::invalid_argument("obstacle dimensions, radius, and length must be positive");
    return number;
}

// Unit quaternion (w,x,y,z) -> the three columns (body axes u,v,w) of the rotation matrix.
inline void quat_to_basis(float qw, float qx, float qy, float qz,
                          float u[3], float v[3], float w[3]) {
    const double n = std::hypot(std::hypot(double(qw), double(qx)),
                                std::hypot(double(qy), double(qz)));
    if (!(n > 0.0)) throw std::invalid_argument("obstacle quaternion must be non-zero");
    qw = float(qw / n); qx = float(qx / n); qy = float(qy / n); qz = float(qz / n);
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
    if (!a.is_array() || a.size() != 3)
        throw std::invalid_argument("obstacle vector must have exactly three values");
    return {finite_number(a.at(0)), finite_number(a.at(1)), finite_number(a.at(2))};
}

inline std::array<float, 3> dimensions(const json& a) {
    auto result = arr3(a);
    for (float value : result)
        if (!(value > 0)) throw std::invalid_argument("obstacle dimensions must be positive");
    return result;
}

// pose [x,y,z,qw,qx,qy,qz] -> center + body axes.
inline void pose_to_frame(const json& pose, float c[3], float u[3], float v[3], float w[3]) {
    if (!pose.is_array() || pose.size() != 7)
        throw std::invalid_argument("expected pose as [x, y, z, qw, qx, qy, qz]");
    c[0] = finite_number(pose.at(0)); c[1] = finite_number(pose.at(1)); c[2] = finite_number(pose.at(2));
    quat_to_basis(finite_number(pose.at(3)), finite_number(pose.at(4)),
                  finite_number(pose.at(5)), finite_number(pose.at(6)), u, v, w);
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
    if (obj.contains("height")) return positive_number(obj.at("height"));
    if (obj.contains("length")) return positive_number(obj.at("length"));
    throw std::runtime_error("cylinder obstacle is missing height/length");
}

}  // namespace detail

// Parse one problem instance's obstacles into host grid_collision primitives.
inline HostEnv problem_dict_to_env(const json& problem) {
    using namespace detail;
    HostEnv env;
    if (!problem.is_object())
        throw std::invalid_argument("collision problem must be an object");
    const bool wrapped = problem.contains("obstacles");
    const json& root = wrapped ? problem.at("obstacles") : problem;
    if (!root.is_object())
        throw std::invalid_argument("obstacles must be an object");
    if (!wrapped && root.empty())
        throw std::invalid_argument("collision problem must specify obstacles (use {} for an empty scene)");
    for (auto it = root.begin(); it != root.end(); ++it) {
        if (it.key() != "sphere" && it.key() != "cuboid" &&
            it.key() != "cylinder" && it.key() != "box")
            throw std::invalid_argument("unsupported obstacle type: " + it.key());
    }

    if (root.contains("sphere")) {
        for_each_shape(root.at("sphere"), [&](const json& o) {
            std::array<float, 3> p;
            if (o.contains("pose")) {
                float u[3], v[3], w[3];
                pose_to_frame(o.at("pose"), p.data(), u, v, w);
            } else p = arr3(o.at("position"));
            env.spheres.push_back(gc::Sphere<float>{p[0], p[1], p[2], positive_number(o.at("radius"))});
        });
    }

    if (root.contains("cuboid")) {
        for_each_shape(root.at("cuboid"), [&](const json& o) {
            float c[3], u[3], v[3], w[3];
            std::array<float, 3> half;
            if (o.contains("pose")) {
                pose_to_frame(o.at("pose"), c, u, v, w);
                auto dims = dimensions(o.at("dims"));
                half = {0.5f * dims[0], 0.5f * dims[1], 0.5f * dims[2]};
            } else {
                auto pos = arr3(o.at("position"));
                auto rpy = arr3(o.at("orientation_euler_xyz"));
                c[0] = pos[0]; c[1] = pos[1]; c[2] = pos[2];
                euler_xyz_to_basis(rpy[0], rpy[1], rpy[2], u, v, w);
                const json& ext = o.contains("half_extents") ? o.at("half_extents") : o.at("dims");
                half = dimensions(ext);
                if (!o.contains("half_extents")) { half[0] *= 0.5f; half[1] *= 0.5f; half[2] *= 0.5f; }
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
                make_cylinder_capsule(c, w, positive_number(o.at("radius")), cylinder_length(o)));
        });
    }

    if (root.contains("box")) {  // legacy euler-only box schema
        for_each_shape(root.at("box"), [&](const json& o) {
            auto pos = arr3(o.at("position"));
            auto rpy = arr3(o.at("orientation_euler_xyz"));
            auto half = dimensions(o.at("half_extents"));
            float c[3] = {pos[0], pos[1], pos[2]}, u[3], v[3], w[3];
            euler_xyz_to_basis(rpy[0], rpy[1], rpy[2], u, v, w);
            env.cuboids.push_back(make_cuboid(c, u, v, w, half.data()));
        });
    }

    return env;
}

// Device-side handles for one uploaded environment (freed together).
struct DeviceEnv {
    hjcd::DeviceAllocations allocations;
    gc::Sphere<float>*  d_spheres  = nullptr;
    gc::Capsule<float>* d_capsules = nullptr;
    gc::Cuboid<float>*  d_cuboids  = nullptr;
    gc::Environment<float> env{nullptr, 0, nullptr, 0, nullptr, 0};  // device pointers, pass by value
};

// Deep-copy each list to the device; the returned Environment holds device pointers.
inline DeviceEnv upload_env(const HostEnv& h) {
    DeviceEnv d;
    auto up = [&d](const auto& vec, auto*& dptr) {
        using Elem = typename std::decay<decltype(vec)>::type::value_type;
        if (vec.empty()) return 0;
        d.allocations.allocate(dptr, sizeof(Elem) * vec.size());
        CUDA_OK(cudaMemcpy(dptr, vec.data(), sizeof(Elem) * vec.size(), cudaMemcpyHostToDevice));
        return (int)vec.size();
    };
    const int ns = up(h.spheres, d.d_spheres);
    const int nc = up(h.capsules, d.d_capsules);
    const int nb = up(h.cuboids, d.d_cuboids);
    d.env = gc::Environment<float>{d.d_spheres, ns, d.d_capsules, nc, d.d_cuboids, nb};
    return d;
}

inline void free_env(DeviceEnv& d) {
    d.allocations.clear();
    d = DeviceEnv{};
}

}  // namespace hjcd_env

#endif  // HJCD_HAS_COLLISION
