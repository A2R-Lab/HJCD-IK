#pragma once
// Runtime collision environment for the grid_collision path.
//
// Converts a host WorldSpec (plain arrays of spheres, capsules and boxes in the robot base frame) into
// grid_collision primitives (Sphere / Capsule / Cuboid), uploads them to the device, and hands back an
// Environment<float> carrying the DEVICE pointers so it can be passed by value straight into the scoring
// kernel (grid_collision::collision_distance).
//
// Move to GRiD (ask G8): every consumer needs arrays -> owning device environment; this file is HJCD's
// interim copy and is deleted when GRiD ships the builder.
//
// REQUIRES grid.cuh to be included FIRST (hjcd_settings.h does this) so grid_collision::{Sphere,
// Capsule,Cuboid,Environment} are visible. No Eigen dependency: quaternion -> body axes is plain math.
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include "kernel/util.h"
#include "kernel/cuda_memory.h"
#include "kernel/hjcd_kernel.h"

// The grid_collision primitives (Sphere/Capsule/Cuboid/Environment) exist only when grid.cuh was
// generated with --collision (sentinel HJCD_HAS_COLLISION). Without it this header is empty, so a
// no-collision build (e.g. the DoF-scaling regens) still compiles; the kernel guards its uses too.
#if defined(HJCD_HAS_COLLISION)

namespace hjcd_env {

namespace gc = grid_collision;

struct HostEnv {
    std::vector<gc::Sphere<float>>  spheres;
    std::vector<gc::Capsule<float>> capsules;
    std::vector<gc::Cuboid<float>>  cuboids;
    bool empty() const { return spheres.empty() && capsules.empty() && cuboids.empty(); }
};

namespace detail {

inline float finite_number(float value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
        throw std::invalid_argument("obstacle geometry must contain finite fp32 numbers");
    return value;
}

inline float positive_number(float value) {
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

inline gc::Cuboid<float> make_cuboid(const float c[3], const float u[3], const float v[3],
                                     const float w[3], const float half[3]) {
    return gc::Cuboid<float>{c[0], c[1], c[2],
                             u[0], u[1], u[2], half[0],
                             v[0], v[1], v[2], half[1],
                             w[0], w[1], w[2], half[2]};
}

}  // namespace detail

// Validate a WorldSpec and convert it to host grid_collision primitives.
inline HostEnv host_env_from_spec(const WorldSpec& spec) {
    using namespace detail;
    if (spec.spheres.size() % 4 || spec.capsules.size() % 7 || spec.boxes.size() % 10)
        throw std::invalid_argument("world arrays must be whole rows: spheres 4, capsules 7, boxes 10 values");
    HostEnv env;
    for (int i = 0; i < spec.num_spheres(); ++i) {
        const float* s = spec.spheres.data() + 4 * i;
        env.spheres.push_back(gc::Sphere<float>{finite_number(s[0]), finite_number(s[1]), finite_number(s[2]),
                                                positive_number(s[3])});
    }
    for (int i = 0; i < spec.num_capsules(); ++i) {
        const float* c = spec.capsules.data() + 7 * i;
        env.capsules.push_back(gc::Capsule<float>{finite_number(c[0]), finite_number(c[1]), finite_number(c[2]),
                                                  finite_number(c[3]), finite_number(c[4]), finite_number(c[5]),
                                                  positive_number(c[6])});
    }
    for (int i = 0; i < spec.num_boxes(); ++i) {
        const float* b = spec.boxes.data() + 10 * i;
        const float c[3] = {finite_number(b[0]), finite_number(b[1]), finite_number(b[2])};
        float u[3], v[3], w[3];
        quat_to_basis(finite_number(b[3]), finite_number(b[4]), finite_number(b[5]), finite_number(b[6]), u, v, w);
        const float half[3] = {0.5f * positive_number(b[7]), 0.5f * positive_number(b[8]), 0.5f * positive_number(b[9])};
        env.cuboids.push_back(make_cuboid(c, u, v, w, half));
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

}  // namespace hjcd_env

#endif  // HJCD_HAS_COLLISION
