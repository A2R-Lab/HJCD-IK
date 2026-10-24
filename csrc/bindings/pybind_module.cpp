#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <cstring>
#include <memory>
#include "kernel/hjcd_kernel.h"

// Low-level bindings. The user-facing API (hjcdik.Solver, Result, worlds) is pure Python on top of
// these; nothing here is re-exported by the package. Every entry point that touches CUDA takes the
// solver's device, runs on it and restores the caller's current device afterwards.

namespace py = pybind11;

namespace {

struct DeviceScope {
    int previous = -1;
    bool switched = false;
    explicit DeviceScope(int device) {
        if (device < 0) return;
        if (cudaGetDevice(&previous) != cudaSuccess) {
            (void)cudaGetLastError();
            throw std::runtime_error("CUDA device query failed (no usable CUDA runtime or device)");
        }
        if (previous != device) {
            if (cudaSetDevice(device) != cudaSuccess) {
                (void)cudaGetLastError();   // clear the sticky error so later calls on the valid device run
                throw std::runtime_error("cudaSetDevice failed for the solver's device " + std::to_string(device));
            }
            switched = true;
        }
    }
    ~DeviceScope() { if (switched) (void)cudaSetDevice(previous); }
};

int current_device() {
    int device = 0;
    cudaError_t e = cudaGetDevice(&device);
    if (e != cudaSuccess) {
        (void)cudaGetLastError();
        throw std::runtime_error(std::string("CUDA device query failed: ") + cudaGetErrorString(e));
    }
    return device;
}

std::vector<float> rows(const py::array_t<double, py::array::c_style | py::array::forcecast>& a,
                        int width, const char* name) {
    if (a.ndim() != 2 || a.shape(1) != width)
        throw py::value_error(std::string(name) + " must have shape (M, " + std::to_string(width) + ")");
    std::vector<float> out(static_cast<size_t>(a.shape(0)) * width);
    const double* src = a.data();
    for (size_t i = 0; i < out.size(); ++i) {
        if (!std::isfinite(src[i])) throw py::value_error(std::string(name) + " must contain finite values");
        out[i] = static_cast<float>(src[i]);
    }
    return out;
}

// Owning device world (RAII through shared_ptr<CollisionWorld>); kernels receive a view by value.
class World {
public:
    World(py::array_t<double, py::array::c_style | py::array::forcecast> spheres,
          py::array_t<double, py::array::c_style | py::array::forcecast> capsules,
          py::array_t<double, py::array::c_style | py::array::forcecast> boxes,
          int device) {
        WorldSpec spec;
        spec.spheres = rows(spheres, 4, "spheres");
        spec.capsules = rows(capsules, 7, "capsules");
        spec.boxes = rows(boxes, 10, "boxes");
        py::gil_scoped_release release;
        DeviceScope scope(device);
        world_ = make_collision_world(spec);
    }
    const CollisionWorld* get() const { return world_.get(); }
    int device() const { return collision_world_device(*world_); }
    bool empty() const { return collision_world_empty(*world_); }
    std::array<int, 3> counts() const { return collision_world_counts(*world_); }
private:
    std::shared_ptr<CollisionWorld> world_;
};

py::dict stats_dict(const SolveStats& s) {
    py::dict d;
    d["batch_size"] = s.batch_size;
    d["n_refined"] = s.n_refined;
    d["n_ik_accurate"] = s.n_ik_accurate;
    d["n_coll_free_refined"] = s.n_coll_free_refined;
    d["n_feasible"] = s.n_feasible;
    d["n_ik_lost"] = s.n_ik_lost;
    d["n_coll_in_refined"] = s.n_coll_in_refined;
    d["n_coll_in_coarse"] = s.n_coll_in_coarse;
    d["env_cost_min_mm"] = s.env_cost_min_mm;
    d["env_cost_max_mm"] = s.env_cost_max_mm;
    d["env_cost_mean_mm"] = s.env_cost_mean_mm;
    d["n_returned"] = s.n_returned;
    d["n_returned_ik_accurate"] = s.n_returned_ik_accurate;
    d["n_returned_coll_free"] = s.n_returned_coll_free;
    d["pct_returned_coll_free"] = s.pct_returned_coll_free;
    d["n_returned_feasible"] = s.n_returned_feasible;
    return d;
}

py::dict solve_one(const std::array<double, 7>& target_pose,
                   int batch_size, int num_solutions, int collision, bool fp64, bool stats,
                   const World* world, int device) {
  if (collision < 0 || collision > 3)
    throw py::value_error("collision code must be 0 (none), 1 (hard), 2 (soft) or 3 (both)");
  const auto mode = static_cast<CollisionMode>(collision);
  const CollisionWorld* native_world = world ? world->get() : nullptr;
  auto tp = target_pose;  // The native boundary validates and normalizes before touching CUDA.

  Result<double> res{};
  {
    py::gil_scoped_release release;
    DeviceScope scope(device);
    res = fp64
        ? generate_ik_solutions<double, double>(tp.data(), batch_size, num_solutions, mode, native_world, stats)
        : generate_ik_solutions<double, float>(tp.data(), batch_size, num_solutions, mode, native_world, stats);
  }

  const int N = grid_num_joints();
  const int S = res.count;   // may be smaller than requested after filtering and duplicate removal

  py::array_t<double> joint_config({S, N});
  py::array_t<double> pose({S, 7});
  py::array_t<double> pos_errors({S});
  py::array_t<double> ori_errors({S});
  if (S > 0) {
    std::memcpy(joint_config.mutable_data(), res.joint_config, sizeof(double) * S * N);
    std::memcpy(pose.mutable_data(),         res.pose,         sizeof(double) * S * 7);
    std::memcpy(pos_errors.mutable_data(),   res.pos_errors,   sizeof(double) * S);
    std::memcpy(ori_errors.mutable_data(),   res.ori_errors,   sizeof(double) * S);
  }

  py::dict out;
  out["joint_config"] = std::move(joint_config);
  out["pose"]         = std::move(pose);
  out["pos_errors"]   = std::move(pos_errors);
  out["ori_errors"]   = std::move(ori_errors);
  out["count"]        = S;
  out["elapsed_ms"]   = res.elapsed_time;
  if (res.has_stats) out["stats"] = stats_dict(res.stats); else out["stats"] = py::none();
  return out;
}

std::vector<std::array<double, 7>> sample_targets(int num_targets, std::uint64_t seed, int device) {
  if (num_targets <= 0) throw py::value_error("num_targets must be positive");
  std::vector<std::array<double, 7>> targets;
  {
    py::gil_scoped_release release;
    DeviceScope scope(device);
    targets = sample_random_target_poses<double>(num_targets, seed);
  }
  return targets;
}

}  // namespace

PYBIND11_MODULE(_hjcdik, m) {
  m.doc() = "Low-level bindings for the HJCD-IK CUDA solver; use hjcdik.Solver.";
  py::class_<World>(m, "World")
      .def(py::init<py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::array_t<double, py::array::c_style | py::array::forcecast>, int>(),
           py::arg("spheres"), py::arg("capsules"), py::arg("boxes"), py::arg("device"),
           "Validate and upload a world (spheres (M,4), capsules (M,7), boxes (M,10)) to a device.")
      .def_property_readonly("device", &World::device)
      .def_property_readonly("empty", &World::empty)
      .def_property_readonly("counts", &World::counts);
  m.def("solve_one", &solve_one,
        py::arg("target_pose"), py::arg("batch_size"), py::arg("num_solutions"), py::arg("collision"),
        py::arg("fp64"), py::arg("stats"), py::arg("world").none(true), py::arg("device"),
        "Solve one target on a device; collision 0 none, 1 hard, 2 soft, 3 both. Releases the GIL.");
  m.def("sample_targets", &sample_targets,
        py::arg("num_targets"), py::arg("seed"), py::arg("device"),
        "Seeded Halton samples over the joint limits, mapped through the compiled FK. Releases the GIL.");
  m.def("current_device", &current_device, "The calling thread's current CUDA device.");
  m.def("num_joints", &grid_num_joints,
        "Return the compiled robot's actuated joint count without initializing CUDA.");
  m.def("collision_enabled", &grid_has_collision,
        "Report whether this build includes robot collision geometry, without initializing CUDA.");
  m.def("build_info", [] {
    py::dict info;
    info["robot"] = HJCDIK_ROBOT_NAME;
    info["num_joints"] = grid_num_joints();
    info["ee_target"] = HJCDIK_EE_TARGET;
    info["collision_enabled"] = grid_has_collision();
    info["grid_header_sha256"] = HJCDIK_GRID_SHA256;
    info["cuda_compiler_version"] = HJCDIK_CUDA_COMPILER_VERSION;
    return info;
  }, R"doc(Return compiled model/build metadata without initializing CUDA.

Includes robot, num_joints, ee_target, collision_enabled, grid_header_sha256 (the SHA-256 of
the selected generated grid.cuh), and cuda_compiler_version (the build
toolkit compiler, not the currently installed driver). Compare the header
hash to detect a stale install or a wheel built for a different robot.
)doc");
}
