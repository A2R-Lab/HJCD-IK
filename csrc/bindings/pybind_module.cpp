#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include "kernel/hjcd_kernel.h"

// Low-level bindings. The user-facing API (hjcdik.Solver, Result, worlds) is pure Python on top of
// these; nothing here is re-exported by the package. Every entry point that touches CUDA takes the
// solver's device, runs on it and restores the caller's current device afterwards.
//
// The same translation unit is compiled twice: by CMake into the packaged `_hjcdik` (the installed
// robot, with the build metadata below defined by CMake) and by `grid_rbd.build_artifact` into a
// per-robot artifact (hjcdik/_build.py passes the defines; HJCDIK_MODULE_NAME keeps the Python init
// symbol unique per artifact). Metadata CMake does not know about defaults to an empty string and the
// Python side fills it in from the artifact record.
#ifndef HJCDIK_MODULE_NAME
#define HJCDIK_MODULE_NAME _hjcdik
#endif
#ifndef HJCDIK_GRID_SHA256
#define HJCDIK_GRID_SHA256 ""
#endif
#ifndef HJCDIK_EE_TARGET
#define HJCDIK_EE_TARGET ""
#endif
#ifndef HJCDIK_ROBOT_NAME
#define HJCDIK_ROBOT_NAME ""
#endif
#ifndef HJCDIK_CUDA_COMPILER_VERSION
#define HJCDIK_CUDA_COMPILER_VERSION ""
#endif
// Bump when the Python package and this binding disagree on names or argument lists.
#define HJCDIK_BINDING_ABI 4

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

// One native call per batch: the host loop over targets runs here with the GIL released once, so the
// Python layer does no per-target work after a GPU wait (measured 2026-10-24: 15-70 us per target when
// it did). Arrays come back NaN-padded to `capacity` rows per target, ready to be wrapped read-only.
struct TargetFailure : std::runtime_error {
    int index;
    TargetFailure(int i, const std::string& what) : std::runtime_error(what), index(i) {}
};

py::dict solve_batch(py::array_t<double, py::array::c_style | py::array::forcecast> poses,
                     int batch_size, int num_solutions, int collision, bool fp64, bool stats,
                     const World* world, int device, double margin_m, bool check_self, std::uint64_t seed,
                     py::object q_init, int capacity) {
  if (poses.ndim() != 2 || poses.shape(1) != 7) throw py::value_error("poses must have shape (T, 7)");
  if (collision < 0 || collision > 3)
    throw py::value_error("collision code must be 0 (none), 1 (hard), 2 (soft) or 3 (both)");
  if (seed > 0xFFFFFFFFull) throw py::value_error("seed must fit an unsigned 32-bit integer");
  if (capacity <= 0) throw py::value_error("capacity must be positive");
  const int T = static_cast<int>(poses.shape(0));
  const int N = grid_num_joints();
  const int C = capacity;
  const auto mode = static_cast<CollisionMode>(collision);
  const CollisionWorld* native_world = world ? world->get() : nullptr;

  // Warm starts: (K, N) shared by every target or (T, K, N) per target.
  py::array_t<double, py::array::c_style | py::array::forcecast> init;
  const double* init_data = nullptr;
  int K = 0;
  bool per_target = false;
  if (!q_init.is_none()) {
    init = py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(q_init);
    if (init.ndim() == 2 && init.shape(1) == N) { K = static_cast<int>(init.shape(0)); }
    else if (init.ndim() == 3 && init.shape(0) == T && init.shape(2) == N) { K = static_cast<int>(init.shape(1)); per_target = true; }
    else throw py::value_error("q_init must have shape (K, num_joints) or (T, K, num_joints)");
    init_data = init.data();
  }

  py::array_t<double> joint_config({T, C, N});
  py::array_t<double> pose({T, C, 7});
  py::array_t<double> pos_errors({T, C});
  py::array_t<double> ori_errors({T, C});
  py::array_t<std::int64_t> count({T});
  py::array_t<double> elapsed({T});
  std::vector<SolveStats> all_stats(stats ? T : 0);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::fill(joint_config.mutable_data(), joint_config.mutable_data() + (size_t)T * C * N, nan);
  std::fill(pose.mutable_data(), pose.mutable_data() + (size_t)T * C * 7, nan);
  std::fill(pos_errors.mutable_data(), pos_errors.mutable_data() + (size_t)T * C, nan);
  std::fill(ori_errors.mutable_data(), ori_errors.mutable_data() + (size_t)T * C, nan);

  const double* P = poses.data();
  double* jc = joint_config.mutable_data();
  double* ps = pose.mutable_data();
  double* pe = pos_errors.mutable_data();
  double* oe = ori_errors.mutable_data();
  std::int64_t* cnt = count.mutable_data();
  double* el = elapsed.mutable_data();
  {
    py::gil_scoped_release release;
    DeviceScope scope(device);
    for (int i = 0; i < T; ++i) {
      SolveOptions options;
      options.margin_m = static_cast<float>(margin_m);
      options.check_self = check_self;
      options.seed = static_cast<std::uint32_t>(seed);
      if (init_data) {
        options.q_init = per_target ? init_data + (size_t)i * K * N : init_data;
        options.num_init = K;
      }
      std::array<double, 7> tp;
      std::copy(P + 7 * i, P + 7 * i + 7, tp.begin());
      Result<double> res{};
      try {
        res = fp64
            ? generate_ik_solutions<double, double>(tp.data(), batch_size, num_solutions, mode, native_world, stats, options)
            : generate_ik_solutions<double, float>(tp.data(), batch_size, num_solutions, mode, native_world, stats, options);
      } catch (const std::invalid_argument& error) {
        throw TargetFailure(i, std::string("invalid argument at target ") + std::to_string(i) + ": " + error.what());
      } catch (const std::exception& error) {
        throw TargetFailure(i, std::string("target ") + std::to_string(i) + ": " + error.what());
      }
      const int Kr = std::min(res.count, C);
      if (Kr > 0) {
        std::memcpy(jc + (size_t)i * C * N, res.joint_config, sizeof(double) * Kr * N);
        std::memcpy(ps + (size_t)i * C * 7, res.pose, sizeof(double) * Kr * 7);
        std::memcpy(pe + (size_t)i * C, res.pos_errors, sizeof(double) * Kr);
        std::memcpy(oe + (size_t)i * C, res.ori_errors, sizeof(double) * Kr);
      }
      cnt[i] = Kr;
      el[i] = res.elapsed_time;
      if (stats) all_stats[i] = res.stats;
    }
  }

  py::dict out;
  out["joint_config"] = std::move(joint_config);
  out["pose"] = std::move(pose);
  out["pos_errors"] = std::move(pos_errors);
  out["ori_errors"] = std::move(ori_errors);
  out["count"] = std::move(count);
  out["elapsed_ms"] = std::move(elapsed);
  if (stats) {
    py::dict d;
    auto ints = [&](const char* name, auto getter) {
      py::array_t<std::int64_t> a({T});
      for (int i = 0; i < T; ++i) a.mutable_data()[i] = getter(all_stats[i]);
      d[name] = std::move(a);
    };
    auto reals = [&](const char* name, auto getter) {
      py::array_t<double> a({T});
      for (int i = 0; i < T; ++i) a.mutable_data()[i] = getter(all_stats[i]);
      d[name] = std::move(a);
    };
    ints("batch_size", [](const SolveStats& s) { return s.batch_size; });
    ints("n_refined", [](const SolveStats& s) { return s.n_refined; });
    ints("n_ik_accurate", [](const SolveStats& s) { return s.n_ik_accurate; });
    ints("n_coll_free_refined", [](const SolveStats& s) { return s.n_coll_free_refined; });
    ints("n_feasible", [](const SolveStats& s) { return s.n_feasible; });
    ints("n_ik_lost", [](const SolveStats& s) { return s.n_ik_lost; });
    ints("n_coll_in_refined", [](const SolveStats& s) { return s.n_coll_in_refined; });
    ints("n_coll_in_coarse", [](const SolveStats& s) { return s.n_coll_in_coarse; });
    reals("env_cost_min_mm", [](const SolveStats& s) { return s.env_cost_min_mm; });
    reals("env_cost_max_mm", [](const SolveStats& s) { return s.env_cost_max_mm; });
    reals("env_cost_mean_mm", [](const SolveStats& s) { return s.env_cost_mean_mm; });
    ints("n_returned", [](const SolveStats& s) { return s.n_returned; });
    ints("n_returned_ik_accurate", [](const SolveStats& s) { return s.n_returned_ik_accurate; });
    ints("n_returned_coll_free", [](const SolveStats& s) { return s.n_returned_coll_free; });
    reals("pct_returned_coll_free", [](const SolveStats& s) { return s.pct_returned_coll_free; });
    ints("n_returned_feasible", [](const SolveStats& s) { return s.n_returned_feasible; });
    out["stats"] = std::move(d);
  } else {
    out["stats"] = py::none();
  }
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

PYBIND11_MODULE(HJCDIK_MODULE_NAME, m) {
  m.doc() = "Low-level bindings for the HJCD-IK CUDA solver; use hjcdik.Solver.";
  m.attr("BINDING_ABI") = HJCDIK_BINDING_ABI;
  py::class_<World>(m, "World")
      .def(py::init<py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::array_t<double, py::array::c_style | py::array::forcecast>, int>(),
           py::arg("spheres"), py::arg("capsules"), py::arg("boxes"), py::arg("device"),
           "Validate and upload a world (spheres (M,4), capsules (M,7), boxes (M,10)) to a device.")
      .def_property_readonly("device", &World::device)
      .def_property_readonly("empty", &World::empty)
      .def_property_readonly("counts", &World::counts);
  py::register_exception<TargetFailure>(m, "TargetFailure", PyExc_RuntimeError);
  m.def("solve_batch", &solve_batch,
        py::arg("poses"), py::arg("batch_size"), py::arg("num_solutions"), py::arg("collision"),
        py::arg("fp64"), py::arg("stats"), py::arg("world").none(true), py::arg("device"),
        py::arg("margin_m") = 0.0, py::arg("check_self") = true, py::arg("seed") = 1337,
        py::arg("q_init") = py::none(), py::arg("capacity") = 1,
        "Solve T targets on a device in one call; collision 0 none, 1 hard, 2 soft, 3 both. Arrays are "
        "(T, capacity, ...) NaN-padded. A failure at target i raises TargetFailure('target i: ...'). Releases the GIL.");
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
    info["binding_abi"] = HJCDIK_BINDING_ABI;
    return info;
  }, R"doc(Return compiled model/build metadata without initializing CUDA.

Includes robot, num_joints, ee_target, collision_enabled, grid_header_sha256 (the SHA-256 of
the selected generated grid.cuh), and cuda_compiler_version (the build
toolkit compiler, not the currently installed driver). Compare the header
hash to detect a stale install or a wheel built for a different robot.
)doc");
}
