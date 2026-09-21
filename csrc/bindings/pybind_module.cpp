#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
#include <cmath>
#include <mutex>
#include "kernel/hjcd_kernel.h"

namespace py = pybind11;

static std::mutex& solver_mutex() {
  static std::mutex mutex;
  return mutex;
}

static grid::robotModel<double>* ensure_robot() {
  static grid::robotModel<double>* model = grid::init_robotModel<double>();
  static bool limits_inited = false;
  if (!limits_inited) {
    init_joint_limits_from_grid();
    limits_inited = true;
  }
  return model;
}

py::dict py_generate_solutions(const std::array<double,7>& target_pose,
                               int batch_size,
                               int num_solutions,
                               bool collision_free,
                               const std::string& problems_json_text,
                               const std::string& problem_set_name,
                               int problem_idx,
                               int refine_fp64,
                               bool write_stats,
                               const std::string& collision_mode) {
  auto* model = ensure_robot();

  if (batch_size <= 0) throw py::value_error("batch_size must be positive");
  if (num_solutions <= 0) throw py::value_error("num_solutions must be positive");
  if (refine_fp64 < -1 || refine_fp64 > 1)
    throw py::value_error("refine_fp64 must be -1 (auto), 0 (fp32), or 1 (fp64)");
  if (problem_idx < 0) throw py::value_error("problem_idx must be non-negative");
  if (collision_free && problems_json_text.empty())
    throw py::value_error("collision_free=True requires problems_json_text");
  if (collision_free && problem_set_name.empty())
    throw py::value_error("collision_free=True requires problem_set_name");
  if (collision_free && !grid_has_collision())
    throw py::value_error("collision_free=True requires a collision-enabled grid.cuh build");

  int collision_mode_code = -1;
  if (collision_mode == "soft") collision_mode_code = 0;
  else if (collision_mode == "hard") collision_mode_code = 1;
  else if (collision_mode == "both") collision_mode_code = 2;
  else if (collision_mode != "auto")
    throw py::value_error("collision_mode must be one of: hard, soft, both, auto");

  double tp[7];
  for (int i = 0; i < 7; ++i) {
    if (!std::isfinite(target_pose[i])) throw py::value_error("target_pose values must be finite");
    tp[i] = target_pose[i];
  }
  const double qnorm = std::sqrt(tp[3]*tp[3] + tp[4]*tp[4] + tp[5]*tp[5] + tp[6]*tp[6]);
  if (!(qnorm > 1e-12)) throw py::value_error("target_pose quaternion must be non-zero");
  for (int i = 3; i < 7; ++i) tp[i] /= qnorm;

  const char* json_cstr = problems_json_text.empty() ? nullptr : problems_json_text.c_str();
  const char* set_cstr  = problem_set_name.empty() ? nullptr : problem_set_name.c_str();

  // refine_fp64 = the LM-refine precision knob (speed/accuracy), TRI-STATE:
  //   -1 = AUTO (default): pick by regime. num_solutions==1 uses early-stop and is LATENCY-bound
  //        where fp32 is ~1.2x slower -> fp64; num_solutions>=2 runs every candidate to convergence
  //        and is THROUGHPUT-bound where fp32 is 5-7x faster (5090 1/64 fp64) -> fp32.
  //        (measured 2026-06-18; see docs/open-tasks/multiwarp_timing_result.md.)
  //    1 = force fp64 (RT=double, sub-micron).   0 = force fp32 (RT=float, faster, ~fp32 accuracy).
  // Either way I/O stays double, and the Cholesky solve precision follows the compute type.
  const bool use_fp64 = (refine_fp64 < 0) ? (num_solutions <= 1) : (refine_fp64 != 0);
  Result<double> res{};
  {
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(solver_mutex());
    res = use_fp64
        ? generate_ik_solutions<double, double>(
              tp, model, batch_size, num_solutions, collision_free, json_cstr, set_cstr,
              problem_idx, write_stats, collision_mode_code)
        : generate_ik_solutions<double, float>(
              tp, model, batch_size, num_solutions, collision_free, json_cstr, set_cstr,
              problem_idx, write_stats, collision_mode_code);
  }

  const int N = grid_num_joints();

  // The solver may return fewer solutions after collision filtering.
  int S = res.count;

  py::array_t<double> joint_config({S, N});
  py::array_t<double> pose({S, 7});
  py::array_t<double> pos_errors({S});
  py::array_t<double> ori_errors({S});

  std::memcpy(joint_config.mutable_data(), res.joint_config, sizeof(double) * S * N);
  std::memcpy(pose.mutable_data(),         res.pose,         sizeof(double) * S * 7);
  std::memcpy(pos_errors.mutable_data(),   res.pos_errors,   sizeof(double) * S);
  std::memcpy(ori_errors.mutable_data(),   res.ori_errors,   sizeof(double) * S);


  py::dict out;
  out["joint_config"] = std::move(joint_config);
  out["pose"]         = std::move(pose);
  out["pos_errors"]   = std::move(pos_errors);
  out["ori_errors"]   = std::move(ori_errors);
  out["count"]        = S;
  return out;
}

std::vector<std::array<double,7>> py_sample_targets(int num_targets, std::uint64_t seed) {
  if (num_targets <= 0) throw py::value_error("num_targets must be positive");
  auto* model = ensure_robot();
  std::vector<std::array<double,7>> targets;
  {
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(solver_mutex());
    targets = sample_random_target_poses<double>(model, num_targets, seed);
  }
  return targets;
}

PYBIND11_MODULE(_hjcdik, m) {
  m.doc() = "Python bindings for the HJCD-IK CUDA solver";
  m.def("generate_solutions", &py_generate_solutions,
      py::arg("target_pose"),
      py::arg("batch_size") = 2000,
      py::arg("num_solutions") = 1,
      py::arg("collision_free") = false,
      py::arg("problems_json_text") = "",
      py::arg("problem_set_name") = "",
      py::arg("problem_idx") = 0,
      py::arg("refine_fp64") = -1,    // -1=auto (fp64 if num_solutions==1 else fp32); 1=fp64; 0=fp32
      py::arg("write_stats") = false,   // append a row to ik_stats.csv
      py::arg("collision_mode") = "hard");  // hard|soft|both|auto (legacy env fallback)
  m.def("sample_targets", &py_sample_targets,
        py::arg("num_targets"), py::arg("seed") = 0);
  m.def("num_joints", &grid_num_joints);
  m.def("collision_enabled", &grid_has_collision);
  m.def("build_info", [] {
    py::dict info;
    info["num_joints"] = grid_num_joints();
    info["collision_enabled"] = grid_has_collision();
    return info;
  });
}
