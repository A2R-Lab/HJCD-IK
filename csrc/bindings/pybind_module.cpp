#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
#include "kernel/hjcd_kernel.h"

namespace py = pybind11;


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
  // Size/index validation lives in the native layer (std::invalid_argument -> ValueError);
  // only the Python-specific arguments are checked here.
  if (refine_fp64 < -1 || refine_fp64 > 1)
    throw py::value_error("refine_fp64 must be -1 (auto), 0 (fp32), or 1 (fp64)");
  if (collision_free && problems_json_text.empty())
    throw py::value_error("collision_free=True requires problems_json_text");
  if (collision_free && problem_set_name.empty())
    throw py::value_error("collision_free=True requires problem_set_name");
  if (collision_free && !grid_has_collision())
    throw py::value_error("collision_free=True requires a collision-enabled grid.cuh build");

  int collision_mode_code = 1;
  if (collision_mode == "soft") collision_mode_code = 0;
  else if (collision_mode == "hard") collision_mode_code = 1;
  else if (collision_mode == "both") collision_mode_code = 2;
  else throw py::value_error("collision_mode must be one of: hard, soft, both");

  auto tp = target_pose;  // The native boundary validates and normalizes before touching CUDA.

  const char* json_cstr = problems_json_text.empty() ? nullptr : problems_json_text.c_str();
  const char* set_cstr  = problem_set_name.empty() ? nullptr : problem_set_name.c_str();

  // refine_fp64 = the LM-refine precision knob (speed/accuracy), TRI-STATE:
  //   -1 = AUTO (default): pick by regime. num_solutions==1 uses early-stop and is LATENCY-bound
  //        where fp32 is ~1.2x slower -> fp64; num_solutions>=2 runs every candidate to convergence
  //        and is THROUGHPUT-bound where fp32 is 5-7x faster (5090 1/64 fp64) -> fp32.
  //        (measured 2026-06-18 on an RTX 5090; see docs/development/agent_debugging_guide.md §5.)
  //    1 = force fp64 (RT=double, sub-micron).   0 = force fp32 (RT=float, faster, ~fp32 accuracy).
  // Either way I/O stays double, and the Cholesky solve precision follows the compute type.
  const bool use_fp64 = (refine_fp64 < 0) ? (num_solutions <= 1) : (refine_fp64 != 0);
  Result<double> res{};
  {
    py::gil_scoped_release release;
    res = use_fp64
        ? generate_ik_solutions<double, double>(
              tp.data(), batch_size, num_solutions, collision_free, json_cstr, set_cstr,
              problem_idx, write_stats, collision_mode_code)
        : generate_ik_solutions<double, float>(
              tp.data(), batch_size, num_solutions, collision_free, json_cstr, set_cstr,
              problem_idx, write_stats, collision_mode_code);
  }

  const int N = grid_num_joints();

  // The solver may return fewer solutions after collision filtering.
  int S = res.count;

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
  return out;
}

std::vector<std::array<double,7>> py_sample_targets(int num_targets, std::uint64_t seed) {
  if (num_targets <= 0) throw py::value_error("num_targets must be positive");
  std::vector<std::array<double,7>> targets;
  {
    py::gil_scoped_release release;
    targets = sample_random_target_poses<double>(nullptr, num_targets, seed);
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
      py::arg("collision_mode") = "hard",
      R"doc(Solve one end-effector target using a GPU batch of candidate configurations.

target_pose is [x, y, z, qw, qx, qy, qz], in meters with a scalar-first
quaternion. Finite nonzero quaternions are normalized. batch_size is the
number of candidates, not the number of target poses.

Returns a dict containing independent, owning float64 NumPy arrays:
joint_config (count, num_joints()) in radians; pose (count, 7) in the
input convention; pos_errors (count,) in millimeters; ori_errors (count,)
in radians; and the integer count. count may be smaller than num_solutions,
including zero. Check both errors: returning a candidate does not certify
that the requested target was reached.

collision_free=True requires a collision-enabled build plus
problems_json_text, problem_set_name, and a nonnegative problem_idx.
collision_mode='hard' filters self/environment collisions against the
compiled sphere model; 'soft' only ranks by penetration and does NOT
guarantee collision freedom; 'both' ranks and filters. These modes apply
only when collision_free=True, and do not check the path to a returned
configuration.

refine_fp64=-1 chooses fp64 for one requested solution, otherwise fp32;
1 forces fp64 and 0 forces fp32. I/O remains float64. write_stats=True
appends diagnostics to ik_stats.csv in the current working directory.

Argument conversion errors raise TypeError; invalid values raise
ValueError. Scene and CUDA failures raise exceptions. Calls release the
GIL but serialize access to shared native state. Keep the CUDA context
alive between calls; cudaDeviceReset invalidates cached models.
)doc");
  m.def("sample_targets", &py_sample_targets,
        py::arg("num_targets"), py::arg("seed") = 0,
        R"doc(Sample reachable poses from a seeded Halton sequence within joint limits.

Returns num_targets lists of [x, y, z, qw, qx, qy, qz], using meters and
scalar-first quaternions. num_targets must be positive and seed must fit
an unsigned 64-bit integer. Sampling does not filter self/environment
collisions. The GIL is released while native sampling runs.
)doc");
  m.def("num_joints", &grid_num_joints,
        "Return the compiled robot's actuated joint count without initializing CUDA.");
  m.def("collision_enabled", &grid_has_collision,
        "Report whether this build includes robot collision geometry, without initializing CUDA.");
  m.def("build_info", [] {
    py::dict info;
    info["num_joints"] = grid_num_joints();
    info["collision_enabled"] = grid_has_collision();
    info["grid_header_sha256"] = HJCDIK_GRID_SHA256;
    info["cuda_compiler_version"] = HJCDIK_CUDA_COMPILER_VERSION;
    return info;
  }, R"doc(Return compiled model/build metadata without initializing CUDA.

Includes num_joints, collision_enabled, grid_header_sha256 (the SHA-256 of
the selected generated grid.cuh), and cuda_compiler_version (the build
toolkit compiler, not the currently installed driver). Compare the header
hash to detect a stale install or a wheel built for a different robot.
)doc");
}
