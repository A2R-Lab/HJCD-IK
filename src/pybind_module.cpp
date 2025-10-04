#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
#include "include/hjcd_kernel.h"

namespace py = pybind11;

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
                               int num_solutions) {
  auto* model = ensure_robot();

  double tp[7];
  for (int i = 0; i < 7; ++i) tp[i] = target_pose[i];

  auto res = generate_ik_solutions<double>(tp, model, batch_size, num_solutions);

  const int N = grid_num_joints();
  const int S = num_solutions;

  py::array_t<double> joint_config({S, N});
  py::array_t<double> pose({S, 7});
  py::array_t<double> pos_errors({S});
  py::array_t<double> ori_errors({S});

  std::memcpy(joint_config.mutable_data(), res.joint_config, sizeof(double) * S * N);
  std::memcpy(pose.mutable_data(),         res.pose,         sizeof(double) * S * 7);
  std::memcpy(pos_errors.mutable_data(),   res.pos_errors,   sizeof(double) * S);
  std::memcpy(ori_errors.mutable_data(),   res.ori_errors,   sizeof(double) * S);

  delete[] res.joint_config;
  delete[] res.pose;
  delete[] res.pos_errors;
  delete[] res.ori_errors;

  py::dict out;
  out["joint_config"] = std::move(joint_config);
  out["pose"]         = std::move(pose);
  out["pos_errors"]   = std::move(pos_errors);
  out["ori_errors"]   = std::move(ori_errors);
  return out;
}

std::vector<std::array<double,7>> py_sample_targets(int num_targets, std::uint64_t seed) {
  auto* model = ensure_robot();
  return sample_random_target_poses<double>(model, num_targets, seed);
}

PYBIND11_MODULE(_hjcdik, m) {
  m.doc() = "Minimal pybind11 bindings for hjcdik";
  m.def("generate_solutions", &py_generate_solutions,
        py::arg("target_pose"),
        py::arg("batch_size") = 2000,
        py::arg("num_solutions") = 1);
  m.def("sample_targets", &py_sample_targets,
        py::arg("num_targets"), py::arg("seed") = 0);
  m.def("num_joints", &grid_num_joints);
}
