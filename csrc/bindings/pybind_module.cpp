#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
#include <vector>
#include "kernel/hjcd_kernel.h"

namespace py = pybind11;

// ---------------------------------------------------------------------------------------------
// Array construction. EVERY array returned to Python is built here.
//
// pybind11's shape-only constructor -- py::array_t<T>(shape) -- yields a ZERO-STRIDED array when
// the shape is ONE-dimensional: every element aliases element 0, so B distinct C++ values collapse
// into one repeated value. Multi-dimensional shapes are unaffected, which is exactly why this hid
// for so long -- the [B, K] error arrays were correct while the [B] cost and success arrays built
// beside them were not. The damage was silent and severe: `cost` became a constant, so argmin over
// it always returned candidate 0 and candidate selection was effectively arbitrary, and `success`
// broadcast one candidate's flag across the whole batch. A bare count, py::array_t<T>(count, ptr),
// has the same defect.
//
// The fix is to always pass an explicit shape AND explicit C-contiguous byte strides. Do not
// construct a py::array_t from a bare shape or a bare count anywhere in this file.
// ---------------------------------------------------------------------------------------------
namespace {

using shape_t = std::vector<py::ssize_t>;

template <typename T>
shape_t c_strides(const shape_t& shape) {
  shape_t st(shape.size());
  py::ssize_t s = static_cast<py::ssize_t>(sizeof(T));
  for (size_t i = shape.size(); i-- > 0;) { st[i] = s; s *= shape[i]; }
  return st;
}

// Uninitialised, explicitly C-contiguous. Caller fills through mutable_data().
template <typename T>
py::array_t<T> make_arr(const shape_t& shape) {
  return py::array_t<T>(py::array::ShapeContainer(shape),
                        py::array::StridesContainer(c_strides<T>(shape)));
}

// Copies src into a fresh numpy-owned buffer (pybind copies when ptr is given without a base).
template <typename T>
py::array_t<T> arr_from(const T* src, const shape_t& shape) {
  return py::array_t<T>(py::array::ShapeContainer(shape),
                        py::array::StridesContainer(c_strides<T>(shape)), src);
}

template <typename T>
py::array_t<T> arr_from(const std::vector<T>& v, const shape_t& shape) {
  return arr_from<T>(v.data(), shape);
}

// The device carries booleans as unsigned char; numpy wants bool.
py::array_t<bool> barr_from(const unsigned char* v, const shape_t& shape) {
  auto a = make_arr<bool>(shape);
  bool* d = a.mutable_data();
  py::ssize_t n = 1;
  for (auto s : shape) n *= s;
  for (py::ssize_t i = 0; i < n; ++i) d[i] = (v[i] != 0);
  return a;
}

py::array_t<bool> barr_from(const std::vector<unsigned char>& v, const shape_t& shape) {
  return barr_from(v.data(), shape);
}

}  // namespace


// ---------------------------------------------------------------------------------------------
// Workspace object. Owned by exactly ONE HJCDSolver on the Python side.
//
// OWNERSHIP   : the Python HJCDSolver holds the only reference; destruction frees the arena.
// DEVICE      : bound to the CUDA device current at construction.
// STREAM      : the default stream. There is no per-solver stream yet.
// THREAD SAFETY: NOT thread-safe. Exactly one active call per instance -- the Python wrapper
//               enforces this with a re-entrancy guard and raises rather than racing.
// ---------------------------------------------------------------------------------------------
class PyWorkspace {
public:
  PyWorkspace() : w_(hjcd_workspace_new()) {}
  ~PyWorkspace() { hjcd_workspace_free(w_); }
  PyWorkspace(const PyWorkspace&) = delete;
  HjcdWorkspace* get() { return w_; }
  py::dict stats() const {
    size_t nm, nf, by; int cap, dev;
    hjcd_workspace_stats(w_, &nm, &nf, &by, &cap, &dev);
    py::dict d;
    d["cuda_mallocs"] = nm; d["cuda_frees"] = nf;
    d["bytes"] = by; d["capacity_B"] = cap; d["device"] = dev;
    return d;
  }
private:
  HjcdWorkspace* w_;
};

// An input array is on the FAST PATH when it is already float32 and the solver computes in float32:
// the numpy buffer is handed straight to cudaMemcpy, with no host conversion at all.
using arrf = py::array_t<float,  py::array::c_style | py::array::forcecast>;

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
                               const std::string& coarse_mode,
                               int coarse_iters,
                               bool coarse_incremental) {
  auto* model = ensure_robot();

  double tp[7];
  for (int i = 0; i < 7; ++i) tp[i] = target_pose[i];

  const char* json_cstr = problems_json_text.empty() ? nullptr : problems_json_text.c_str();
  const char* set_cstr  = problem_set_name.empty() ? nullptr : problem_set_name.c_str();

  // refine_fp64 = the LM-refine precision knob (speed/accuracy), TRI-STATE:
  //   -1 = AUTO (default): pick by regime. num_solutions==1 uses early-stop and is LATENCY-bound
  //        where fp32 is ~1.2x slower -> fp64; num_solutions>=2 runs every candidate to convergence
  //        and is THROUGHPUT-bound where fp32 is 5-7x faster (5090 1/64 fp64) -> fp32.
  //        (measured 2026-06-18; see docs/open-tasks/multiwarp_timing_result.md.)
  //    1 = force fp64 (RT=double, sub-micron).   0 = force fp32 (RT=float, faster, ~fp32 accuracy).
  // Either way I/O stays double, and the Cholesky solve precision follows the compute type.
  int cm = 0;
  if      (coarse_mode == "auto")         cm = 0;
  else if (coarse_mode == "none")         cm = 1;
  else if (coarse_mode == "multi_target") cm = 2;
  else if (coarse_mode == "legacy")       cm = 3;
  else throw std::invalid_argument(
      "coarse_mode must be one of: auto, none, multi_target, legacy (got '" + coarse_mode + "')");

  const bool use_fp64 = (refine_fp64 < 0) ? (num_solutions <= 1) : (refine_fp64 != 0);
  auto res = use_fp64
      ? generate_ik_solutions<double, double>(
            tp, model, batch_size, num_solutions, collision_free, json_cstr, set_cstr, problem_idx,
            write_stats, cm, coarse_iters, coarse_incremental ? 1 : 0)
      : generate_ik_solutions<double, float>(
            tp, model, batch_size, num_solutions, collision_free, json_cstr, set_cstr, problem_idx,
            write_stats, cm, coarse_iters, coarse_incremental ? 1 : 0);

  const int N = grid_num_joints();

  // The solver may return fewer solutions after collision filtering.
  int S = res.count;

  auto joint_config = arr_from<double>(res.joint_config, {S, N});
  auto pose         = arr_from<double>(res.pose,         {S, 7});
  auto pos_errors   = arr_from<double>(res.pos_errors,   {S});
  auto ori_errors   = arr_from<double>(res.ori_errors,   {S});

  delete[] res.joint_config;
  delete[] res.pose;
  delete[] res.pos_errors;
  delete[] res.ori_errors;

  py::dict out;
  out["joint_config"] = std::move(joint_config);
  out["pose"]         = std::move(pose);
  out["pos_errors"]   = std::move(pos_errors);
  out["ori_errors"]   = std::move(ori_errors);
  out["count"]        = S;
  return out;
}

std::vector<std::array<double,7>> py_sample_targets(int num_targets, std::uint64_t seed) {
  auto* model = ensure_robot();
  return sample_random_target_poses<double>(model, num_targets, seed);
}

// q: (B, N) -> (B, F, 4, 4) world transforms, F = grid_num_frames(). The 4x4s are returned
// ROW-major (numpy-natural); the kernel stores them column-major, so we transpose on the way out.
py::array_t<double> py_link_transforms(py::array_t<double, py::array::c_style | py::array::forcecast> q) {
  auto* model = ensure_robot();
  const int N = grid_num_joints();
  const int F = grid_num_frames();

  if (q.ndim() != 2 || q.shape(1) != N)
    throw std::invalid_argument("q must have shape (B, " + std::to_string(N) + ")");
  const int B = (int)q.shape(0);
  if (B <= 0) throw std::invalid_argument("q must have at least one row");

  std::vector<double> flat = compute_link_transforms<double>(q.data(), B, model);

  auto out = make_arr<double>({B, F, 4, 4});
  double* o = out.mutable_data();
  for (int b = 0; b < B; ++b)
    for (int f = 0; f < F; ++f)
      for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
          o[((size_t)b * F + f) * 16 + r * 4 + c] = flat[((size_t)b * F + f) * 16 + c * 4 + r];
  return out;
}

// q: (B, N) -> (B, K, 4, 4) world pose of each generated target frame, K = grid_num_targets().
// Returned ROW-major; the kernel stores column-major.
py::array_t<double> py_target_transforms(py::array_t<double, py::array::c_style | py::array::forcecast> q) {
  auto* model = ensure_robot();
  const int N = grid_num_joints();
  const int K = grid_num_targets();

  if (q.ndim() != 2 || q.shape(1) != N)
    throw std::invalid_argument("q must have shape (B, " + std::to_string(N) + ")");
  const int B = (int)q.shape(0);
  if (B <= 0) throw std::invalid_argument("q must have at least one row");

  std::vector<double> flat = compute_target_transforms<double>(q.data(), B, model);

  auto out = make_arr<double>({B, K, 4, 4});
  double* o = out.mutable_data();
  for (int b = 0; b < B; ++b)
    for (int k = 0; k < K; ++k)
      for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
          o[((size_t)b * K + k) * 16 + r * 4 + c] = flat[((size_t)b * K + k) * 16 + c * 4 + r];
  return out;
}

// The generated target metadata as the DEVICE sees it (dumped from a kernel, not a host copy).
py::dict py_target_metadata() {
  ensure_robot();
  TargetMetadata m = read_target_metadata();
  const int K = m.num_targets;

  auto tool = make_arr<double>({K, 4, 4});
  double* t = tool.mutable_data();
  for (int k = 0; k < K; ++k)
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c)
        t[(size_t)k * 16 + r * 4 + c] = m.tool_xform[(size_t)k * 16 + c * 4 + r];  // col -> row major

  py::dict d;
  d["num_targets"] = m.num_targets;
  d["num_joints"] = m.num_joints;
  d["anchor_jid"] = m.anchor_jid;
  d["target_ancestor_mask"] = m.target_ancestor_mask;
  d["joint_target_mask"] = m.joint_target_mask;
  d["tool_xform"] = std::move(tool);
  return d;
}

// Residual diagnostics. This binding takes ONLY canonical, fully-broadcast arrays -- all validation,
// weight broadcasting and [B,K]-bool -> [B]-bitmask packing happens in hjcdik/__init__.py, above
// this layer, so nothing unvalidated ever reaches CUDA.
using arrd = py::array_t<double, py::array::c_style | py::array::forcecast>;
using arru = py::array_t<std::uint32_t, py::array::c_style | py::array::forcecast>;

py::dict py_target_residuals(arrd q, arrd tgt_p, arrd tgt_q, arru active,
                             arrd w_pos, arrd w_ori, arrd eps_pos, arrd eps_ori) {
  auto* model = ensure_robot();
  const int N = grid_num_joints();
  const int K = grid_num_targets();

  if (q.ndim() != 2 || q.shape(1) != N)
    throw std::invalid_argument("q must have shape (B, " + std::to_string(N) + ")");
  const int B = (int)q.shape(0);
  auto want = [&](const py::array& a, std::vector<py::ssize_t> s, const char* nm) {
    if (a.ndim() != (py::ssize_t)s.size())
      throw std::invalid_argument(std::string(nm) + ": wrong rank");
    for (size_t i = 0; i < s.size(); ++i)
      if (a.shape((py::ssize_t)i) != s[i])
        throw std::invalid_argument(std::string(nm) + ": wrong shape");
  };
  want(tgt_p, {B, K, 3}, "target_positions");
  want(tgt_q, {B, K, 4}, "target_quaternions");
  want(active, {B}, "active_target_mask");
  want(w_pos, {B, K}, "position_weights");
  want(w_ori, {B, K}, "orientation_weights");
  want(eps_pos, {K}, "position_tol");
  want(eps_ori, {K}, "orientation_tol");

  ResidualOutputs r = compute_target_residuals(
      q.data(), tgt_p.data(), tgt_q.data(), active.data(), w_pos.data(), w_ori.data(),
      eps_pos.data(), eps_ori.data(), B, model);

  py::dict o;
  o["position_residuals"]    = arr_from(r.e_pos, {B, K, 3});
  o["orientation_residuals"] = arr_from(r.e_ori, {B, K, 3});
  o["position_errors"]       = arr_from(r.pos_norm, {B, K});
  o["orientation_errors"]    = arr_from(r.ori_norm, {B, K});
  o["target_costs"]          = arr_from(r.cost, {B, K});
  o["cost_raw"]              = arr_from(r.cost_raw, {B});
  o["cost_normalized"]       = arr_from(r.cost_norm, {B});
  o["target_success"]        = barr_from(r.success, {B, K});
  o["success"]               = barr_from(r.success_all, {B});
  o["active_target_mask"]    = arr_from(active.data(), {B});
  return o;
}

py::dict py_normal_equations(arrd q, arrd tgt_p, arrd tgt_q, arru active, arrd w_pos, arrd w_ori) {
  auto* model = ensure_robot();
  const int N = grid_num_joints();
  const int B = (int)q.shape(0);
  NormalEquations r = compute_normal_equations(q.data(), tgt_p.data(), tgt_q.data(), active.data(),
                                               w_pos.data(), w_ori.data(), B, model);
  py::dict o;
  o["A"] = arr_from(r.A, {B, N, N});
  o["b"] = arr_from(r.b, {B, N});
  return o;
}

py::dict py_lm_refine(py::array q, py::array tgt_p, py::array tgt_q, arru active,
                      py::array w_pos, py::array w_ori,
                      double eps_pos, double eps_ori, double lambda_init, int max_iters,
                      bool diagnostics, bool return_trace, int precision,
                      int stag_patience, double stag_rel, PyWorkspace* ws) {
  auto* model = ensure_robot();
  const int N = grid_num_joints();
  const int K = grid_num_targets();
  const int B = (int)q.shape(0);

  // Python guarantees these are C-contiguous and all of ONE dtype: float32 for an fp32 solve
  // (direct H2D, zero host conversion), float64 otherwise (compatibility, narrowed once).
  const bool in_f32 = q.dtype().is(py::dtype::of<float>());
  SolveInputs in{q.data(), tgt_p.data(), tgt_q.data(), w_pos.data(), w_ori.data(),
                 (const unsigned int*)active.data(), in_f32};

  // Allocate the OUTPUT config first, in the compute type, and let the D2H land straight in it.
  const bool out_f32 = (precision == 1);
  py::array qa = out_f32 ? py::array(make_arr<float>({B, N})) : py::array(make_arr<double>({B, N}));

  LMRefineOutputs r = compute_lm_refine(in, B, eps_pos, eps_ori, lambda_init, max_iters,
                                        model, diagnostics, precision, stag_patience, stag_rel,
                                        ws->get(), qa.mutable_data());
  py::dict o;
  o["joint_config"] = std::move(qa);
  o["precision"] = out_f32 ? "float32" : "float64";
  o["kernel_ms"] = r.kernel_ms;
  o["position_errors"] = arr_from(r.pos_err, {B, K});
  o["orientation_errors"] = arr_from(r.ori_err, {B, K});
  o["cost"] = arr_from(r.cost, {B});
  o["success"] = barr_from(r.success, {B});
  o["active_target_mask"] = arr_from((const unsigned int*)active.data(), {B});
  if (diagnostics) {
    o["lm_iterations"] = arr_from(r.lm_iterations, {B});
    o["lm_trials"] = arr_from(r.lm_trials, {B});
    o["line_searches"] = arr_from(r.line_searches, {B});
    o["accepted_lm_steps"] = arr_from(r.accepted_steps, {B});
    o["rejected_lm_steps"] = arr_from(r.rejected_steps, {B});
    o["iterations"] = arr_from(r.lm_iterations, {B});
    if (return_trace)
      o["trace"] = arr_from(r.trace, {B, r.trace_cap, r.trace_cols});
  }
  return o;
}

py::dict py_incremental_probe(arrd q, py::array_t<int, py::array::c_style|py::array::forcecast> upd_j,
                              arrd upd_v,
                              py::array_t<bool, py::array::c_style|py::array::forcecast> accept,
                              arrd tgt_p, arrd tgt_q, arru active, arrd w_pos, arrd w_ori) {
  auto* model = ensure_robot();
  const int N = grid_num_joints();
  const int K = grid_num_targets();
  const int B = (int)q.shape(0);
  const int M = (upd_j.ndim() == 2) ? (int)upd_j.shape(1) : 0;
  std::vector<unsigned char> acc((size_t)B * (M > 0 ? M : 1), 0);
  if (M > 0) { const bool* a = accept.data(); for (size_t i = 0; i < acc.size(); ++i) acc[i] = a[i]; }

  IncrementalOutputs r = compute_incremental_probe(
      q.data(), upd_j.data(), upd_v.data(), acc.data(), M,
      tgt_p.data(), tgt_q.data(), active.data(), w_pos.data(), w_ori.data(), B, model);

  // 4x4s come back column-major from the kernel; transpose to row-major for numpy.
  auto mk4 = [&](const std::vector<double>& v, int cnt) {
    auto a = make_arr<double>({B, cnt, 4, 4});
    double* o = a.mutable_data();
    for (int b = 0; b < B; ++b)
      for (int f = 0; f < cnt; ++f)
        for (int rr = 0; rr < 4; ++rr)
          for (int cc = 0; cc < 4; ++cc)
            o[((size_t)b*cnt + f)*16 + rr*4 + cc] = v[((size_t)b*cnt + f)*16 + cc*4 + rr];
    return a;
  };
  py::dict o;
  o["joint_config"] = arr_from(r.q, {B, N});
  o["joint_transforms"] = mk4(r.joint_xform, N);
  o["target_transforms"] = mk4(r.target_xform, K);
  o["position_residuals"] = arr_from(r.e_pos, {B, K, 3});
  o["orientation_residuals"] = arr_from(r.e_ori, {B, K, 3});
  o["position_errors"] = arr_from(r.pos_norm, {B, K});
  o["orientation_errors"] = arr_from(r.ori_norm, {B, K});
  o["target_costs"] = arr_from(r.cost, {B, K});
  o["cost_raw"] = arr_from(r.total_cost, {B});
  return o;
}

double py_bench_fk(arrd q, int j, int iters, int mode, arrd tgt_p, arrd tgt_q, arru active,
                   arrd w_pos, arrd w_ori) {
  auto* model = ensure_robot();
  return bench_fk_mode(q.data(), j, iters, mode, tgt_p.data(), tgt_q.data(), active.data(),
                       w_pos.data(), w_ori.data(), (int)q.shape(0), model);
}

py::dict py_coarse_search(py::array q, py::array tgt_p, py::array tgt_q, arru active,
                          py::array w_pos, py::array w_ori,
                          double eps_pos, double eps_ori, double lambda_coord, double h_min,
                          double max_step, int max_iters, int stall_lim, int use_incremental,
                          std::uint64_t seed, bool diagnostics, bool return_trace,
                          const std::string& problems_json_text, const std::string& problem_set_name,
                          int problem_idx, int max_pert_attempts, int precision, PyWorkspace* ws) {
  auto* model = ensure_robot();
  const int N = grid_num_joints();
  const int K = grid_num_targets();
  const int B = (int)q.shape(0);

  const void* cc_model = nullptr;
  const void* cc_env = nullptr;
  if (!problems_json_text.empty() && !problem_set_name.empty()) {
    if (bind_collision_env(problems_json_text.c_str(), problem_set_name.c_str(), problem_idx)) {
      cc_model = collision_model_ptr();
      cc_env = collision_env_ptr();
    }
  }

  const bool in_f32 = q.dtype().is(py::dtype::of<float>());
  SolveInputs in{q.data(), tgt_p.data(), tgt_q.data(), w_pos.data(), w_ori.data(),
                 (const unsigned int*)active.data(), in_f32};

  const bool out_f32 = (precision == 1);
  py::array qa = out_f32 ? py::array(make_arr<float>({B, N})) : py::array(make_arr<double>({B, N}));

  CoarseOutputs r = compute_coarse_search(
      in, B, eps_pos, eps_ori, lambda_coord, h_min, max_step, max_iters, stall_lim,
      use_incremental, seed, model, diagnostics, cc_model, cc_env, max_pert_attempts, precision,
      ws->get(), qa.mutable_data());

  py::dict o;
  o["joint_config"] = std::move(qa);
  o["precision"] = out_f32 ? "float32" : "float64";
  o["kernel_ms"] = r.kernel_ms;
  o["position_errors"] = arr_from(r.pos_err, {B, K});
  o["orientation_errors"] = arr_from(r.ori_err, {B, K});
  o["cost"] = arr_from(r.cost, {B});
  o["success"] = barr_from(r.success, {B});
  o["active_target_mask"] = arr_from((const unsigned int*)active.data(), {B});
  if (diagnostics) {
    o["coarse_iterations"] = arr_from(r.iterations, {B});
    o["accepted_coarse_steps"] = arr_from(r.accepted, {B});
    o["rejected_coarse_steps"] = arr_from(r.rejected, {B});
    o["coarse_perturbations"] = arr_from(r.perturbations, {B});
    o["coarse_max_stall"] = arr_from(r.max_stall, {B});
    o["coarse_perturbation_events"] = arr_from(r.pert_events, {B});
    o["coarse_perturbation_attempts"] = arr_from(r.pert_attempts, {B});
    o["coarse_perturbations_rejected"] = arr_from(r.pert_rejected, {B});
    o["coarse_perturbations_exhausted"] = arr_from(r.pert_exhausted, {B});
    if (return_trace)
      o["trace"] = arr_from(r.trace, {B, r.trace_cap, r.trace_cols});
  }
  return o;
}

py::array_t<bool> py_collision_free(arrd q, const std::string& json, const std::string& set_name,
                                    int idx) {
  ensure_robot();
  const int B = (int)q.shape(0);
  std::vector<unsigned char> v = check_collision_free(q.data(), B, json.c_str(), set_name.c_str(), idx);
  return barr_from(v, {B});
}

PYBIND11_MODULE(_hjcdik, m) {
  m.doc() = "Minimal pybind11 bindings for hjcdik";
  py::class_<PyWorkspace>(m, "Workspace")
      .def(py::init<>())
      .def("stats", &PyWorkspace::stats,
           "cuda_mallocs / cuda_frees / bytes / capacity_B / device. After warm-up, a steady stream "
           "of same-or-smaller solves must not increment cuda_mallocs.");
  m.def("generate_solutions", &py_generate_solutions,
      py::arg("target_pose"),
      py::arg("batch_size") = 2000,
      py::arg("num_solutions") = 1,
      py::arg("collision_free") = false,
      py::arg("problems_json_text") = "",
      py::arg("problem_set_name") = "",
      py::arg("problem_idx") = 0,
      py::arg("refine_fp64") = -1,    // -1=auto (fp64 if num_solutions==1 else fp32); 1=fp64; 0=fp32
      py::arg("write_stats") = false,   // append a row to ik_stats.csv (see scripts/ik_stats_summary.py)
      // Coarse stage. "auto" (default) dispatches on __popc(active_target_mask): a single active
      // target goes LM-only (the coarse search measurably WORSENS K=1 accuracy); two or more use the
      // new multi-target coarse search (LM alone converges 0% on G1 K=4). "legacy" is the old
      // single-target Panda coarse search, kept only for compatibility/ablation -- never a default.
      py::arg("coarse_mode") = "auto",
      py::arg("coarse_iters") = 60,
      py::arg("coarse_incremental") = true);
  m.def("sample_targets", &py_sample_targets,
        py::arg("num_targets"), py::arg("seed") = 0);
  m.def("num_joints", &grid_num_joints);
  m.def("num_frames", &grid_num_frames);
  m.def("num_targets", &grid_num_targets);
  m.def("joint_limits", []() {
    ensure_robot();
    std::vector<double> v = get_joint_limits();
    const int N = grid_num_joints();
    return arr_from(v, {N, 2});
  }, "Per-joint (lower, upper) limits as the solver clamps them: [N, 2].");
  m.def("link_transforms", &py_link_transforms, py::arg("q"));
  m.def("target_transforms", &py_target_transforms, py::arg("q"));
  m.def("target_metadata", &py_target_metadata);
  m.def("_target_residuals_raw", &py_target_residuals,
        py::arg("q"), py::arg("target_positions"), py::arg("target_quaternions"),
        py::arg("active_target_mask"), py::arg("position_weights"), py::arg("orientation_weights"),
        py::arg("position_tol"), py::arg("orientation_tol"));
  m.def("_incremental_probe_raw", &py_incremental_probe,
        py::arg("q"), py::arg("upd_j"), py::arg("upd_v"), py::arg("accept"),
        py::arg("target_positions"), py::arg("target_quaternions"),
        py::arg("active_target_mask"), py::arg("position_weights"), py::arg("orientation_weights"));
  m.def("_bench_fk_raw", &py_bench_fk,
        py::arg("q"), py::arg("j"), py::arg("iters"), py::arg("mode"),
        py::arg("target_positions"), py::arg("target_quaternions"),
        py::arg("active_target_mask"), py::arg("position_weights"), py::arg("orientation_weights"));
  m.def("_normal_equations_raw", &py_normal_equations,
        py::arg("q"), py::arg("target_positions"), py::arg("target_quaternions"),
        py::arg("active_target_mask"), py::arg("position_weights"), py::arg("orientation_weights"));
  m.def("collision_free", &py_collision_free, py::arg("q"), py::arg("problems_json_text"),
        py::arg("problem_set_name"), py::arg("problem_idx") = 0,
        "Exact collision check (self + environment), the same evaluator the solver gates on.");
  m.def("_coarse_search_raw", &py_coarse_search,
        py::arg("q"), py::arg("target_positions"), py::arg("target_quaternions"),
        py::arg("active_target_mask"), py::arg("position_weights"), py::arg("orientation_weights"),
        py::arg("eps_pos"), py::arg("eps_ori"), py::arg("lambda_coord"), py::arg("h_min"),
        py::arg("max_step"), py::arg("max_iters"), py::arg("stall_lim"), py::arg("use_incremental"),
        py::arg("seed"), py::arg("diagnostics"), py::arg("return_trace"),
        py::arg("problems_json_text"), py::arg("problem_set_name"), py::arg("problem_idx"),
        py::arg("max_pert_attempts"), py::arg("precision"), py::arg("workspace"));
  m.def("_lm_refine_raw", &py_lm_refine,
        py::arg("q"), py::arg("target_positions"), py::arg("target_quaternions"),
        py::arg("active_target_mask"), py::arg("position_weights"), py::arg("orientation_weights"),
        py::arg("eps_pos"), py::arg("eps_ori"), py::arg("lambda_init"), py::arg("max_iters"),
        py::arg("diagnostics"), py::arg("return_trace"), py::arg("precision"),
        py::arg("stag_patience"), py::arg("stag_rel"), py::arg("workspace"));
}
