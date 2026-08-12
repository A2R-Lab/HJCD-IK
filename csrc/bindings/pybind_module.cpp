#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <cstdint>
#include "kernel/hjcd_kernel.h"

namespace py = pybind11;

// ---------------------------------------------------------------------------------------------
// Collision sidecar (Checkpoint 3): the validated GPU self-collision sidecar is compiled into this
// module as a separate CUDA TU (src/collision_sidecar.cu). Reach it ONLY through these host
// extern "C" entry points -- no CUDA headers leak into this host-only .cpp. DORMANT unless the
// Python solve wrapper calls them (self_collision_mode="final").
namespace g1sc {
extern "C" void sidecar_upload_sdf(int cid, const short* grid, int n);
extern "C" void sidecar_upload_convex(const double* verts, int n_verts);
extern "C" void sidecar_full_check(const float* q, unsigned char* out, int B, float margin);
extern "C" void sidecar_prim_gaps(const float* q, float* gap, int B);
extern "C" void sidecar_cluster_gaps(const float* q, float* gap, int* ev, int B);
extern "C" void sidecar_gjk_gaps(const float* q, float* gap, int* iters, int B);
extern "C" void sidecar_fk_batch(const float* q, float* T, int B);
}  // namespace g1sc
namespace g1sc { extern "C" int sidecar_model_uploaded(); }
static void require_sidecar_model(const char* who) {
  if (!g1sc::sidecar_model_uploaded())
    throw std::runtime_error(std::string(who) + ": self-collision model not uploaded; call "
                             "hjcdik._ensure_self_collision_sidecar() first");
}
extern "C" const char* sidecar_hash_str(int which);
extern "C" int sidecar_model_int(int which);
namespace g1sc { extern "C" int sidecar_ws_nalloc(); }

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
                      int stag_patience, double stag_rel, PyWorkspace* ws,
                      int seeds_per_problem) {
  auto* model = ensure_robot();
  const int N = grid_num_joints();
  const int K = grid_num_targets();
  const int B = (int)q.shape(0);                 // candidates
  const int P = (int)active.shape(0);            // problems: targets/weights/mask are [P, ...]
  const int S = seeds_per_problem >= 1 ? seeds_per_problem : 1;
  if ((long long)P * S != (long long)B)
    throw std::invalid_argument("candidates B != num_problems * seeds_per_problem");

  // Python guarantees these are C-contiguous and all of ONE dtype: float32 for an fp32 solve
  // (direct H2D, zero host conversion), float64 otherwise (compatibility, narrowed once).
  const bool in_f32 = q.dtype().is(py::dtype::of<float>());
  SolveInputs in{q.data(), tgt_p.data(), tgt_q.data(), w_pos.data(), w_ori.data(),
                 (const unsigned int*)active.data(), in_f32, P, S};

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
  o["active_target_mask"] = arr_from((const unsigned int*)active.data(), {P});
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
                          int problem_idx, int max_pert_attempts, int precision, PyWorkspace* ws,
                          int seeds_per_problem,
                          int hard_self_collision, int hard_top_k, double hard_margin,
                          int hard_max_reseed, bool hard_diagnostics, int hard_oracle_every,
                          int hard_reseed_mode, int hard_reseed_candidates,
                          int hard_reseed_rounds, std::vector<double> hard_reseed_scales) {
  auto* model = ensure_robot();
  const int N = grid_num_joints();
  const int K = grid_num_targets();
  const int B = (int)q.shape(0);
  const int P = (int)active.shape(0);
  const int S = seeds_per_problem >= 1 ? seeds_per_problem : 1;
  if ((long long)P * S != (long long)B)
    throw std::invalid_argument("candidates B != num_problems * seeds_per_problem");

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
                 (const unsigned int*)active.data(), in_f32, P, S};
  // Stage 3D/3E. Validated HERE, before any CUDA work: an out-of-range top-K that reached the
  // kernel would silently clamp instead of telling the caller their request was not honoured.
  if (hard_self_collision) {
    if (!hjcd_hard_available())
      throw std::invalid_argument("self_collision_mode='hard': this build's robot does not match "
                                  "the G1 self-collision sidecar model");
    if (hard_top_k < 1 || hard_top_k > hjcd_hard_max_top_k())
      throw std::invalid_argument("collision_top_k out of range 1.." +
                                  std::to_string(hjcd_hard_max_top_k()));
  }
  in.hard_self_collision = hard_self_collision;
  in.hard_top_k = hard_top_k;
  in.hard_margin = (float)hard_margin;
  in.hard_max_reseed = hard_max_reseed;
  in.hard_diagnostics = hard_diagnostics ? 1 : 0;
  // The oracle only records into the counter block, so asking for it implies collecting counters.
  in.hard_oracle_every = hard_oracle_every;
  if (hard_oracle_every > 0) in.hard_diagnostics = 1;
  if (hard_self_collision) {
    if (hard_reseed_mode != 0 && hard_reseed_mode != 1)
      throw std::invalid_argument("collision_reseed_mode must be 0 (legacy kick) or 1 (generator)");
    if (hard_reseed_candidates < 1 || hard_reseed_candidates > 256)
      throw std::invalid_argument("collision_reseed_candidates out of range 1..256");
    if (hard_reseed_rounds < 1 || hard_reseed_rounds > 16)
      throw std::invalid_argument("collision_reseed_rounds out of range 1..16");
    if (hard_reseed_scales.empty() || hard_reseed_scales.size() > 8)
      throw std::invalid_argument("collision_reseed_scales must have 1..8 entries");
    for (double v : hard_reseed_scales)
      if (!(v > 0.0) || v > 4.0)
        throw std::invalid_argument("collision_reseed_scales entries must be in (0, 4]");
  }
  in.hard_reseed_mode = hard_reseed_mode;
  in.hard_reseed_candidates = hard_reseed_candidates;
  in.hard_reseed_rounds = hard_reseed_rounds;
  in.hard_reseed_n_scales = (int)hard_reseed_scales.size();
  for (size_t i = 0; i < hard_reseed_scales.size() && i < 8; ++i)
    in.hard_reseed_scales[i] = (float)hard_reseed_scales[i];

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
  o["active_target_mask"] = arr_from((const unsigned int*)active.data(), {P});
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
  // Stage 3D/3E report. Present ONLY when hard mode actually ran, so off/final dicts are unchanged
  // key-for-key (asserted by test_off_returns_no_hard_keys).
  if (r.hard_ran) {
    py::dict h;
    h["initially_free"] = r.hard_initial_free;
    h["initially_colliding"] = r.hard_initial_colliding;
    h["reseed_attempts"] = r.hard_reseed_attempts;
    h["recovered"] = r.hard_recovered;
    h["seed_failures"] = r.hard_seed_failures;
    h["init_ms"] = r.hard_init_ms;
    h["ws_nalloc"] = hjcd_hard_ws_nalloc();
    h["ws_capacity"] = hjcd_hard_ws_capacity();
    h["reseed_ws_capacity"] = hjcd_hard_reseed_ws_capacity();
    h["reseed_ws_nalloc"] = hjcd_hard_reseed_ws_nalloc();
    h["reseed_rounds_run"] = r.hard_reseed_rounds_run;
    // NOT "candidates_checked": that key already means "IK candidates the LM check looked at".
    // Two different quantities under one name silently overwrote each other.
    h["reseed_candidates_checked"] = r.hard_candidates_checked;
    h["selected_perturb"] = r.hard_sel_perturb;
    h["selected_nominal"] = r.hard_sel_nominal;
    h["selected_broad"] = r.hard_sel_broad;
    h["reseed_gen_ms"] = r.hard_gen_ms;
    h["reseed_check_ms"] = r.hard_check_ms;
    h["reseed_select_ms"] = r.hard_select_ms;
    h["reseed_verify_ms"] = r.hard_verify_ms;
    o["hard_qfree"] = arr_from(r.hard_qfree, {B, (int)(r.hard_qfree.size() / (size_t)B)});
    o["hard_flags"] = arr_from(r.hard_flags, {B});   // BITFIELD -- never barr_from(), see below
    if (!r.hard_counters.empty())
      o["hard_counters"] = arr_from(r.hard_counters, {B, r.hard_ctr_stride});
    o["hard"] = std::move(h);
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

// Milestone 3: batched-problem solve with on-device per-problem top-1 selection. Consumes already
// canonicalized [B,N] seeds + [P,K,...] problem data + a [B] per-candidate dispatch flag. Returns
// only the selected [P,1,...] outputs and per-problem summaries (and, if return_all, [P,S,...]).
py::dict py_solve_problems(py::array q, py::array tgt_p, py::array tgt_q, arru active,
                           py::array w_pos, py::array w_ori,
                           py::array_t<unsigned char, py::array::c_style | py::array::forcecast> use_coarse,
                           bool run_coarse,
                           double eps_pos, double eps_ori,
                           double lambda_coord, double h_min, double max_step,
                           int coarse_iters, int coarse_stall_lim, int use_incremental,
                           std::uint64_t seed, int max_pert_attempts,
                           double lambda_init, int lm_iters, int stag_patience, double stag_rel,
                           int num_solutions, int precision, bool return_all,
                           const std::string& problems_json_text, const std::string& problem_set_name,
                           int problem_idx, PyWorkspace* ws,
                           py::array base_p, py::array base_q, py::array base_diag,
                           bool base_update_enabled, int base_update_interval,
                           double base_damping, double base_step_scale,
                           double base_damping_scale_p, double base_damping_scale_R,
                           double base_max_translation_step, double base_max_rotation_step,
                           std::array<double,3> base_position_lower,
                           std::array<double,3> base_position_upper,
                           py::array_t<unsigned int, py::array::c_style | py::array::forcecast> problem_seeds) {
  auto* model = ensure_robot();
  const int N = grid_num_joints();
  const int K = grid_num_targets();
  const int B = (int)q.shape(0);
  const int P = (int)active.shape(0);
  const int S = (P > 0) ? B / P : 0;
  if ((long long)P * S != (long long)B) throw std::invalid_argument("B != P*S");

  const void* cc_model = nullptr; const void* cc_env = nullptr;
  if (!problems_json_text.empty() && !problem_set_name.empty())
    if (bind_collision_env(problems_json_text.c_str(), problem_set_name.c_str(), problem_idx)) {
      cc_model = collision_model_ptr(); cc_env = collision_env_ptr();
    }

  const bool in_f32 = q.dtype().is(py::dtype::of<float>());
  SolveInputs in{q.data(), tgt_p.data(), tgt_q.data(), w_pos.data(), w_ori.data(),
                 (const unsigned int*)active.data(), in_f32, P, S};
  // 5D.14c: semantic per-problem RNG roots (rng_policy_version = semantic_problem_rng_v2).
  // Empty array => nullptr => the kernel's slot-derived fallback, which is NOT authoritative.
  if (problem_seeds.size() > 0) {
    if ((int)problem_seeds.size() != in.num_problems)
      throw std::invalid_argument("problem_seeds must have length num_problems (P)");
    in.problem_seeds = problem_seeds.data();
  }
  // Floating base: candidate-level [B,3]/[B,4], or BOTH empty for a fixed-base solve (which
  // leaves in.base_* null and every downstream path bit-identical). Shapes, dtype and
  // quaternion norms are validated in hjcdik/__init__.py, like every other input.
  if (base_p.size() && base_q.size()) {
    // IN/OUT: the seed base goes down, the optimized base comes back in the SAME buffers, which
    // is why these must be mutable and why the caller gets them returned below.
    in.base_p = base_p.mutable_data();
    in.base_q = base_q.mutable_data();
    if (base_diag.size()) {
      if ((size_t)base_diag.size() != (size_t)B * 3)
        throw std::invalid_argument("base_diag must be [B,3] int32");
      in.base_diag = base_diag.mutable_data();
    }
    in.base_update_enabled = base_update_enabled ? 1 : 0;
    if (base_update_interval < 1)
      throw std::invalid_argument("base_update_interval must be >= 1");
    if (!(base_damping >= 0.0))
      throw std::invalid_argument("base_damping must be >= 0");
    if (!(base_step_scale > 0.0))
      throw std::invalid_argument("base_step_scale must be > 0");
    // Strictly positive: they are divided by (as s^-2) and they are what makes lambda*D positive
    // definite, which is what lets the kernel drop the zero-diagonal pin. Zero would reintroduce
    // a singular H_lambda through the back door.
    if (!(base_damping_scale_p > 0.0) || !std::isfinite(base_damping_scale_p))
      throw std::invalid_argument("base_damping_scale_p must be a positive finite length (metres)");
    if (!(base_damping_scale_R > 0.0) || !std::isfinite(base_damping_scale_R))
      throw std::invalid_argument("base_damping_scale_R must be a positive finite angle (radians)");
    for (int i = 0; i < 3; ++i)
      if (!(base_position_lower[i] <= base_position_upper[i]))
        throw std::invalid_argument("base_position_lower must be <= base_position_upper");
    in.base_update_interval = base_update_interval;
    in.base_damping = base_damping;
    in.base_damping_scale_p = base_damping_scale_p;
    in.base_damping_scale_R = base_damping_scale_R;
    in.base_step_scale = base_step_scale;
    in.base_max_translation_step = base_max_translation_step;
    in.base_max_rotation_step = base_max_rotation_step;
    for (int i = 0; i < 3; ++i) {
      in.base_position_lower[i] = base_position_lower[i];
      in.base_position_upper[i] = base_position_upper[i];
    }
  } else if (base_update_enabled) {
    throw std::invalid_argument(
        "base_update_enabled=True needs base_positions/base_quaternions: there is no base to move");
  }

  const bool out_f32 = (precision == 1);
  const int M = (num_solutions >= 1) ? num_solutions : 1;
  // selected config, shape [P, M, N]
  py::array sel_q = out_f32 ? py::array(make_arr<float>({P, M, N}))
                            : py::array(make_arr<double>({P, M, N}));
  py::array all_q;                                          // [P, S, N] only when return_all
  void* all_ptr = nullptr;
  if (return_all) {
    all_q = out_f32 ? py::array(make_arr<float>({P, S, N})) : py::array(make_arr<double>({P, S, N}));
    all_ptr = all_q.mutable_data();
  }

  SolveProblemsOutputs r = compute_solve_problems(
      in, B, num_solutions, eps_pos, eps_ori, lambda_coord, h_min, max_step, coarse_iters,
      coarse_stall_lim, use_incremental, seed, max_pert_attempts, lambda_init, lm_iters,
      stag_patience, stag_rel, model, precision, use_coarse.data(), run_coarse,
      cc_model, cc_env, return_all, ws->get(), sel_q.mutable_data(), all_ptr);

  py::dict o;
  o["joint_config"] = std::move(sel_q);
  // Raw M4 surface: per-CANDIDATE [B,3]/[B,4], not gathered to [P,M,...]. Pair a solution with
  // its base via selected_seed_ids: base[p*S + selected_seed_ids[p,m]]. M5 does the gather.
  if (base_p.size() && base_q.size()) {
    o["base_position"] = base_p;
    o["base_quaternion"] = base_q;
    if (base_diag.size()) o["base_diag"] = base_diag;
  }                    // [P,M,N]
  o["position_errors"] = arr_from(r.sel_pe, {P, M, K});
  o["orientation_errors"] = arr_from(r.sel_oe, {P, M, K});
  o["cost_lm"] = arr_from(r.sel_cost, {P, M});
  o["cost_physical"] = arr_from(r.sel_ephys, {P, M});
  o["selected_seed_ids"] = arr_from(r.sel_seed, {P, M});
  o["success"] = barr_from(r.sel_succ, {P, M});
  o["valid"] = barr_from(r.sel_valid, {P, M});
  o["problem_success"] = barr_from(r.prob_success, {P});
  o["num_solved"] = arr_from(r.num_solved, {P});
  o["num_valid"] = arr_from(r.num_valid, {P});
  o["active_masks"] = arr_from((const unsigned int*)active.data(), {P});
  o["precision"] = out_f32 ? "float32" : "float64";
  o["coarse_kernel_ms"] = r.coarse_ms;
  o["lm_kernel_ms"] = r.lm_ms;
  o["select_kernel_ms"] = r.select_ms;
  o["collision_enabled"] = r.cc_enabled;
  if (r.cc_enabled) {
    o["collision_free"] = barr_from(r.sel_cfree, {P, M});
    o["used_coarse_fallback"] = barr_from(r.sel_fb, {P, M});
    o["num_collision_free"] = arr_from(r.num_cfree, {P});
    o["num_lm_colliding"] = arr_from(r.num_lm_coll, {P});
    o["num_coarse_fallbacks"] = arr_from(r.num_fb, {P});
    o["num_infeasible"] = arr_from(r.num_infeas, {P});
  }
  if (return_all) {
    o["all_joint_config"] = std::move(all_q);              // [P,S,N]
    o["all_position_errors"] = arr_from(r.all_pe, {P, S, K});
    o["all_orientation_errors"] = arr_from(r.all_oe, {P, S, K});
    o["all_cost_lm"] = arr_from(r.all_cost, {P, S});
    o["all_success"] = barr_from(r.all_succ, {P, S});
    if (r.cc_enabled) {
      o["all_collision_free"] = barr_from(r.all_cfree, {P, S});
      o["all_used_coarse_fallback"] = barr_from(r.all_fb, {P, S});
    }
  }
  return o;
}

// ---------------------------------------------------------------------------------------------
// Collision-sidecar pybind wrappers (Checkpoint 3B). Thin: marshal numpy <-> the extern "C" API.
static void py_sidecar_upload_sdf(int cid,
    py::array_t<int16_t, py::array::c_style | py::array::forcecast> g) {
  g1sc::sidecar_upload_sdf(cid, g.data(), (int)g.size());
}
static void py_sidecar_upload_convex(
    py::array_t<double, py::array::c_style | py::array::forcecast> v) {
  if (v.ndim() != 2 || v.shape(1) != 3) throw std::invalid_argument("convex verts must be [N,3]");
  g1sc::sidecar_upload_convex(v.data(), (int)v.shape(0));
}
static py::array_t<uint8_t> py_sidecar_full_check(
    py::array_t<float, py::array::c_style | py::array::forcecast> q, float margin) {
  require_sidecar_model("sidecar_full_check");
  if (q.ndim() != 2 || q.shape(1) != sidecar_model_int(0))
    throw std::invalid_argument("q must be [B,29]");
  const int B = (int)q.shape(0), NP = sidecar_model_int(2);
  auto out = py::array_t<uint8_t>({(py::ssize_t)B, (py::ssize_t)NP});
  g1sc::sidecar_full_check(q.data(), out.mutable_data(), B, margin);
  return out;
}
static py::array_t<float> py_sidecar_prim_gaps(
    py::array_t<float, py::array::c_style | py::array::forcecast> q) {
  const int B = (int)q.shape(0), NP = sidecar_model_int(2);
  auto g = py::array_t<float>({(py::ssize_t)B, (py::ssize_t)NP});
  g1sc::sidecar_prim_gaps(q.data(), g.mutable_data(), B); return g;
}
static py::tuple py_sidecar_cluster_gaps(
    py::array_t<float, py::array::c_style | py::array::forcecast> q) {
  require_sidecar_model("sidecar_cluster_gaps");
  const int B = (int)q.shape(0), NP = sidecar_model_int(2);
  auto g = py::array_t<float>({(py::ssize_t)B, (py::ssize_t)NP});
  auto e = py::array_t<int32_t>({(py::ssize_t)B, (py::ssize_t)NP});
  g1sc::sidecar_cluster_gaps(q.data(), g.mutable_data(), e.mutable_data(), B);
  return py::make_tuple(g, e);
}
static py::tuple py_sidecar_gjk_gaps(
    py::array_t<float, py::array::c_style | py::array::forcecast> q) {
  require_sidecar_model("sidecar_gjk_gaps");
  const int B = (int)q.shape(0), NG = sidecar_model_int(5);
  auto g = py::array_t<float>({(py::ssize_t)B, (py::ssize_t)NG});
  auto it = py::array_t<int32_t>({(py::ssize_t)B, (py::ssize_t)NG});
  g1sc::sidecar_gjk_gaps(q.data(), g.mutable_data(), it.mutable_data(), B);
  return py::make_tuple(g, it);
}
static py::array_t<float> py_sidecar_fk(
    py::array_t<float, py::array::c_style | py::array::forcecast> q) {
  const int B = (int)q.shape(0), NL = sidecar_model_int(1);
  auto T = py::array_t<float>({(py::ssize_t)B, (py::ssize_t)NL, (py::ssize_t)16});
  g1sc::sidecar_fk_batch(q.data(), T.mutable_data(), B); return T;
}
static py::dict py_sidecar_model_info() {
  py::dict d, h;
  const char* names[8] = {"urdf", "joint_order", "proxy_yaml", "torso_sdf", "pelvis_sdf",
                          "convex", "pair_policy", "typed_piece"};
  for (int i = 0; i < 8; ++i) h[names[i]] = std::string(sidecar_hash_str(i));
  d["hashes"] = h;
  d["sidecar_compiled"] = true;
  d["supported_modes"] = py::make_tuple("off", "final");
  d["hard_enabled"] = false;
  d["shoulder_torso_gjk"] = true;      // Checkpoint 3C.1: exact convex/GJK for shoulder_yaw<->torso
  d["fused_final_path"] = true;        // persistent-buffer full-check (no per-call malloc/free)
  d["n_joints"] = sidecar_model_int(0); d["n_links"] = sidecar_model_int(1);
  d["n_checked_pairs"] = sidecar_model_int(2); d["n_gjk_pairs"] = sidecar_model_int(5);
  d["n_clusters"] = sidecar_model_int(6); d["n_convex_verts"] = sidecar_model_int(7);
  return d;
}

PYBIND11_MODULE(_hjcdik, m) {
  m.doc() = "Minimal pybind11 bindings for hjcdik";
  // collision sidecar (Checkpoint 3): self-collision full-check + diagnostics + model info
  m.def("sidecar_upload_sdf", &py_sidecar_upload_sdf, py::arg("cid"), py::arg("grid_i16"));
  m.def("sidecar_upload_convex", &py_sidecar_upload_convex, py::arg("verts"));
  m.def("sidecar_full_check", &py_sidecar_full_check, py::arg("q"), py::arg("margin") = 0.0f);
  m.def("sidecar_prim_gaps", &py_sidecar_prim_gaps, py::arg("q"));
  m.def("sidecar_cluster_gaps", &py_sidecar_cluster_gaps, py::arg("q"));
  m.def("sidecar_gjk_gaps", &py_sidecar_gjk_gaps, py::arg("q"));
  m.def("sidecar_fk", &py_sidecar_fk, py::arg("q"));
  m.def("sidecar_model_info", &py_sidecar_model_info);
  m.def("hard_available", []() { return hjcd_hard_available() != 0; });
  m.def("hard_max_top_k", []() { return hjcd_hard_max_top_k(); });
  m.def("hard_ws_nalloc", []() { return hjcd_hard_ws_nalloc(); });
  m.def("hard_ws_capacity", []() { return hjcd_hard_ws_capacity(); });
  m.def("hard_ctr_stride", []() { return hjcd_hard_ctr_stride(); });
  m.def("hard_ws_release", []() { hjcd_hard_ws_release(); });
  m.def("hard_reseed_ws_capacity", []() { return hjcd_hard_reseed_ws_capacity(); });
  m.def("hard_reseed_ws_nalloc", []() { return hjcd_hard_reseed_ws_nalloc(); });
  // Last reseed round's candidate arena -> (cand_q, cand_free, cand_comp, cand_dist, fail_idx, sel)
  m.def("hard_reseed_dump", [](int F, int R) {
    const int FR = F * R, J = sidecar_model_int(0);
    if (FR <= 0 || hjcd_hard_reseed_ws_capacity() <= 0) return py::tuple(py::make_tuple());
    auto cq = make_arr<float>({FR, J});
    auto cf = make_arr<uint8_t>({FR});
    auto cc = make_arr<uint8_t>({FR});
    auto cd = make_arr<float>({FR});
    auto fi = make_arr<int32_t>({F});
    auto se = make_arr<int32_t>({F});
    hjcd_hard_reseed_dump(cq.mutable_data(), cf.mutable_data(), cc.mutable_data(),
                          cd.mutable_data(), fi.mutable_data(), se.mutable_data(), FR, F);
    return py::tuple(py::make_tuple(cq, cf, cc, cd, fi, se));
  }, py::arg("F"), py::arg("R"));
  // Committed hard-mode state, for the invariant tests. Returns (qc, qfree, flags, Tf, Td).
  m.def("hard_dump", []() {
    const int B = hjcd_hard_ws_capacity();
    if (B <= 0) return py::tuple(py::make_tuple());
    const int J = sidecar_model_int(0), L = sidecar_model_int(1) * 16;
    auto qc = make_arr<float>({B, J}), qf = make_arr<float>({B, J});
    auto fl = make_arr<uint8_t>({B});
    auto Tf = make_arr<float>({B, L});
    auto Td = make_arr<double>({B, L});
    hjcd_hard_dump(qc.mutable_data(), qf.mutable_data(), fl.mutable_data(),
                   Tf.mutable_data(), Td.mutable_data(), B);
    return py::tuple(py::make_tuple(qc, qf, fl, Tf, Td));
  });
  m.def("sidecar_ws_nalloc", []() { return g1sc::sidecar_ws_nalloc(); },
        "full-check workspace (re)allocation count -- 0 growth after warm-up (fused path proof)");
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
        py::arg("max_pert_attempts"), py::arg("precision"), py::arg("workspace"),
        py::arg("seeds_per_problem") = 1,
        // Stage 3D/3E. Defaulted so every existing _coarse_search_raw call is unchanged.
        py::arg("hard_self_collision") = 0, py::arg("hard_top_k") = 3,
        py::arg("hard_margin") = 0.0, py::arg("hard_max_reseed") = 8,
        py::arg("hard_diagnostics") = false, py::arg("hard_oracle_every") = 0,
        py::arg("hard_reseed_mode") = 1, py::arg("hard_reseed_candidates") = 16,
        py::arg("hard_reseed_rounds") = 2,
        py::arg("hard_reseed_scales") = std::vector<double>{0.10, 0.20, 0.35, 0.50});
  m.def("_lm_refine_raw", &py_lm_refine,
        py::arg("q"), py::arg("target_positions"), py::arg("target_quaternions"),
        py::arg("active_target_mask"), py::arg("position_weights"), py::arg("orientation_weights"),
        py::arg("eps_pos"), py::arg("eps_ori"), py::arg("lambda_init"), py::arg("max_iters"),
        py::arg("diagnostics"), py::arg("return_trace"), py::arg("precision"),
        py::arg("stag_patience"), py::arg("stag_rel"), py::arg("workspace"), py::arg("seeds_per_problem") = 1);
  m.def("_solve_problems_raw", &py_solve_problems,
        py::arg("q"), py::arg("target_positions"), py::arg("target_quaternions"),
        py::arg("active_target_mask"), py::arg("position_weights"), py::arg("orientation_weights"),
        py::arg("use_coarse"), py::arg("run_coarse"),
        py::arg("eps_pos"), py::arg("eps_ori"), py::arg("lambda_coord"), py::arg("h_min"),
        py::arg("max_step"), py::arg("coarse_iters"), py::arg("coarse_stall_lim"),
        py::arg("use_incremental"), py::arg("seed"), py::arg("max_pert_attempts"),
        py::arg("lambda_init"), py::arg("lm_iters"), py::arg("stag_patience"), py::arg("stag_rel"),
        py::arg("num_solutions"), py::arg("precision"), py::arg("return_all"),
        py::arg("problems_json_text"), py::arg("problem_set_name"), py::arg("problem_idx"),
        py::arg("workspace"),
        // Defaulted: every existing _solve_problems_raw call keeps working unchanged.
        py::arg("base_positions") = py::array(), py::arg("base_quaternions") = py::array(),
        py::arg("base_diag") = py::array(),
        py::arg("base_update_enabled") = false, py::arg("base_update_interval") = 1,
        py::arg("base_damping") = 1e-3, py::arg("base_step_scale") = 1.0,
        py::arg("base_damping_scale_p") = 1.0, py::arg("base_damping_scale_R") = 1.0,
        py::arg("base_max_translation_step") = 0.05, py::arg("base_max_rotation_step") = 0.10,
        py::arg("base_position_lower") = std::array<double,3>{-1e30,-1e30,-1e30},
        py::arg("base_position_upper") = std::array<double,3>{ 1e30, 1e30, 1e30},
        // 5D.14c: [P] uint32 semantic per-problem RNG seeds. Empty => legacy slot-derived
        // fallback; the production planner MUST pass these.
        py::arg("problem_seeds") = py::array_t<unsigned int>());
}
