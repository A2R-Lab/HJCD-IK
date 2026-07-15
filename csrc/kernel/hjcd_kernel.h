#pragma once
#include <array>
#include <vector>
#include <string>
#include <cstdint>

namespace grid {
    template<typename T> struct robotModel;
    template<typename T> robotModel<T>* init_robotModel();
}

#define PI 3.14159265358979323846

template<typename T>
struct Result {
    T* joint_config;
    T* pose;
    T* pos_errors;
    T* ori_errors;
    T  elapsed_time;
    int count;
};

// RT = LM-refine compute precision (speed/accuracy knob): RT=double (default) is full fp64;
// RT=float runs FK/Jacobian/residual/line-search in fp32 with the Cholesky solve still fp64.
template<typename T, typename RT = double>
Result<T> generate_ik_solutions(
    T* target_pose,
    const grid::robotModel<T>* d_robotModel,
    int b_size,
    int num_solutions = 1,
    bool collision_free = false,
    const char* problems_json_text = nullptr,
    const char* problem_set_name = nullptr,
    int problem_idx = 0,
    bool write_stats = false,
    // Coarse stage. 0 = auto (dispatch on __popc(active_target_mask): 1 bit -> LM-only,
    // >= 2 bits -> new multi-target coarse), 1 = none, 2 = multi_target, 3 = legacy.
    int coarse_mode = 0,
    int coarse_iters = 60,
    int coarse_incremental = 1
);

template<typename T>
std::vector<std::array<T, 7>> sample_random_target_poses(
    const grid::robotModel<T>* d_robotModel,
    int num_configs,
    std::uint64_t seed
);

// World transforms (column-major 4x4) of every frame, for B configs: B * grid_num_frames() * 16.
template<typename T>
std::vector<T> compute_link_transforms(
    const T* h_q,
    int B,
    const grid::robotModel<T>* d_robotModel
);

// World transforms (column-major 4x4) of every TARGET frame: B * grid_num_targets() * 16.
// One full GRiD FK per config, then one 4x4 compose per target.
template<typename T>
std::vector<T> compute_target_transforms(
    const T* h_q,
    int B,
    const grid::robotModel<T>* d_robotModel
);

// The generated target metadata, read back out of device code (so tests see what kernels see).
struct TargetMetadata {
    int num_targets;
    int num_joints;
    std::vector<int> anchor_jid;                    // [num_targets]
    std::vector<unsigned int> target_ancestor_mask; // [num_targets], bit j = joint j moves target k
    std::vector<unsigned int> joint_target_mask;    // [num_joints],  bit k = joint j affects target k
    std::vector<double> tool_xform;                 // [num_targets*16], column-major
};
TargetMetadata read_target_metadata();

// Multi-target residual diagnostics (Phase 2). Unweighted physical residuals; weights applied once,
// in the cost. Quaternions are WXYZ and unit. Inactive targets are exactly zero everywhere.
struct ResidualOutputs {
    int num_targets;
    std::vector<double> e_pos;       // B x K x 3   (world, metres)  p* - p
    std::vector<double> e_ori;       // B x K x 3   (world, radians) Log(R* R^T)
    std::vector<double> pos_norm;    // B x K
    std::vector<double> ori_norm;    // B x K
    std::vector<double> cost;        // B x K       w_p|e_p|^2 + w_R|e_R|^2
    std::vector<double> cost_raw;    // B           sum over ACTIVE targets
    std::vector<double> cost_norm;   // B           reporting only; never used by an optimizer
    std::vector<unsigned char> success;      // B x K  (0 for inactive: not evaluated)
    std::vector<unsigned char> success_all;  // B      AND over active targets; 0 if mask empty
};

// Bind (and cache) the collision environment for a problem. Both the coarse-search winner gate and
// the post-solve scorers use this same upload. Returns false when the build has no collision or the
// problem has none.
bool bind_collision_env(const char* problems_json, const char* set_name, int idx);
const void* collision_model_ptr();   // grid::robotModel<float>*, or null
const void* collision_env_ptr();     // grid_collision::Environment<float>*, or null

// Exact collision check for a batch, using the SAME evaluator the coarse gate uses.
std::vector<unsigned char> check_collision_free(
    const double* h_q, int B, const char* json, const char* set_name, int idx);

// Persistent device workspace (Phase 0E). Owned by exactly one solver instance; NOT thread-safe.
class HjcdWorkspace;
HjcdWorkspace* hjcd_workspace_new();
void hjcd_workspace_free(HjcdWorkspace* w);
void hjcd_workspace_stats(const HjcdWorkspace* w, size_t* n_malloc, size_t* n_free,
                          size_t* bytes, int* cap_B, int* device);

// Solver inputs. `f32 == true` means the five array pointers are float32 and the fp32 kernels can
// take them by a direct H2D with NO host conversion. float64 stays fully supported (compatibility).
struct SolveInputs {
    const void* q; const void* tgt_p; const void* tgt_q; const void* wp; const void* wo;
    const unsigned int* active;
    bool f32;
};

// Multi-target coarse search (Phase 5): aggregate weighted coordinate Gauss-Newton, one warp per
// problem, exact evaluation of the winning proposal only, Phase-4 incremental FK.
struct CoarseOutputs {
    int num_targets;
    double kernel_ms = 0.0;        // CUDA-event device time for THIS launch (same invocation)
    bool fp32 = false;             // the GPU compute type this was produced with
    std::vector<float> q32;        // B x N   populated ONLY when fp32; the config in CT
    std::vector<double> q;         // B x N   best config seen (always populated, widened if fp32)
    std::vector<double> pos_err;   // B x K   metres
    std::vector<double> ori_err;   // B x K   radians
    std::vector<double> cost;      // B       scaled weighted cost
    std::vector<unsigned char> success;   // B
    // Diagnostics -- present only when diagnostics = true. All DERIVED FROM THE TRACE.
    //   perturbations         kicks RETAINED (passed the collision gate)
    //   pert_events           times the stall threshold fired a kick
    //   pert_attempts         total kick attempts (>= events; bounded by max_pert_attempts each)
    //   pert_rejected         attempts refused by the collision gate
    //   pert_exhausted        events where EVERY attempt collided -> state restored, best_x untouched
    std::vector<int> iterations, accepted, rejected, perturbations, max_stall;
    std::vector<int> pert_events, pert_attempts, pert_rejected, pert_exhausted;
    // B x trace_cap x trace_cols; cols = valid, it, joint, delta, pred, cost_before, cost_after,
    //                                    accepted, stall, perturbed, pert_attempts,
    //                                    pert_collision_rejects, pert_exhausted
    std::vector<double> trace;
    int trace_cap = 0, trace_cols = 0;
};
CoarseOutputs compute_coarse_search(
    const SolveInputs& in,
    int B, double eps_pos, double eps_ori, double lambda_coord, double h_min, double max_step,
    int max_iters, int stall_lim, int use_incremental, unsigned long long seed,
    const grid::robotModel<double>* d_robotModel, bool diagnostics,
    // Exact collision gate on the winning proposal AND on every stall perturbation. Both null =>
    // open-world (gate off). Opaque here so this header stays free of grid_collision (which only
    // exists in a --collision build).
    const void* cc_model, const void* cc_env_ptr,
    int max_pert_attempts, int precision,
    HjcdWorkspace* ws, void* out_q_ct);   // out_q_ct: caller's [B, N] buffer, in the COMPUTE type

// Multi-target LM refine (Phase 3). LM only -- no coarse search. Seeds in, refined configs out.
struct LMRefineOutputs {
    int num_targets;
    double kernel_ms = 0.0;        // CUDA-event device time for THIS launch (same invocation)
    bool fp32 = false;             // the GPU compute type this was produced with
    std::vector<float> q32;        // B x N   populated ONLY when fp32; the config in CT
    std::vector<double> q;         // B x N   refined (best seen, not last; widened if fp32)
    std::vector<double> pos_err;   // B x K   metres
    std::vector<double> ori_err;   // B x K   radians
    std::vector<double> cost;      // B       raw weighted
    std::vector<unsigned char> success;  // B  all active targets within tolerance

    // LM diagnostics. Present ONLY when compute_lm_refine was called with diagnostics = true;
    // otherwise every vector below is empty and no trace buffer was allocated or written.
    // All of these are DERIVED FROM THE TRACE, which is the authoritative source -- see the
    // Phase-3C note in the report. There is deliberately no compact per-problem counter buffer.
    std::vector<int> lm_iterations;   // B  valid trace rows = outer LM linearizations
    std::vector<int> lm_trials;       // B  cumulative damped systems solved (last valid row)
    std::vector<int> line_searches;   // B  cumulative backtracking evaluations (last valid row)
    std::vector<int> accepted_steps;  // B  cumulative accepted (last valid row)
    std::vector<int> rejected_steps;  // B  lm_iterations - accepted_steps
    // Per-outer-iteration trace: B x trace_cap x trace_cols. Column 0 is an EXPLICIT valid flag.
    //   0 valid  1 it  2 lm_trials(cum)  3 accepted(this)  4 accepted(cum)
    //   5 cost   6 max_pos_err  7 max_ori_err  8 lambda  9 line_searches(cum)
    std::vector<double> trace;
    int trace_cap = 0, trace_cols = 0;
};
LMRefineOutputs compute_lm_refine(
    const SolveInputs& in,
    int B, double eps_pos, double eps_ori, double lambda_init, int max_iters,
    const grid::robotModel<double>* d_robotModel, bool diagnostics,
    int precision,                 // 0 = float64, 1 = float32 (GPU compute type)
    // Policy B (stagnation stopping). patience = 0 DISABLES it -- the default, so behaviour is
    // unchanged. Stagnation is measured on E_phys (a tolerance-normalised PHYSICAL error), never on
    // the row-scaled cost: the row scales are re-frozen every iteration, so consecutive scaled costs
    // are not comparable and a "relative improvement" computed from them is meaningless.
    int stag_patience, double stag_rel,
    HjcdWorkspace* ws, void* out_q_ct);

// Incremental (subtree) FK + incremental target cache (Phase 4). Probe entry point: runs a sequence
// of accepted/rejected coordinate updates and dumps the resulting state for comparison vs full FK.
struct IncrementalOutputs {
    int n, num_targets;
    std::vector<double> q;             // B x N
    std::vector<double> joint_xform;   // B x N*16   column-major
    std::vector<double> target_xform;  // B x K*16   column-major
    std::vector<double> e_pos, e_ori;  // B x K x 3
    std::vector<double> pos_norm, ori_norm, cost;   // B x K
    std::vector<double> total_cost;    // B
};
IncrementalOutputs compute_incremental_probe(
    const double* h_q, const int* h_upd_j, const double* h_upd_v,
    const unsigned char* h_accept, int M,
    const double* h_tgt_p, const double* h_tgt_q, const unsigned int* h_active,
    const double* h_wp, const double* h_wo, int B,
    const grid::robotModel<double>* d_robotModel);

// mode 0 = full FK + full recompose + full rescore; mode 1 = incremental subtree path. Returns ms.
double bench_fk_mode(const double* h_q, int j, int iters, int mode,
                     const double* h_tgt_p, const double* h_tgt_q, const unsigned int* h_active,
                     const double* h_wp, const double* h_wo, int B,
                     const grid::robotModel<double>* d_robotModel);

// Accumulated normal equations (Phase 3), for validating against a CPU stacked Jacobian.
// A = sum_k J_k^T W_k J_k   (N x N),   b = sum_k J_k^T W_k e_k   (N)
struct NormalEquations {
    int n;
    std::vector<double> A;   // B x N x N, row-major
    std::vector<double> b;   // B x N
};
NormalEquations compute_normal_equations(
    const double* h_q, const double* h_tgt_p, const double* h_tgt_q,
    const unsigned int* h_active, const double* h_wp, const double* h_wo, int B,
    const grid::robotModel<double>* d_robotModel);

ResidualOutputs compute_target_residuals(
    const double* h_q,        // B x N
    const double* h_tgt_p,    // B x K x 3
    const double* h_tgt_q,    // B x K x 4  (wxyz, unit)
    const unsigned int* h_active,  // B      (bit k = target k active)
    const double* h_wp,       // B x K
    const double* h_wo,       // B x K
    const double* h_eps_p,    // K
    const double* h_eps_o,    // K
    int B,
    const grid::robotModel<double>* d_robotModel
);

void init_joint_limits_constants();

void init_joint_limits_from_grid();

extern "C" int grid_num_joints();

extern "C" int grid_num_frames();

extern "C" int grid_num_targets();

// [2*N] flattened (lower, upper) per joint, as the solver clamps them.
std::vector<double> get_joint_limits();