#pragma once
// Fixed-base multi-target evaluation and LM for TREE models (humanoid-v1 M3). Host API only.
// Layouts (all host, C-contiguous, batch-major; B configurations, K targets, N joints):
//   q       [B,N]           joint configurations (GRiD order)
//   target  [B,K,7]         desired base-frame poses (x,y,z, qw,qx,qy,qz); quaternions unit
//   active  [B]             uint32 target bitmask
//   w_pos, w_ori [B,K]      >= 0
//   mode    [B,K]           0 NONE, 1 AXIS, 2 FULL, 3 AXIS_BOUNDED_TWIST
//   axis    [B,K,3]         target-local unit axis (AXIS / BOUNDED)
//   twist   [B,K,5]         bounded-twist spec (grip axis in target frame [3], lo, hi)
//   tool    [K,16] or null  per-call EFFECTIVE tools, COLUMN-major 4x4 (cell 4*col+row); REPLACES
//                           the generated model tool when given (never composed with it)

namespace hjcd_tree {

// Problem data is indexed by PROBLEM p; candidate c = p*seeds_per_problem + s reads problem p.
// tool_per_problem: 0 => one [K,16] table for the whole call; 1 => [P,K,16] (each problem its own).
template<typename T>
struct ProblemIn {
    const T* target; const unsigned int* active; const T* w_pos; const T* w_ori;
    const int* mode; const T* axis; const T* twist; const T* tool;
    int seeds_per_problem = 1;
    int tool_per_problem = 0;
    // Floating base (M4b): per-CANDIDATE world pose [B,7] = (x,y,z, qw,qx,qy,qz), unit quaternion (normalised
    // on the host). nullptr => fixed base: targets are used verbatim (M4a path, bit-identical).
    const T* base = nullptr;
};

// Diagnostic evaluation at q (no optimisation). Outputs (all [B,...]):
//   pose [K,16] row-major target frames; e [K,6] (e_p, e_R); pn/on [K]; J [K*6,N] (d prediction/dq);
//   scale [K*6]; A_raw [N,N], b_raw [N] (unit scaling); A [N,N], b [N] (row-scaled), colnorm2 [N];
//   pinned [1] bitmask; Ap [N,N], bp [N] after the null-column pin; dq [N] = posv(Ap + lam diag Ap, bp); fail [1]
template<typename T>
struct EvalOut {
    T *pose, *e, *pn, *on, *J, *scale, *A_raw, *b_raw, *A, *b, *colnorm2, *Ap, *bp, *dq;
    unsigned int* pinned; int* fail;
};
template<typename T>
void evaluate(const T* q, int batch, const ProblemIn<T>& in, T lambda, EvalOut<T>& out);

// Base update = bbc53a9 base_update_warp (position-only 6-DoF damped GN, world-frame left perturbation),
// attempted after the joint step every `base_interval` iterations while the joint LM has not stopped.
struct LMConfig { double eps_pos, eps_ori, lambda_init, stag_rel; int k_max, stag_patience, warps_per_block, repeats, uniform_lambda;
                  int base_enabled = 0, base_interval = 1;
                  double base_damping = 1e-3, base_scale_p = 1.0, base_scale_R = 1.0, base_step_scale = 1.0,
                         base_max_translation = 0.05, base_max_rotation = 0.10;   // <= 0 => no clip
                  double base_lo[3] = {-1e30, -1e30, -1e30}, base_hi[3] = {1e30, 1e30, 1e30}; };
template<typename T>
struct LMOut {
    T *x, *pn0, *on0, *pn, *on, *cost, *ephys0, *ephys;
    unsigned char* success; int *iters, *accepted, *solve_fail, *kicks;
    T* base = nullptr;          // [B,7] final (best-state) base; written only with a floating base
    int* base_diag = nullptr;   // [B,4] attempted, accepted, numerical failures, rejected (not improving)
};
template<typename T>
void lm(const T* q0, int batch, const ProblemIn<T>& in, const LMConfig& cfg, LMOut<T>& out, float* kernel_ms);

// Batched fixed-base solve (M4a): P problems x S seeds, one LM launch over all P*S candidates, then a
// segmented top-M per problem. Ranking (port of bbc53a9 segmented_topM_kernel, collision terms removed):
//   class 0 = finite q and every ACTIVE target within (eps_pos, eps_ori) on the reported errors,
//   class 1 = finite but not within, class 2 = non-finite q or error;
//   key = (class, E_phys in double, seed index s); M rounds of masked argmin => distinct seeds.
//   A slot whose winner is class 2 is a PAD: seed -1, errors +inf, valid 0, q = the problem's first seed
//   (zeros if non-finite). Counts: num_solved = #class0, num_valid = #class<=1, num_converged = #LM success.
template<typename T>
struct SelectOut {
    T *q, *pe, *oe, *cost; double* ephys; int* seed; unsigned char *exact, *valid;
    int *num_solved, *num_valid, *num_converged;
    T* base = nullptr;          // [P,M,7] selected candidate's base (pads: the problem's first seed's base)
};
struct BatchTiming { float upload_ms, lm_ms, select_ms, download_ms, init_ms, total_ms; float coarse_ms = 0.f; };

// Multi-target coarse search (M5): port of bbc53a9 coarse_search_mt_kernel (no collision / hard mode).
// One warp per candidate, the base held FIXED (targets re-expressed once). Per iteration: row scales + full
// re-score; every joint proposes delta = g/(h + lambda_coord) on the row-scaled multi-target objective (clipped
// to max_step and the joint limits; invalid if h <= h_min or the predicted gain 2 delta g - delta^2 h <= 0);
// the warp's best predicted gain (ties -> lower joint) is applied with a subtree FK update and an incremental
// re-score of the affected targets; kept iff trial + 1e-20 < cost, else restored exactly. Best state on E_phys.
// After stall_lim rejections, every joint is kicked by 0.1*span*(u - 0.5),
// u = semantic_rng(root, 0x2, sample, it, joint, 0) (root = problem_seeds[p] or hash(seed, p);
// sample = candidate_ids[c] or s). Stops at k_max or when every active weighted component is within tolerance.
struct CoarseConfig { int k_max = 120, stall_lim = 5; double lambda_coord = 1e-6, h_min = 1e-9, max_step = 0.35;
                      unsigned long long seed = 0; };
template<typename T>
struct CoarseIO {
    const unsigned char* use_coarse = nullptr;   // [B] host flags (auto: popcount(mask) >= 2)
    const unsigned int* problem_seeds = nullptr; // [P] or null
    const unsigned int* candidate_ids = nullptr; // [B] or null
    // outputs [B,...] (nullable): coarse joints, cost before/after, iterations, kicks, success, pn/on
    T *x = nullptr, *cost0 = nullptr, *cost1 = nullptr, *pn = nullptr, *on = nullptr;
    int *iters = nullptr, *kicks = nullptr; unsigned char* success = nullptr;
};
// Environment collision (M6a). Robot = sphere table [NS,5] (anchor, ox, oy, oz, r): anchor = movable joint whose
// child-link frame carries the sphere (-1 = base frame), offset in that frame. Sphere world centre
//   p_W = R_B (X_anchor o) + p_B      (X = the tree FK the LM uses; B = the candidate's own base, identity if none).
// Environment = GRiD grid_collision rows in the WORLD frame (make_host_environment layouts: spheres [M,4],
// capsules [M,7], cuboids [M,10] centre+wxyz+full dims, planes [M,4] n.p >= d), deep-copied per call into an
// owned grid_collision::DeviceEnvironment<T> (no global state). Per candidate:
//   status 0 FREE / 1 COLLIDING / 2 INVALID (non-finite joints or base: not evaluated);
//   COLLIDING iff some sphere fails GRiD's grid_cc_sphere_in_environment with radius r + margin (squared gap
//   strictly < 0, i.e. clearance < margin; tangency at clearance == margin is FREE);
//   min_clearance = min over spheres of grid_cc_nearest_obstacle (signed: centre distance - r; GRiD's
//   deepest-face convention inside a box), +inf for an empty environment, NaN when INVALID; argmin sphere -1 then.
enum EnvStatus : unsigned char { ENV_FREE = 0, ENV_COLLIDING = 1, ENV_INVALID = 2, ENV_NOT_EVALUATED = 3 };
// Selection policy when the channel is on: GATE (bbc53a9 cc_enabled semantics: status != FREE => class 2, never
// selected), RANK (class first, then status FREE < COLLIDING < INVALID, then E_phys, seed), ANNOTATE (M5 order).
enum EnvSelect : int { ENV_SELECT_GATE = 0, ENV_SELECT_RANK = 1, ENV_SELECT_ANNOTATE = 2 };
template<typename T>
struct EnvCollision {
    const T* spheres_robot = nullptr; int n_robot = 0;           // host [NS,5]
    const T* spheres = nullptr; int n_spheres = 0;               // host environment rows (world frame)
    const T* capsules = nullptr; int n_capsules = 0;
    const T* cuboids = nullptr; int n_cuboids = 0;
    const T* planes = nullptr; int n_planes = 0;
    double margin = 0.0;
    int selection = ENV_SELECT_GATE;
};
template<typename T>
struct EnvOut {   // host [B] (nullable members); per-sphere arrays only from env_collision()
    unsigned char* status = nullptr; T* min_clearance = nullptr; int* argmin_sphere = nullptr;
    T* sphere_world = nullptr /* [B,NS,3] */; T* sphere_clearance = nullptr /* [B,NS] */;
    // selection extras [P]: #FREE, #COLLIDING, #INVALID, #(class 0 && FREE); [P,M]: selected slot's status (pad 3)
    int *num_free = nullptr, *num_colliding = nullptr, *num_invalid = nullptr, *num_solved_free = nullptr;
    unsigned char* sel_status = nullptr; T* sel_min_clearance = nullptr;
    float prep_ms = 0.f, upload_ms = 0.f, kernel_ms = 0.f;
};
// Self-collision (M6b): same sphere table, base-frame centres (a rigid base motion cannot change it; no base input).
// Bodies = sphere anchors (fixed-joint links merged). pairs [NP,2] = CHECKED body pairs (anchors, -1 = root), the
// policy is the caller's (hjcdik._tree_collision.self_collision_pairs: topology only). Pair (i,j) of spheres on a
// checked body pair collides iff GRiD grid_cc_sphere_sphere(c_i, r_i + margin, c_j, r_j) < 0 (clearance < margin,
// strict). min_clearance = exact min of grid_cc_sphere_sphere_signed over all checked sphere pairs (a body-level
// bounding-sphere lower bound only skips body pairs that provably cannot lower min(min, margin)); closest pair
// (i < j, original sphere indices) ties -> lowest (i, j). +inf / (-1,-1) with no eligible pair; INVALID on non-finite q.
template<typename T>
struct SelfCollision {
    const T* spheres_robot = nullptr; int n_robot = 0;   // host [NS,5]
    const int* pairs = nullptr; int n_pairs = 0;         // host [NP,2] anchors
    double margin = 0.0;
    int selection = ENV_SELECT_ANNOTATE;                 // default: report only
};
template<typename T>
struct SelfOut {
    unsigned char* status = nullptr; T* min_clearance = nullptr; int *pair_i = nullptr, *pair_j = nullptr;
    long long* evaluated_pairs = nullptr;                // sphere pairs actually tested per candidate
    int *num_free = nullptr, *num_colliding = nullptr, *num_invalid = nullptr, *num_solved_free = nullptr;  // [P]
    unsigned char* sel_status = nullptr; T* sel_min_clearance = nullptr;                                   // [P,M]
    long long eligible_pairs = 0; int n_bodies = 0, n_body_pairs = 0;
    float prep_ms = 0.f, upload_ms = 0.f, kernel_ms = 0.f;
};
template<typename T>
void self_collision(const T* q, int B, const SelfCollision<T>& sc, SelfOut<T>& out);

// Standalone evaluation of B configurations (base [B,7] or null = identity).
template<typename T>
void env_collision(const T* q, const T* base, int B, const EnvCollision<T>& env, EnvOut<T>& out);
template<typename T>
void solve_batch(const T* seeds, int P, int S, int M, const ProblemIn<T>& in, const LMConfig& cfg,
                 LMOut<T>* raw /* nullable: per-candidate results [P*S,...] */, SelectOut<T>& sel, BatchTiming* timing,
                 const CoarseConfig* coarse = nullptr, CoarseIO<T>* cio = nullptr,
                 const EnvCollision<T>* env = nullptr /* M6a: post-LM, before selection */, EnvOut<T>* eout = nullptr,
                 const SelfCollision<T>* self = nullptr /* M6b: post-LM, independent channel */, SelfOut<T>* sout = nullptr);
// Selection alone on given per-candidate arrays (testing the ranking independently of the LM).
template<typename T>
void select_only(const T* q, const T* pe, const T* oe, const T* cost, const unsigned char* succ, const T* seeds,
                 const unsigned int* active, int P, int S, int M, double eps_pos, double eps_ori, SelectOut<T>& sel,
                 const unsigned char* env_status = nullptr /* [P*S] (M6a) */, const T* env_min_clearance = nullptr,
                 int env_selection = ENV_SELECT_GATE, EnvOut<T>* eout = nullptr,
                 const unsigned char* self_status = nullptr /* [P*S] (M6b) */, int self_selection = ENV_SELECT_ANNOTATE,
                 SelfOut<T>* sout = nullptr);

// One base step at (q, base) with joints held (diagnostic / oracle tests); out [B,65], see hjcd_tree_lm.cu.
template<typename T>
void base_step(const T* q, int batch, const ProblemIn<T>& in, const LMConfig& cfg, T* out);

struct LMKernelInfo { int num_regs, local_bytes, static_shared_bytes, dynamic_shared_bytes, max_threads_per_block; };
// Every joint's coarse proposal at q (diagnostic): out [B,N,6] = (v, delta, predicted gain, coarse cost,
// subtree-vs-full max |target pose diff|, |incremental - full cost| after applying that joint's proposal).
template<typename T> void proposals(const T* q, int batch, const ProblemIn<T>& in, const CoarseConfig& cc, T* out);
template<typename T> LMKernelInfo coarse_kernel_info();
template<typename T> LMKernelInfo lm_kernel_info(int warps_per_block);
template<typename T> LMKernelInfo env_kernel_info();
template<typename T> LMKernelInfo self_kernel_info();

}  // namespace hjcd_tree
