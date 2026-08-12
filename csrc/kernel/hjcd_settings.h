#pragma once 

// Generated GRiD headers call this helper while defining template bodies.
template<typename T>
__device__ __forceinline__
void mat4_mul(const T* A, const T* B, T* C) {
    T tmp[16];

    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            tmp[c * 4 + r] =
                A[0 * 4 + r] * B[c * 4 + 0] +
                A[1 * 4 + r] * B[c * 4 + 1] +
                A[2 * 4 + r] * B[c * 4 + 2] +
                A[3 * 4 + r] * B[c * 4 + 3];
        }
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        C[i] = tmp[i];
    }
}

#include "grid.cuh"

// External GLASS (full, with the glass::warp:: sub-namespace) at GLOBAL scope. GRiD now
// vendors its own pinned GLASS isolated under grid::glass (see GRiDCodeGenerator
// _lin_alg_helpers.py) AND namespaces its vendored macro guards (GRID_VENDORED_GLASS_*),
// so this no longer ODR-clashes with grid.cuh's vendored copy.
#include "glass.cuh"

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef UNREFINE
#define UNREFINE 0
#endif

#ifndef FULL_WARP_MASK
#define FULL_WARP_MASK 0xFFFFFFFFu
#endif

#ifndef PI
#define PI 3.14159265358979323846
#endif

#include "hjcd_targets.cuh"   // generated: the ordered target set (indices/masks/4x4s, no strings)

namespace hjcd {
    static constexpr int N = grid::NUM_JOINTS;            // actuated joints (7 for Panda)
    // Explicit alias for what N actually means HERE. Note grid::NUM_JOINTS is emitted from
    // GRiD's get_num_pos(), which coincides with the joint count only for a FIXED-base robot
    // (a floating-base G1 would report 36 = 35 vel + 1 quaternion slot, not 30 joints). HJCD
    // is fixed-base by construction -- see FLOATING_BASE_DOF -- so the two agree and N is
    // unambiguously the actuated-joint count.
    static constexpr int NUM_ACTUATED_JOINTS = grid::NUM_JOINTS;
    // The floating base is an HJCD-LEVEL concept layered on a fixed-base GRiD build: it is a
    // rigid transform of the target set, never a solver coordinate and never a GRiD joint.
    // That is deliberate and is what keeps NUM_JOINTS = 29, the uint32 ancestor masks, the
    // lane==joint mapping and the codegen path completely untouched. GRiD will not emit
    // ee_pose_inner_{thread,warp} for a floating-base robot at all
    // (GRiDCodeGenerator/algorithms/_eepose_gradient_hessian.py:2823), so a -f build would not
    // compile no matter how wide the masks were.
    static constexpr int FLOATING_BASE_DOF = 6;           // (3 translation, 3 SO(3) tangent)
    static constexpr int NT = hjcd_gen::NUM_TARGETS;      // generated target count (Panda 1, G1 4)
    static constexpr int XHOM = grid::XHOM_T_COUNT;       // full s_XmatsHom frame storage (16*num_frames)
    static constexpr int FLANGE_JID = N - 1;              // cumulative world transform of the last joint
    // Fixed EE-offset frame inside s_XmatsHom: the frame index where GRiD places the named EE target
    // (its end_effector_pose_inner_<target> epilogue chains s_Xhom[16*EE_FIXED_FRAME_IDX] onto joint
    // FLANGE_JID). This index is ROBOT-SPECIFIC and shifts with DoF, so it is resolved at codegen time
    // and injected into grid.cuh by scripts/codegen/generate_grid.py (Panda grasptarget=10, etc.) — never hardcode.
    static constexpr int GRASP_FIXED_IDX = grid::EE_FIXED_FRAME_IDX;

    // Shared storage for GRiD's runtime topology helpers. grid::load_update_XmatsHom_helpers
    // unconditionally copies TOPOLOGY_HELPERS_COUNT ints into the pointer it is handed. That count is
    // 0 for a serial chain with identical motion subspaces (Panda) — which is why passing nullptr has
    // worked so far — but it is 175 for the branched G1, where nullptr is a device-side null write.
    // Size is clamped to >= 1 because a zero-length __shared__ array is ill-formed.
    static constexpr int TOPO = grid::TOPOLOGY_HELPERS_COUNT > 0 ? grid::TOPOLOGY_HELPERS_COUNT : 1;
}

// ---------------------------------------------------------------------------------------------
// Floating base (Architecture B). See docs/open-tasks/floating-base-audit-and-design.md.
//
// The base is a RIGID TRANSFORM on top of the fixed-base FK:
//
//     x_world_i = R_b * fk_i(q_j) + p_b                                              (1)
//
// so it never enters the FK, the coordinate machinery, or the cost. It enters in exactly ONE
// place: each candidate's private copy of the targets is stored in ITS OWN base frame,
//
//     p_base,i = R_b^T (p*_world,i - p_b)          POSITION                             (2a)
//     q_base,i = q_b^-1 (x) q*_world,i             ORIENTATION -- transformed TOO, and it
//                                                  must be: leaving q* in world frame while
//                                                  the FK is base-frame would silently score
//                                                  every orientation residual against a frame
//                                                  rotated by R_b.                       (2b)
//
// WHY A BASE UPDATE NEEDS NO CHAIN FK
// -----------------------------------
// The fixed-base FK is expressed in the BASE frame and depends only on q_j. Moving the base does
// not change any joint, so s_XmatsHom / s_jointX / s_target_X remain EXACTLY valid -- there is
// nothing to recompute. Only the immutable world targets are re-expressed through (2). A base
// step therefore costs one re-transform of K targets plus one re-score, against a joint step's
// full/subtree FK. (See base_retarget_and_eval_warp.)
//
// EVERY TERM OF THE PHYSICAL ACCEPTANCE COST IS INVARIANT UNDER (2)
// ----------------------------------------------------------------
// The acceptance metric is E_phys = sum_{k active} (pn_k/eps_p)^2 + (on_k/eps_o)^2. Term by term:
//
//   pn_k = ||e_pos||   e_pos^W = p*^W - x^W = R_b (p*^B - fk) = R_b e_pos^B
//                      => ||e_pos^W|| = ||e_pos^B||           (R_b orthogonal)
//   on_k = ||e_ori||   e_ori^W = Log(R*^W R^W,T) = Log(R_b (R*^B R^B,T) R_b^T)
//                              = R_b Log(R*^B R^B,T) = R_b e_ori^B      (Log equivariance)
//                      => ||e_ori^W|| = ||e_ori^B||
//   eps_p, eps_o       constants
//   active             unchanged by a base move
//
// So E_phys computed in the base frame IS the world-frame physical cost -- the acceptance test
// compares like with like, and cost / weighted_cost_warp / all_active_converged are likewise
// unaffected. Fixed base (p_b = 0, q_b = identity) reduces (2) to a verbatim copy.
//
// Quaternions are WXYZ and unit, matching tgt_q and the rest of hjcdik.
// ---------------------------------------------------------------------------------------------

// out = v (cross) w. HJCD had no named cross product -- it is inlined in four places
// (hjcd_kernel.cu:1114, :1212, :1962, :324). New code should use this.
template<typename T>
__device__ __forceinline__
void vec3_cross(const T* __restrict__ v, const T* __restrict__ w, T* __restrict__ out) {
    out[0] = v[1]*w[2] - v[2]*w[1];
    out[1] = v[2]*w[0] - v[0]*w[2];
    out[2] = v[0]*w[1] - v[1]*w[0];
}

// out = R(q)^T v, i.e. rotate v by the INVERSE of unit quaternion q (wxyz).
// Uses the standard t = 2(qv x v); R(q)v = v + qw t + qv x t, with qv negated for R^T.
template<typename T>
__device__ __forceinline__
void quat_rotate_inv(const T* __restrict__ q, const T* __restrict__ v, T* __restrict__ out) {
    const T qv[3] = { -q[1], -q[2], -q[3] };            // conjugate: R(q)^T == R(q^-1)
    T t[3];
    vec3_cross(qv, v, t);
    t[0] += t[0]; t[1] += t[1]; t[2] += t[2];           // t = 2 (qv x v)
    T c[3];
    vec3_cross(qv, t, c);
    #pragma unroll
    for (int i = 0; i < 3; ++i) out[i] = v[i] + q[0] * t[i] + c[i];
}

// WXYZ unit quaternion -> 3x3 rotation, COLUMN-MAJOR (R[3*c + r]), matching the 4x4 convention
// used everywhere else in this file.
template<typename T>
__device__ __forceinline__
void quat_to_mat3(const T* __restrict__ q, T* __restrict__ R) {
    const T w = q[0], x = q[1], y = q[2], z = q[3];
    R[0] = (T)1 - (T)2*(y*y + z*z);  R[1] = (T)2*(x*y + w*z);         R[2] = (T)2*(x*z - w*y);
    R[3] = (T)2*(x*y - w*z);         R[4] = (T)1 - (T)2*(x*x + z*z);  R[5] = (T)2*(y*z + w*x);
    R[6] = (T)2*(x*z + w*y);         R[7] = (T)2*(y*z - w*x);         R[8] = (T)1 - (T)2*(x*x + y*y);
}

// out = a (x) b, Hamilton product of WXYZ quaternions.
template<typename T>
__device__ __forceinline__
void quat_mul_wxyz(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ out) {
    out[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
    out[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
    out[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
    out[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
}

// Pull ONE world target into the candidate's base frame, eq. (2). `base_p`/`base_q` null =>
// fixed base => verbatim copy, so the fixed-base path is bit-identical to before.
template<typename T>
__device__ __forceinline__
void world_target_to_base(const T* __restrict__ tgt_p_w, const T* __restrict__ tgt_q_w,
                          const T* __restrict__ base_p, const T* __restrict__ base_q,
                          T* __restrict__ out_p, T* __restrict__ out_q) {
    if (base_p == nullptr) {
        #pragma unroll
        for (int c = 0; c < 3; ++c) out_p[c] = tgt_p_w[c];
        #pragma unroll
        for (int c = 0; c < 4; ++c) out_q[c] = tgt_q_w[c];
        return;
    }
    const T d[3] = { tgt_p_w[0] - base_p[0], tgt_p_w[1] - base_p[1], tgt_p_w[2] - base_p[2] };
    quat_rotate_inv(base_q, d, out_p);                              // R_b^T (p* - p_b)
    const T qinv[4] = { base_q[0], -base_q[1], -base_q[2], -base_q[3] };   // unit => conj
    quat_mul_wxyz(qinv, tgt_q_w, out_q);                            // q_b^-1 (x) q*
}

// Compose every target frame from ONE full-body FK:
//
//     X_target[k] = X_world[ anchor(k) ] * TOOL[k]        (column-major 4x4)
//
// The FK (grid::ee_pose_inner_warp) already gives the world transform of every movable joint, so
// each target costs exactly one 4x4 multiply -- no per-target FK re-walk.
//
// Written for G1's register budget, which is already at the 255 cap: the output lives in SHARED
// memory (s_target_X, NT*16 scalars, warp-shared), the loop is lane-parallel over output CELLS so
// no lane ever holds a whole 4x4, and the trip count is bounded by NT at compile time. Peak
// per-lane state is one accumulator plus two pointers, independent of NT.
template<typename T>
__device__ __forceinline__
void compose_target_frames_warp(T* __restrict__ s_target_X, const T* __restrict__ s_jointX) {
    const int lane = threadIdx.x & 31;
    for (int idx = lane; idx < hjcd::NT * 16; idx += WARP_SIZE) {
        const int k = idx >> 4;              // target
        const int e = idx & 15;              // cell inside its 4x4
        const int r = e & 3, c = e >> 2;     // column-major: cell == 4*col + row
        const T* __restrict__ A = &s_jointX[16 * hjcd_gen::TARGET_ANCHOR_JID[k]];
        T acc = (T)0;
        #pragma unroll
        for (int m = 0; m < 4; ++m)
            acc += A[4 * m + r] * hjcd_gen::tool_xform<T>(16 * k + 4 * c + m);
        s_target_X[idx] = acc;
    }
    __syncwarp(FULL_WARP_MASK);
}

// Recompose ONLY the target frames named by `tmask` (bit k). Same math as
// compose_target_frames_warp, restricted -- an unaffected target's 16 cells are never written, so
// its cached state stays bitwise identical.
template<typename T>
__device__ __forceinline__
void compose_target_frames_masked_warp(T* __restrict__ s_target_X,
                                       const T* __restrict__ s_jointX,
                                       unsigned int tmask) {
    const int lane = threadIdx.x & 31;
    for (int idx = lane; idx < hjcd::NT * 16; idx += WARP_SIZE) {
        const int k = idx >> 4;
        if (!((tmask >> k) & 1u)) continue;
        const int e = idx & 15;
        const int r = e & 3, c = e >> 2;
        const T* __restrict__ A = &s_jointX[16 * hjcd_gen::TARGET_ANCHOR_JID[k]];
        T acc = (T)0;
        #pragma unroll
        for (int m = 0; m < 4; ++m)
            acc += A[4 * m + r] * hjcd_gen::tool_xform<T>(16 * k + 4 * c + m);
        s_target_X[idx] = acc;
    }
    __syncwarp(FULL_WARP_MASK);
}

// Refresh joint j's LOCAL homogeneous transform for a new angle, in place inside s_XmatsHom.
// grid::update_XmatHom_joint copies j's 16 cells then overwrites only the q-dependent ones, so it
// is safe to target the same slot it reads (the copy is a self-assign); the staging buffer keeps
// that obvious and keeps the 16 cells off one lane's registers.
template<typename T>
__device__ __forceinline__
void update_joint_local_warp(T* __restrict__ s_XmatsHom, T* __restrict__ s_loc16, int j, T theta) {
    const int lane = threadIdx.x & 31;
    if (lane == 0) grid::update_XmatHom_joint<T>(s_loc16, s_XmatsHom, j, theta);
    __syncwarp(FULL_WARP_MASK);
    if (lane < 16) s_XmatsHom[16 * j + lane] = s_loc16[lane];
    __syncwarp(FULL_WARP_MASK);
}

// Tree-correct incremental FK: recompute the world transforms of j's SUBTREE only.
//
//     for u ascending: if u in subtree(j):  X_world[u] = X_world[parent(u)] * X_local[u]
//
// The ascending scan IS the topological order: GRiD's DFS pre-order numbering guarantees
// parent(u) < u (asserted at codegen), so parent(u)'s world transform is already final by the time
// u is visited. The mask includes j itself, whose LOCAL transform the caller just updated.
//
// This replaces the old ee_fk_suffix_thread, whose `for kk = jovr..FLANGE_JID` walk assumed a serial
// chain (parent(u) == u-1) and is simply wrong on a branched robot.
template<typename T>
__device__ __forceinline__
void subtree_fk_body(T* __restrict__ s_jointX, const T* __restrict__ s_XmatsHom, int u, int lane) {
    const int par = hjcd_gen::JOINT_PARENT_JID[u];
    if (lane < 16) {
        const int r = lane & 3, c = lane >> 2;          // column-major cell = 4*c + r == lane
        T acc;
        if (par < 0) {
            acc = s_XmatsHom[16 * u + lane];            // attached to the world root
        } else {
            acc = (T)0;
            #pragma unroll
            for (int m = 0; m < 4; ++m)
                acc += s_jointX[16 * par + 4 * m + r] * s_XmatsHom[16 * u + 4 * c + m];
        }
        s_jointX[16 * u + lane] = acc;                  // par != u, so no read/write overlap
    }
    __syncwarp(FULL_WARP_MASK);                         // u's children may be next
}

// SHIPPED: the plain ascending mask scan. Visits all N joints, doing work only for those in the
// subtree. The skipped joints cost one shift-and-test each, which measured NEGLIGIBLE against the
// __syncwarp-bound real work -- see subtree_fk_ffs_warp below.
template<typename T>
__device__ __forceinline__
void subtree_fk_warp(T* __restrict__ s_jointX, const T* __restrict__ s_XmatsHom, unsigned int desc) {
    const int lane = threadIdx.x & 31;
    for (int u = 0; u < hjcd::N; ++u) {
        if (!((desc >> u) & 1u)) continue;
        subtree_fk_body<T>(s_jointX, s_XmatsHom, u, lane);
    }
}

// The "compact list" alternative, without the list: iterating the mask's set bits with __ffs visits
// only |subtree| joints (still ascending, so still parent-before-child) and needs no generated
// descendant array at all. MEASURED on G1 K=4 against the scan above: 0.94x - 1.005x, i.e. no
// improvement and sometimes slower. Kept solely as the benchmark comparator (fk_bench mode 2); the
// solve path uses the simple scan. A generated DESCENDANT_OFFSET/INDICES list can only do worse than
// this -- it pays the same loop with extra memory traffic -- so it is not implemented.
template<typename T>
__device__ __forceinline__
void subtree_fk_ffs_warp(T* __restrict__ s_jointX, const T* __restrict__ s_XmatsHom,
                         unsigned int desc) {
    const int lane = threadIdx.x & 31;
    unsigned int rem = desc;
    while (rem) {
        const int u = __ffs(rem) - 1;
        rem &= rem - 1u;
        subtree_fk_body<T>(s_jointX, s_XmatsHom, u, lane);
    }
}

// Joint j's motion axis in WORLD frame, from its cumulative world transform s_jointX[16*j].
//
// GRiD keeps the URDF <axis>; it does not rotate joints onto local z. So the world axis is the
// JOINT_AXIS_COL[j]-th column of the joint's world rotation, signed by JOINT_AXIS_SIGN[j] — see
// scripts/codegen/generate_grid.py:inject_joint_axis_metadata for why that column is q-invariant.
// Panda is all-z (col 2, sign +1), so this is bit-identical to the previous hardcoded Ci[8..10].
template<typename T>
__device__ __forceinline__
void joint_world_axis(const T* __restrict__ Cj, int j, T* __restrict__ a3) {
    if constexpr (grid::ALL_AXIS_Z) {
        // Every joint rotates about +z (Panda + its DoF variants, the Fetch arm): the column offset
        // is a compile-time constant, exactly as before this metadata existed. Keeping this path
        // separate is not cosmetic — the general indexed read below costs +26 registers on
        // lm_tuner<double>, and an all-z robot gains nothing from paying it.
        a3[0] = Cj[8]; a3[1] = Cj[9]; a3[2] = Cj[10];
    } else {
        const int c = 4 * grid::JOINT_AXIS_COL[j];
        const T s = (T)grid::JOINT_AXIS_SIGN[j];
        a3[0] = s * Cj[c + 0];
        a3[1] = s * Cj[c + 1];
        a3[2] = s * Cj[c + 2];
    }
}

// ---------------------------------------------------------------------------
// Forward-kinematics-to-grasptarget helpers.
//
// GRiD's stock FK (grid::ee_pose_inner_warp / _thread) fills the cumulative world
// transforms s_jointX[16*j] for the actuated joints j=0..N-1 only; it does not apply
// the fixed grasptarget tool offset. HJCD's solver reads the grasptarget world pose at
// slot EE_IDX (== grid::NUM_JOINTS), so we append it here:
//     s_jointX[16*ee_slot] = s_jointX[16*FLANGE_JID] * s_XmatsHom_fixed[16*GRASP_FIXED_IDX]
// (T_lastjoint * X_fixed) — exactly the multiply the old bespoke X_warp/X_single_thread did.
// ---------------------------------------------------------------------------

// Warp-cooperative: must be entered by all 32 lanes of a single warp.
template<typename T>
__device__ __forceinline__
void ee_fk_warp(T* s_jointX, T* s_XmatsHom, T* s_q, int ee_slot) {
    grid::ee_pose_inner_warp<T>(s_jointX, s_XmatsHom, s_q, hjcd::FLANGE_JID);
    // Grasptarget offset T_lastjoint * X_fixed (column-major 4x4) via the warp GEMM:
    // all 32 lanes cooperate (flat per-element parallelism), C overwritten (beta=0).
    // C (ee_slot) is disjoint from A (FLANGE_JID) and B (s_XmatsHom), as gemm requires.
    glass::warp::gemm<T, 4, 4, 4>(
        static_cast<T>(1),
        &s_jointX[16 * hjcd::FLANGE_JID],
        &s_XmatsHom[16 * hjcd::GRASP_FIXED_IDX],
        &s_jointX[16 * ee_slot]);
    __syncwarp(FULL_WARP_MASK);
}

// Single-thread: must be entered by exactly one thread. s_fixed_src holds the grasptarget
// fixed frame at index GRASP_FIXED_IDX (the full shared s_XmatsHom; pass it explicitly so the
// greedy candidate sites — which work on a truncated local copy — can source it from shared).
template<typename T>
__device__ __forceinline__
void ee_fk_thread(T* s_jointX, T* s_XmatsHom, T* s_q, int ee_slot, const T* s_fixed_src) {
    grid::ee_pose_inner_thread<T>(s_jointX, s_XmatsHom, s_q, hjcd::FLANGE_JID);
    mat4_mul(&s_jointX[16 * hjcd::FLANGE_JID],
             &s_fixed_src[16 * hjcd::GRASP_FIXED_IDX],
             &s_jointX[16 * ee_slot]);
}

// Single-joint SUFFIX FK for coordinate-descent candidates. Computes the EE world
// transform (grasptarget applied, written to out_ee16[0..15]) of a config that equals an
// ANCHOR config EXCEPT joint `jovr` is set to angle `aovr`. Reuses the anchor's cumulative
// world chain (l_anchorX) and per-joint locals (l_anchorLoc) for joints before/after jovr
// and recomputes only joints jovr..FLANGE_JID via grid::update_XmatHom_joint. O(1) scratch
// (a running 4x4 + one overridden local) — independent of DoF, so it scales to large robots
// where a full per-candidate chain copy would not fit shared memory.
//
// BIT-IDENTICAL to a full thread FK of the candidate when the anchor was itself built by a
// thread FK (same locals, same compose order; only joint jovr's local differs).
// Assumes a SERIAL chain (parent(j) == j-1) — true for all current robots (Panda + DoF
// variants + the Fetch arm). Tree/branched robots (humanoids) need the parent table and a
// subtree walk (follow-up); the grid::update_XmatHom_joint primitive itself is general.
template<typename T>
__device__ __forceinline__
void ee_fk_suffix_thread(T* out_ee16, const T* l_anchorX, const T* l_anchorLoc,
                         const T* s_XmatsHom_full, int jovr, T aovr) {
    T W[16];
    if (jovr <= 0) {
        #pragma unroll
        for (int m = 0; m < 16; ++m) W[m] = (T)0;
        W[0] = W[5] = W[10] = W[15] = (T)1;                 // identity: no parent
    } else {
        #pragma unroll
        for (int m = 0; m < 16; ++m) W[m] = l_anchorX[16 * (jovr - 1) + m];
    }
    T locbuf[16], Wt[16];
    for (int kk = (jovr > 0 ? jovr : 0); kk <= hjcd::FLANGE_JID; ++kk) {
        const T* loc;
        if (kk == jovr) { grid::update_XmatHom_joint<T>(locbuf, l_anchorLoc, kk, aovr); loc = locbuf; }
        else            { loc = &l_anchorLoc[16 * kk]; }
        mat4_mul(W, loc, Wt);
        #pragma unroll
        for (int m = 0; m < 16; ++m) W[m] = Wt[m];
    }
    mat4_mul(W, &s_XmatsHom_full[16 * hjcd::GRASP_FIXED_IDX], out_ee16);
}

struct RefineSchedule {
    int    top_k;
    int    repeats;
    double sigma_frac;
    bool   keep_one;
};

inline RefineSchedule schedule_for_B(int B) {
    RefineSchedule s;
    s.keep_one   = true;
    s.sigma_frac = 0.1;
    s.repeats    = 16;

    if (B <= 16) {
        s.top_k     = B;
        s.repeats   = 16;
        s.sigma_frac= 0.25;
    } else {
        s.top_k = 16 + (int)((B - 1000)/1000 * 8);
    }

    return s;
}

template<typename T>
struct HJCDSettings {
    // Coarse phase settings
    static constexpr T epsilon = static_cast<T>(20e-3);   // 20 mm
    static constexpr T nu = static_cast<T>(90 * PI / 180.0);
    static constexpr int k_max  = 20;

    // Refine phase settings
    static constexpr T lambda_init = static_cast<T>(5e-3);
};
