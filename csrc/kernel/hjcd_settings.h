#pragma once

// Generated GRiD kinematics. Since the header is generated with vendor_glass=False
// (scripts/codegen/generate_grid.py) it includes the top-level external/GLASS itself and
// aliases `grid::glass` to `::glass`, so exactly ONE GLASS is compiled per translation unit.
#include "grid.cuh"
#include "glass.cuh"

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef FULL_WARP_MASK
#define FULL_WARP_MASK 0xFFFFFFFFu
#endif

#ifndef PI
#define PI 3.14159265358979323846
#endif

namespace hjcd {
    static constexpr int N = grid::NUM_JOINTS;            // actuated joints (7 for Panda)
    static_assert(N > 0 && N <= 32, "HJCD-IK requires 1 to 32 actuated joints");
    static constexpr int FLANGE_JID = N - 1;              // cumulative world transform of the last joint
    // Fixed EE-offset frame inside s_XmatsHom: the frame index where GRiD places the named EE target
    // (its end_effector_pose_inner_<target> epilogue chains s_Xhom[16*EE_FIXED_FRAME_IDX] onto joint
    // FLANGE_JID). This index is ROBOT-SPECIFIC and shifts with DoF, so it is resolved at codegen time
    // and injected into grid.cuh by scripts/codegen/generate_grid.py (Panda grasptarget=10, etc.) — never hardcode.
    static constexpr int GRASP_FIXED_IDX = grid::EE_FIXED_FRAME_IDX;
}

// 4x4 column-major homogeneous product C = A * B on one thread (C must not alias A or B).
template<typename T>
__device__ __forceinline__
void mat4_mul(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C) {
    glass::thread::gemm<T, 4, 4, 4>(static_cast<T>(1), A, B, C);
}

// ---------------------------------------------------------------------------
// Forward-kinematics-to-grasptarget helpers.
//
// GRiD's stock FK (grid::ee_pose_inner_warp / _thread) fills the cumulative world
// transforms s_jointX[16*j] for the actuated joints j=0..N-1 only; it does not apply
// the fixed grasptarget tool offset. HJCD's solver reads the grasptarget world pose at
// slot EE_IDX (== grid::NUM_JOINTS), so we append it here:
//     s_jointX[16*ee_slot] = s_jointX[16*FLANGE_JID] * s_XmatsHom_fixed[16*GRASP_FIXED_IDX]
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
// Assumes a SERIAL chain (parent(j) == j-1) — true for every robot codegen accepts
// (generate_grid.py rejects branched models). Tree/branched robots need the parent table and
// a subtree walk; the grid::update_XmatHom_joint primitive itself is general.
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

// Host-side refine schedule: how many coarse candidates are polished, and how many perturbed
// copies of each are refined (the first copy of each group is kept unperturbed).
struct RefineSchedule {
    int    top_k;
    int    repeats;
    double sigma_frac;
};

inline RefineSchedule schedule_for_B(int B) {
    RefineSchedule s;
    s.repeats = 16;
    if (B <= 16) {
        s.top_k      = B;
        s.sigma_frac = 0.25;
    } else {
        s.top_k      = 16 + (int)((B - 1000)/1000 * 8);
        s.sigma_frac = 0.1;
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
    static constexpr int lm_max_iters = 40;
    // Convergence / early-stop tolerance (pos in m, ori in rad). 1e-8 m is far below the fp32
    // representable floor at ~0.5 m coords, so fp32 refinement cannot early-stop at this default;
    // HJCD_LM_EPS_POS / HJCD_LM_EPS_ORI override it for precision-appropriate sweeps.
    static constexpr T lm_eps_pos = static_cast<T>(1e-8);
    static constexpr T lm_eps_ori = static_cast<T>(1e-8);
};
