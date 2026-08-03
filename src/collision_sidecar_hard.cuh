// Hard-mode (Checkpoint 3D/3E) device layer for the G1 self-collision sidecar.
//
// WHY THIS IS A SEPARATE HEADER: `collision_sidecar.cuh` is the VALIDATED Checkpoint-2 geometry
// (FK, primitive/SDF/GJK narrow phases). Nothing in it changes here. This header adds only the
// state management hard mode needs -- a per-seed persistent workspace in GLOBAL memory, a
// descendant-only FK update, and a warp-cooperative incremental trial check -- built strictly on
// top of those validated primitives, so an incremental verdict is the same arithmetic as a full
// verdict, not a reimplementation of it.
//
// RESOURCE CONTRACT (spec section 2). The 162/177-register full checker stays a SEPARATE batched
// kernel (`full_check_kernel`, unchanged). The coarse-search hot path reaches collision ONLY
// through `sidecar_hard_trial` / `sidecar_hard_restore`, both `__noinline__`, so ptxas allocates
// their (large, GJK/SDF-dominated) frames as ABI callees instead of folding them into
// coarse_search_mt_kernel's register frame. Per-seed state lives in global memory, never in
// registers or local memory: the complete collision state is 40 link transforms in f32 AND f64
// (7680 B), which no register frame can hold.
#pragma once
#include "collision_sidecar.cuh"

namespace g1sc {

// ================= per-seed persistent workspace (GLOBAL memory) =================
// Allocated ONLY in hard mode. `off` and `final` never touch it (the pointers stay null and the
// host never calls ensure()).
//
// Tf/Td are the COMMITTED link transforms and double as the trial scratch: a trial overwrites the
// moving joint's descendant links in place, and a rejected trial restores them by recomputing the
// same descendant FK at the committed joint value. Recomputation is BYTE-IDENTICAL (identical
// inputs through identical code), which is what lets us skip a 7680 B/seed shadow copy while still
// guaranteeing "a rejected trial mutates no committed state".
struct HardWorkspace {
    float*         Tf;       // [B * N_LINKS * 16] committed f32 link transforms (column-major)
    double*        Td;       // [B * N_LINKS * 16] committed f64 link transforms (GJK)
    float*         qc;       // [B * N_JOINTS]     committed q -- mirrors the solver's q exactly
    float*         qfree;    // [B * N_JOINTS]     last collision-free coarse q
    unsigned char* flags;    // [B]  bit0 = collision-state valid, bit1 = has_collision_free_coarse
    int*           ctr;      // [B * HARD_CTR_STRIDE] diagnostics, or NULL in the fast path
};

static constexpr unsigned char HARD_FLAG_STATE_VALID = 0x1u;
static constexpr unsigned char HARD_FLAG_HAS_FREE_Q  = 0x2u;

// Ranked-proposal cap. `collision_top_k` is validated against this host-side.
static constexpr int HARD_MAX_K = 8;

// Diagnostic counter slots (per seed). Off the fast path: written only when ws.ctr != NULL.
enum : int {
    HARD_CTR_PROPOSALS = 0,   // trial checks actually performed
    HARD_CTR_ALLK,            // iterations where every ranked candidate collided
    HARD_CTR_REJECTED,        // trials rejected by the collision gate
    HARD_CTR_GJK_PAIRS,       // GJK link-pairs evaluated (post broad-phase entry)
    HARD_CTR_GJK_ITERS,       // cumulative GJK iterations
    HARD_CTR_NONGJK_PAIRS,    // primitive + cluster-SDF link-pairs evaluated
    HARD_CTR_SDF_EVALS,       // trilinear SDF evaluations
    HARD_CTR_PERT_SKIPPED,    // Stage 3F placeholder: perturbations skipped in hard mode
    HARD_CTR_NO_GJK_TRIALS,   // trials that returned a verdict without entering GJK
    HARD_CTR_ORACLE_CHECKS,   // debug oracle: incremental-vs-full cross-checks performed
    HARD_CTR_ORACLE_MISMATCH, // debug oracle: cross-checks that DISAGREED (must stay 0)
    HARD_CTR_ACCEPT_RANK0,    // [HARD_MAX_K] accepted-at-rank histogram
    HARD_CTR_REJ_BY_JOINT = HARD_CTR_ACCEPT_RANK0 + HARD_MAX_K,   // [N_JOINTS]
    HARD_CTR_STRIDE       = HARD_CTR_REJ_BY_JOINT + N_JOINTS
};

__device__ __forceinline__ void hard_ctr_add(const HardWorkspace& ws, int b, int slot, int v) {
    if (ws.ctr) ws.ctr[(size_t)b * HARD_CTR_STRIDE + slot] += v;
}

// ================= descendant-only FK =================
// Recompute the link transforms of every DESCENDANT of joint `j` (JOINT_DESC CSR) from `q`.
//
// BYTE-IDENTITY WITH THE FULL FK IS THE WHOLE POINT. `sidecar_fk` walks links 1..N_LINKS-1 in BFS
// order computing T[L] = (T[parent] * ORIGIN[L]) * R(axis, q). JOINT_DESC[j] is that same BFS order
// restricted to j's subtree, and every descendant's parent is either committed (unchanged by q[j])
// or an earlier entry in the same list -- so the identical operand order runs on identical inputs
// and produces identical bits. That is what makes "incremental verdict == full verdict" exact
// rather than approximate, and it is asserted by the debug oracle.
//
// Single-lane (lane 0) on purpose: the chain is a serial dependency, and this matches
// full_check_kernel, which also computes FK on lane 0. Caller must __syncwarp() afterwards.
//
// The trial value is substituted POSITIONALLY (`qi == j ? vnew : q[qi]`) rather than through a
// mutated copy of q, so no scratch array -- shared or local -- is needed for a trial. Passing
// vnew = q[j] makes it the exact restore of the committed state.
__device__ inline void sidecar_fk_desc(const float* q, int j, float vnew, float* T_out) {
    const int o0 = JOINT_DESC_OFF[j], o1 = JOINT_DESC_OFF[j + 1];
    for (int k = o0; k < o1; ++k) {
        const int L = JOINT_DESC[k];
        const int par = LINK_PARENT[L];
        float To[16];
        mat4_mul(&T_out[par * 16], &LINK_T_ORIGIN[L * 16], To);
        const int qi = LINK_QINDEX[L];
        if (qi >= 0) {
            float Rj[16];
            axis_angle(&LINK_AXIS[L * 3], (qi == j) ? vnew : q[qi], Rj);
            mat4_mul(To, Rj, &T_out[L * 16]);
        } else {
            mat4_copy(To, &T_out[L * 16]);
        }
    }
}
__device__ inline void sidecar_fk_desc_d(const float* q, int j, float vnew, double* T_out) {
    const int o0 = JOINT_DESC_OFF[j], o1 = JOINT_DESC_OFF[j + 1];
    for (int k = o0; k < o1; ++k) {
        const int L = JOINT_DESC[k];
        const int par = LINK_PARENT[L];
        double To[16], Orig[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) Orig[i] = (double)LINK_T_ORIGIN[L * 16 + i];
        dmat4_mul(&T_out[par * 16], Orig, To);
        const int qi = LINK_QINDEX[L];
        if (qi >= 0) {
            double Rj[16];
            daxis_angle(&LINK_AXIS[L * 3], (double)((qi == j) ? vnew : q[qi]), Rj);
            dmat4_mul(To, Rj, &T_out[L * 16]);
        } else {
            for (int i = 0; i < 16; ++i) T_out[L * 16 + i] = To[i];
        }
    }
}

// ================= affected-pair incremental verdict =================
// Colliding verdict over ONLY joint `j`'s affected pairs (JOINT_AFFPAIR CSR), given transforms that
// already reflect the trial configuration.
//
// WHY ONLY THE AFFECTED PAIRS ARE ENOUGH: hard mode's committed state is collision-free by
// construction (that is the invariant Stage 3D establishes and Stage 3E preserves), so every pair
// NOT affected by q[j] is known-free without recomputation. There is no 351-byte committed verdict
// vector to store or overlay -- the committed verdict is identically zero. A trial therefore
// collides iff one of j's affected pairs collides.
//
// Cheap-first ordering: primitive and cluster-SDF pairs are lane-parallel and settle in a warp
// ballot; the f64 GJK pairs run full-warp cooperatively and only if nothing cheaper already hit.
// Returns 1 = colliding. *first_pair (optional) gets a colliding pair index for diagnostics.
__device__ inline int sidecar_affected_colliding(const HardWorkspace& ws, int b, int j,
                                                 const float* T, const double* Td,
                                                 float margin, int lane, int* first_pair) {
    const int o0 = JOINT_AFFPAIR_OFF[j], o1 = JOINT_AFFPAIR_OFF[j + 1];

    // ---- cheap phases: lane-parallel over the CSR, then a warp ballot ----
    int hit = 0, hit_g = -1, n_cheap = 0;
    for (int k = o0 + lane; k < o1; k += 32) {
        const int g = JOINT_AFFPAIR[k];
        if (PAIR_TYPE[g] == PAIR_CONVEX_GJK) continue;
        ++n_cheap;
        if (linkpair_colliding_nongjk(g, T, margin)) { hit = 1; if (hit_g < 0) hit_g = g; }
    }
    int reached_gjk = 0;
    if (ws.ctr) {
        int tot = n_cheap;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) tot += __shfl_down_sync(0xffffffffu, tot, off);
        if (lane == 0) hard_ctr_add(ws, b, HARD_CTR_NONGJK_PAIRS, tot);
    }
    // Warp-uniform verdict: every lane must agree BEFORE the warp-cooperative GJK loop below,
    // otherwise a divergent early-exit would deadlock the __shfl_sync inside GJK.
    unsigned int ballot = __ballot_sync(0xffffffffu, hit);
    if (ballot) {
        if (first_pair) {
            const int src = __ffs((int)ballot) - 1;
            *first_pair = __shfl_sync(0xffffffffu, hit_g, src);
        }
        if (ws.ctr && lane == 0) hard_ctr_add(ws, b, HARD_CTR_NO_GJK_TRIALS, 1);
        return 1;                                   // settled by the cheap phases: no GJK at all
    }

    // ---- exact f64 GJK on the affected convex pairs (full warp, uniform control flow) ----
    for (int k = o0; k < o1; ++k) {
        const int g = JOINT_AFFPAIR[k];
        if (PAIR_TYPE[g] != PAIR_CONVEX_GJK) continue;
        int iters = 0;
        const double gap = gjk_linkpair_gap_d(PAIR_LINK_A[g], PAIR_LINK_B[g], Td,
                                              (double)margin, lane, &iters);
        if (ws.ctr && lane == 0) {
            hard_ctr_add(ws, b, HARD_CTR_GJK_PAIRS, 1);
            hard_ctr_add(ws, b, HARD_CTR_GJK_ITERS, iters);
        }
        reached_gjk = 1;
        if (gap < (double)margin) {                 // uniform: gap is warp-uniform by construction
            if (first_pair) *first_pair = g;
            return 1;
        }
    }
    if (ws.ctr && lane == 0 && !reached_gjk) hard_ctr_add(ws, b, HARD_CTR_NO_GJK_TRIALS, 1);
    return 0;
}

// ================= debug oracle (spec section 11) =================
// Cross-check the INCREMENTAL verdict against a FULL sweep of all N_CHECKED_PAIRS on the very same
// trial transforms. What this proves is the thing that could actually be wrong: that joint j's
// affected-pair CSR is SUFFICIENT -- that no pair outside it can change verdict when only q[j]
// moves. (The other half of the argument, that the descendant-only FK reproduces a full FK bitwise,
// is checked host-side against a fresh sidecar_fk; see test_incremental_fk_matches_full_fk.)
//
// Validation only: quadratic in checked pairs versus the affected subset, and never enabled in a
// performance run. Returns 1 on MISMATCH.
__device__ __noinline__ int sidecar_hard_oracle(HardWorkspace ws, int b, float margin, int lane,
                                                int incr_verdict) {
    const float*  T  = &ws.Tf[(size_t)b * N_LINKS * 16];
    const double* Td = &ws.Td[(size_t)b * N_LINKS * 16];
    int hit = 0;
    for (int g = lane; g < N_CHECKED_PAIRS; g += 32) {
        if (PAIR_TYPE[g] == PAIR_CONVEX_GJK) continue;
        if (linkpair_colliding_nongjk(g, T, margin)) hit = 1;
    }
    hit = __ballot_sync(0xffffffffu, hit) ? 1 : 0;
    if (!hit) {
        for (int g = 0; g < N_CHECKED_PAIRS; ++g) {
            if (PAIR_TYPE[g] != PAIR_CONVEX_GJK) continue;
            if (linkpair_colliding_gjk(g, Td, margin, lane)) { hit = 1; break; }
        }
    }
    return (hit != incr_verdict) ? 1 : 0;
}

// ================= the two hot-path entry points =================
// __noinline__ ON PURPOSE (spec section 2): these pull in the capsule-SDF branch-and-bound stack
// (2 KB of local memory) and the f64 GJK simplex. As ABI callees their frames are allocated at
// call time instead of being folded into coarse_search_mt_kernel's register/stack frame.

// Trial: set q[j] = vnew, refresh j's descendant links IN PLACE in the persistent workspace, and
// return the collision verdict over j's affected pairs.
//
// The workspace is left holding the TRIAL transforms whichever way the verdict goes. The caller
// MUST then call exactly one of:
//    free      -> sidecar_hard_commit(ws, b, j, vnew, lane)      (publishes q[j], keeps transforms)
//    colliding -> sidecar_hard_restore(ws, b, j, lane)           (recomputes at the committed q[j])
// This two-call shape is what keeps "commit" and "discard" symmetric and auditable; a partially
// updated committed state is not reachable, because ws.qc[j] is written only by the commit call.
__device__ __noinline__ int sidecar_hard_trial(HardWorkspace ws, int b, int j, float vnew,
                                               float margin, int lane, int* first_pair) {
    float*  T  = &ws.Tf[(size_t)b * N_LINKS * 16];
    double* Td = &ws.Td[(size_t)b * N_LINKS * 16];
    const float* q = &ws.qc[(size_t)b * N_JOINTS];

    if (lane == 0) {                       // serial chain, matching full_check_kernel's FK on lane 0
        sidecar_fk_desc(q, j, vnew, T);
        sidecar_fk_desc_d(q, j, vnew, Td);
    }
    __syncwarp();

    if (ws.ctr && lane == 0) hard_ctr_add(ws, b, HARD_CTR_PROPOSALS, 1);
    const int c = sidecar_affected_colliding(ws, b, j, T, Td, margin, lane, first_pair);
    if (ws.ctr && lane == 0 && c) {
        hard_ctr_add(ws, b, HARD_CTR_REJECTED, 1);
        hard_ctr_add(ws, b, HARD_CTR_REJ_BY_JOINT + j, 1);
    }
    return c;
}

// Publish a trial as committed. Only ws.qc changes here: the transforms already hold the trial.
__device__ __forceinline__ void sidecar_hard_commit(HardWorkspace ws, int b, int j, float vnew,
                                                    int lane) {
    if (lane == 0) {
        ws.qc[(size_t)b * N_JOINTS + j] = vnew;
        ws.flags[b] |= HARD_FLAG_STATE_VALID;
    }
    __syncwarp();
}

// Discard a trial: recompute j's descendants at the COMMITTED q[j]. Byte-identical to the state
// before the trial (same inputs, same code), so the committed collision state is bitwise restored
// without a shadow copy.
__device__ __noinline__ void sidecar_hard_restore(HardWorkspace ws, int b, int j, int lane) {
    float*  T  = &ws.Tf[(size_t)b * N_LINKS * 16];
    double* Td = &ws.Td[(size_t)b * N_LINKS * 16];
    const float* q = &ws.qc[(size_t)b * N_JOINTS];
    if (lane == 0) {
        const float qj = q[j];
        sidecar_fk_desc(q, j, qj, T);
        sidecar_fk_desc_d(q, j, qj, Td);
    }
    __syncwarp();
}

// Record the current committed q as the last collision-free coarse state (Stage 3E / 3D).
__device__ __forceinline__ void sidecar_hard_mark_free(HardWorkspace ws, int b, int lane) {
    const float* q = &ws.qc[(size_t)b * N_JOINTS];
    float* qf = &ws.qfree[(size_t)b * N_JOINTS];
    for (int i = lane; i < N_JOINTS; i += 32) qf[i] = q[i];
    if (lane == 0) ws.flags[b] |= HARD_FLAG_HAS_FREE_Q;
    __syncwarp();
}

}  // namespace g1sc
