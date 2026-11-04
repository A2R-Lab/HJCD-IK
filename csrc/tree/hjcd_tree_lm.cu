// Fixed-base multi-target evaluation + Levenberg-Marquardt for TREE models (humanoid-v1 M3).
//
// The LM is the humanoid branch's lm_multi_target_kernel (bbc53a9) with the floating-base step,
// the cross-candidate early stop (g_stop) and the trace removed, on the M2b tree FK. Unchanged:
// frozen row preconditioner, weighted cost as acceptance objective, null-column pin, lambda*diag(A)
// Levenberg damping via glass::warp::posv<REG_DIAG>, worst-error trust radius + per-joint clip,
// 4-step backtracking with joint-limit clamping, gain-ratio lambda policy, E_phys best tracking,
// stall kick, all-active convergence. One warp per configuration, lane j = joint j.
#include <stdexcept>
#include <string>
#include <chrono>
#include <cmath>
#include "tree/hjcd_tree_math.cuh"
#include "tree/hjcd_tree_lm.h"
#include "kernel/cuda_memory.h"
#include "grid_collision_geometry.cuh"   // GRiD 0.6.0 environment primitives + DeviceEnvironment (M6a)

static_assert(hjcd_model::NUM_JOINTS == grid::NUM_JOINTS, "joint model / GRiD header mismatch");
static_assert(hjcd_targets::NUM_JOINTS == grid::NUM_JOINTS, "target metadata / GRiD header mismatch");

namespace hjcd_tree {
namespace {

template<typename T>
struct WarpState {
    Problem<T> P;
    T x[N], x_old[N], best_x[N];
    T local[16 * N], world[16 * N], targetX[16 * K];
    T e_pos[3 * K], e_ori[3 * K], pn[K], on[K], scale[6 * K];
    T A[N * N], b[N], g[N], dq[N], diagA[N], colnorm[N];
    T cost, trial_cost, best_cost, prev_cost, best_ephys, ephys, prev_ephys, max_pn, max_on;
    int s_break, stall, accepted, s_fail, kicked, take_best, stag_n;
    // floating base (M4b): world targets are kept so the base-frame targets in P can be re-derived
    T tgt_wp[3 * K], tgt_wq[4 * K];
    T base_p[3], base_q[4], best_base_p[3], best_base_q[4], bak_p[3], bak_q[4], Hb[36], bb[6];
    int has_base, base_fail, base_att, base_acc, base_nf, base_rej, base_take;
};

template<typename T>
__host__ __device__ constexpr size_t prologue_bytes() {
    return sizeof(T) * (size_t(grid::XHOM_T_COUNT) + 3 * N) + sizeof(int) * size_t(grid::TOPOLOGY_HELPERS_COUNT);
}
template<typename T>
__host__ __device__ constexpr size_t shared_bytes(int warps) {
    return ((prologue_bytes<T>() + 15) / 16) * 16 + size_t(warps) * ((sizeof(WarpState<T>) + 15) / 16) * 16;
}

template<typename T>
__device__ __forceinline__ void block_prologue(unsigned char* smem, const grid::robotModel<T>* model, T*& s_xhom) {
    s_xhom = reinterpret_cast<T*>(smem);
    T* s_q0 = s_xhom + grid::XHOM_T_COUNT;
    T* s_tmp = s_q0 + N;
    int* s_topo = reinterpret_cast<int*>(s_tmp + 2 * N);
    for (int i = threadIdx.x; i < N; i += blockDim.x) s_q0[i] = T(0);
    __syncthreads();
    grid::load_update_XmatsHom_helpers<T>(s_xhom, s_topo, s_q0, model, s_tmp);
    __syncthreads();
}

template<typename T>
__device__ __forceinline__ WarpState<T>* warp_state(unsigned char* smem) {
    const int warp = threadIdx.x >> 5;
    return reinterpret_cast<WarpState<T>*>(smem + ((prologue_bytes<T>() + 15) / 16) * 16
                                           + size_t(warp) * ((sizeof(WarpState<T>) + 15) / 16) * 16);
}

template<typename T>
__device__ __forceinline__ void load_problem(Problem<T>& P, const ProblemIn<T>& in, const T* __restrict__ d_tool, int c) {
    const int lane = threadIdx.x & 31;
    const int b = c / in.seeds_per_problem;                  // problem index; S = 1 => b = c (M3 path)
    d_tool += size_t(in.tool_per_problem ? b : 0) * K * 16;
    if (lane < K) {
        const size_t bk = size_t(b) * K + lane;
        for (int c = 0; c < 3; ++c) P.tgt_p[3*lane + c] = in.target[bk * 7 + c];
        for (int c = 0; c < 4; ++c) P.tgt_q[4*lane + c] = in.target[bk * 7 + 3 + c];
        P.wp[lane] = in.w_pos[bk]; P.wo[lane] = in.w_ori[bk]; P.mode[lane] = in.mode[bk];
        for (int c = 0; c < 3; ++c) P.axis[3*lane + c] = in.axis[bk * 3 + c];
        for (int c = 0; c < 5; ++c) P.twist[5*lane + c] = in.twist[bk * 5 + c];
        for (int c = 0; c < 16; ++c) P.tool[16*lane + c] = d_tool[16*lane + c];
    }
    if (lane == 0) P.active = in.active[b];
    __syncwarp(FULL_MASK);
}

// Load candidate c's base (if any): keep the problem's WORLD targets, re-express them in this base's frame.
// No base => nothing changes (fixed-base targets are used verbatim, the M4a path).
template<typename T>
__device__ __forceinline__ void load_base(WarpState<T>* st, const ProblemIn<T>& in, int c) {
    const int lane = threadIdx.x & 31;
    if (lane == 0) { st->has_base = in.base != nullptr; st->base_att = st->base_acc = st->base_nf = st->base_rej = 0; st->base_fail = 0; }
    __syncwarp(FULL_MASK);
    if (!st->has_base) return;
    if (lane == 0) {
        for (int i = 0; i < 3; ++i) st->base_p[i] = st->best_base_p[i] = in.base[size_t(c) * 7 + i];
        for (int i = 0; i < 4; ++i) st->base_q[i] = st->best_base_q[i] = in.base[size_t(c) * 7 + 3 + i];
    }
    if (lane < K) {
        for (int i = 0; i < 3; ++i) st->tgt_wp[3*lane + i] = st->P.tgt_p[3*lane + i];
        for (int i = 0; i < 4; ++i) st->tgt_wq[4*lane + i] = st->P.tgt_q[4*lane + i];
    }
    __syncwarp(FULL_MASK);
    if (lane < K) world_target_to_base<T>(&st->tgt_wp[3*lane], &st->tgt_wq[4*lane], st->base_p, st->base_q,
                                          &st->P.tgt_p[3*lane], &st->P.tgt_q[4*lane]);
    __syncwarp(FULL_MASK);
}

template<typename T>
__device__ __forceinline__ void retarget_and_eval(WarpState<T>* st) {   // bbc53a9 base_retarget_and_eval_warp
    const int lane = threadIdx.x & 31;
    if (lane < K) world_target_to_base<T>(&st->tgt_wp[3*lane], &st->tgt_wq[4*lane], st->base_p, st->base_q,
                                          &st->P.tgt_p[3*lane], &st->P.tgt_q[4*lane]);
    __syncwarp(FULL_MASK);
    eval_targets_warp<T>(st->P, st->targetX, st->e_pos, st->e_ori, st->pn, st->on);
}

// One base step with the joints held (bbc53a9 base_update_warp). H, b from the ACTIVE targets with w_p > 0:
//   arm_k = R_b fk_k (base origin -> contact, world),  r_k = R_b e_p,k (world position residual),
//   J_k = [I3, -[arm_k]x],  H = sum w J^T J + damping * diag(s_p^-2 I3, s_R^-2 I3),  b = sum w J^T r,
//   solve with posv<6,REGULARIZE=false,CHECK>; clip |dp| and |dphi| separately; p += a dp (box clamp);
//   q = normalize(exp(a dphi) (x) q) (LEFT, world frame). Accept iff E_phys strictly decreases, else restore.
// dbg (nullable): [0..35] H before the solve (col-major), [36..41] b, [42..47] raw dxi, [48..53] applied dxi.
template<typename T>
__device__ __forceinline__ void base_update_warp(WarpState<T>* st, const LMConfig& cfg, T eps_pos, T eps_ori, T* dbg) {
    const int lane = threadIdx.x & 31;
    constexpr int D = 6;
    const T ephys_before = st->ephys;
    if (lane == 0) {
        ++st->base_att;
        for (int i = 0; i < 3; ++i) st->bak_p[i] = st->base_p[i];
        for (int i = 0; i < 4; ++i) st->bak_q[i] = st->base_q[i];
        T Rb[9]; quat_to_mat3<T>(st->base_q, Rb);
        for (int i = 0; i < D*D; ++i) st->Hb[i] = (T)0;
        for (int i = 0; i < D; ++i) st->bb[i] = (T)0;
        for (int k = 0; k < K; ++k) {
            if (!((st->P.active >> k) & 1u)) continue;
            const T w = st->P.wp[k];
            if (!(w > (T)0)) continue;
            const T* fk = &st->targetX[16*k + 12];
            T arm[3], r[3];
            for (int i = 0; i < 3; ++i) arm[i] = Rb[i]*fk[0] + Rb[3+i]*fk[1] + Rb[6+i]*fk[2];
            for (int i = 0; i < 3; ++i) r[i] = Rb[i]*st->e_pos[3*k+0] + Rb[3+i]*st->e_pos[3*k+1] + Rb[6+i]*st->e_pos[3*k+2];
            T J[3*D];
            for (int i = 0; i < 3*D; ++i) J[i] = (T)0;
            J[0*D+0] = J[1*D+1] = J[2*D+2] = (T)1;
            J[0*D+4] =  arm[2]; J[0*D+5] = -arm[1];
            J[1*D+3] = -arm[2]; J[1*D+5] =  arm[0];
            J[2*D+3] =  arm[1]; J[2*D+4] = -arm[0];
            for (int a = 0; a < D; ++a) {
                T ba = (T)0;
                for (int i = 0; i < 3; ++i) ba += J[i*D+a] * r[i];
                st->bb[a] += w * ba;
                for (int c = 0; c < D; ++c) {
                    T h = (T)0;
                    for (int i = 0; i < 3; ++i) h += J[i*D+a] * J[i*D+c];
                    st->Hb[a*D + c] += w * h;
                }
            }
        }
        const T inv_sp2 = (T)1 / ((T)cfg.base_scale_p * (T)cfg.base_scale_p);
        const T inv_sR2 = (T)1 / ((T)cfg.base_scale_R * (T)cfg.base_scale_R);
        for (int i = 0; i < D; ++i) st->Hb[i*D + i] += (T)cfg.base_damping * (i < 3 ? inv_sp2 : inv_sR2);
        if (dbg) { for (int i = 0; i < 36; ++i) dbg[i] = st->Hb[i]; for (int i = 0; i < 6; ++i) dbg[36 + i] = st->bb[i]; }
        st->base_fail = 0;
    }
    __syncwarp(FULL_MASK);
    glass::warp::posv<T, D, 1, false, true, false>(st->Hb, st->bb, (T)0, &st->base_fail);
    __syncwarp(FULL_MASK);
    if (lane == 0) {
        if (st->base_fail) {
            ++st->base_nf;
        } else {
            T dp[3] = { st->bb[0], st->bb[1], st->bb[2] }, dr[3] = { st->bb[3], st->bb[4], st->bb[5] };
            if (dbg) for (int i = 0; i < 6; ++i) dbg[42 + i] = st->bb[i];
            if (!(isfinite(dp[0]) && isfinite(dp[1]) && isfinite(dp[2]) && isfinite(dr[0]) && isfinite(dr[1]) && isfinite(dr[2]))) {
                ++st->base_nf; st->base_fail = 1;
            } else {
                const T maxt = (T)cfg.base_max_translation, maxr = (T)cfg.base_max_rotation, a = (T)cfg.base_step_scale;
                const T nt = sqrt(dp[0]*dp[0] + dp[1]*dp[1] + dp[2]*dp[2]);
                if (maxt > (T)0 && nt > maxt) { const T sc = maxt / nt; for (int i = 0; i < 3; ++i) dp[i] *= sc; }
                const T nr = sqrt(dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]);
                if (maxr > (T)0 && nr > maxr) { const T sc = maxr / nr; for (int i = 0; i < 3; ++i) dr[i] *= sc; }
                if (dbg) { for (int i = 0; i < 3; ++i) { dbg[48 + i] = a * dp[i]; dbg[51 + i] = a * dr[i]; } }
                for (int i = 0; i < 3; ++i) st->base_p[i] = fmin(fmax(st->base_p[i] + a * dp[i], (T)cfg.base_lo[i]), (T)cfg.base_hi[i]);
                T w[3] = { a*dr[0], a*dr[1], a*dr[2] }, dq[4], qn[4];
                quat_from_rotvec<T>(w, dq);
                quat_mul_wxyz<T>(dq, st->base_q, qn);
                const T n = sqrt(qn[0]*qn[0] + qn[1]*qn[1] + qn[2]*qn[2] + qn[3]*qn[3]);
                for (int i = 0; i < 4; ++i) st->base_q[i] = qn[i] / n;
            }
        }
    }
    __syncwarp(FULL_MASK);
    if (st->base_fail) return;
    retarget_and_eval<T>(st);
    if (lane == 0) {
        const T after = e_phys<T>(st->P, st->pn, st->on, eps_pos, eps_ori);
        st->base_take = 0;
        if (after < ephys_before) { st->ephys = after; ++st->base_acc; }
        else {
            for (int i = 0; i < 3; ++i) st->base_p[i] = st->bak_p[i];
            for (int i = 0; i < 4; ++i) st->base_q[i] = st->bak_q[i];
            ++st->base_rej; st->base_take = 1;
        }
    }
    __syncwarp(FULL_MASK);
    if (st->base_take) retarget_and_eval<T>(st);
    __syncwarp(FULL_MASK);
}

template<typename T>
__device__ __forceinline__ void refresh(WarpState<T>* st, const T* s_xhom) {
    tree_fk_warp<T>(s_xhom, st->x, st->local, st->world);
    compose_targets_warp<T>(st->world, st->P.tool, st->targetX);
    eval_targets_warp<T>(st->P, st->targetX, st->e_pos, st->e_ori, st->pn, st->on);
}

template<typename T>
__global__ void tree_eval_kernel(const T* __restrict__ q, int batch, ProblemIn<T> in, const T* __restrict__ d_tool,
                                 const grid::robotModel<T>* __restrict__ model, T lambda, EvalOut<T> out) {
    extern __shared__ __align__(16) unsigned char smem[];
    T* s_xhom; block_prologue<T>(smem, model, s_xhom);
    WarpState<T>* st = warp_state<T>(smem);
    const int lane = threadIdx.x & 31, warps = blockDim.x >> 5;
    const int b = blockIdx.x * warps + (threadIdx.x >> 5);
    if (b >= batch) return;
    load_problem<T>(st->P, in, d_tool, b);
    load_base<T>(st, in, b);
    if (lane < N) st->x[lane] = q[size_t(b) * N + lane];
    __syncwarp(FULL_MASK);
    refresh<T>(st, s_xhom);
    if (lane < K) {
        for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c)
            out.pose[(size_t(b) * K + lane) * 16 + 4*r + c] = st->targetX[16*lane + 4*c + r];
        for (int r = 0; r < 3; ++r) { out.e[(size_t(b)*K + lane)*6 + r] = st->e_pos[3*lane + r];
                                      out.e[(size_t(b)*K + lane)*6 + 3 + r] = st->e_ori[3*lane + r]; }
        out.pn[size_t(b)*K + lane] = st->pn[lane]; out.on[size_t(b)*K + lane] = st->on[lane];
    }
    for (int k = 0; k < K; ++k) {
        T c[6]; target_column<T>(st->P, k, st->world, &st->targetX[16*k], lane, c);
        if (lane < N) for (int r = 0; r < 6; ++r) out.J[(size_t(b) * K * 6 + 6*k + r) * N + lane] = c[r];
    }
    accumulate_normal_equations_warp<T>(st->A, st->b, st->P, st->world, st->targetX, st->e_pos, st->e_ori, nullptr, nullptr);
    for (int i = lane; i < N * N; i += 32) out.A_raw[size_t(b) * N * N + i] = st->A[i];
    if (lane < N) out.b_raw[size_t(b) * N + lane] = st->b[lane];
    __syncwarp(FULL_MASK);
    compute_row_scales_warp<T>(st->scale, st->P, st->world, st->targetX);
    if (lane < 6 * K) out.scale[size_t(b) * 6 * K + lane] = st->scale[lane];
    accumulate_normal_equations_warp<T>(st->A, st->b, st->P, st->world, st->targetX, st->e_pos, st->e_ori, st->scale, st->colnorm);
    for (int i = lane; i < N * N; i += 32) out.A[size_t(b) * N * N + i] = st->A[i];
    if (lane < N) { out.b[size_t(b) * N + lane] = st->b[lane]; out.colnorm2[size_t(b) * N + lane] = st->colnorm[lane]; }
    __syncwarp(FULL_MASK);
    const unsigned int pinned = pin_null_columns_warp<T>(st->A, st->b, st->colnorm, st->P.active);
    if (lane == 0) out.pinned[b] = pinned;
    for (int i = lane; i < N * N; i += 32) out.Ap[size_t(b) * N * N + i] = st->A[i];
    if (lane < N) out.bp[size_t(b) * N + lane] = st->b[lane];
    __syncwarp(FULL_MASK);
    glass::warp::posv<T, N, 1, true, true, true>(st->A, st->b, lambda, &st->s_fail);
    __syncwarp(FULL_MASK);
    if (lane < N) out.dq[size_t(b) * N + lane] = st->s_fail == 0 ? st->b[lane] : T(0);
    if (lane == 0) out.fail[b] = st->s_fail;
}

// One base step at (q, base) with the joints held: the LM's base step, exposed for the oracle tests.
// out [B, 54 + 7 + 4]: dbg (H, b, raw dxi, applied dxi) | new base | accepted, numfail, ephys_before, ephys_after
template<typename T>
__global__ void tree_base_step_kernel(const T* __restrict__ q, int batch, ProblemIn<T> in, const T* __restrict__ d_tool,
                                      const grid::robotModel<T>* __restrict__ model, LMConfig cfg, T* __restrict__ out) {
    extern __shared__ __align__(16) unsigned char smem[];
    T* s_xhom; block_prologue<T>(smem, model, s_xhom);
    WarpState<T>* st = warp_state<T>(smem);
    const int lane = threadIdx.x & 31, warps = blockDim.x >> 5;
    const int b = blockIdx.x * warps + (threadIdx.x >> 5);
    if (b >= batch) return;
    load_problem<T>(st->P, in, d_tool, b);
    load_base<T>(st, in, b);
    if (lane < N) st->x[lane] = q[size_t(b) * N + lane];
    __syncwarp(FULL_MASK);
    refresh<T>(st, s_xhom);
    T* o = out + size_t(b) * 65;
    if (lane == 0) { st->ephys = e_phys<T>(st->P, st->pn, st->on, (T)cfg.eps_pos, (T)cfg.eps_ori); o[63] = st->ephys;
                     for (int i = 0; i < 54; ++i) o[i] = (T)0; }
    __syncwarp(FULL_MASK);
    base_update_warp<T>(st, cfg, (T)cfg.eps_pos, (T)cfg.eps_ori, o);
    if (lane == 0) {
        for (int i = 0; i < 3; ++i) o[54 + i] = st->base_p[i];
        for (int i = 0; i < 4; ++i) o[57 + i] = st->base_q[i];
        o[61] = (T)st->base_acc; o[62] = (T)st->base_nf; o[64] = e_phys<T>(st->P, st->pn, st->on, (T)cfg.eps_pos, (T)cfg.eps_ori);
    }
}


// ======================= multi-target coarse search (M5; port of bbc53a9 coarse_search_mt_kernel) =======================
template<typename T>
struct CoarseExtra {
    T ck[K], v_e_pos[3 * K], v_e_ori[3 * K], v_pn[K], v_on[K], v_ck[K], p_x[N];
    T total, trial_total, best_total, v_total, v_theta, best_ephys, cost0;
    int stall, take_best;
};
template<typename T>
__host__ __device__ constexpr size_t coarse_shared_bytes(int warps) {
    return shared_bytes<T>(warps) + size_t(warps) * ((sizeof(CoarseExtra<T>) + 15) / 16) * 16;
}
template<typename T>
__device__ __forceinline__ CoarseExtra<T>* coarse_extra(unsigned char* smem) {
    const int warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    return reinterpret_cast<CoarseExtra<T>*>(smem + shared_bytes<T>(warps) + size_t(warp) * ((sizeof(CoarseExtra<T>) + 15) / 16) * 16);
}
__device__ __forceinline__ uint32_t wanghash(uint32_t a) {
    a = (a ^ 61u) ^ (a >> 16); a *= 9u; a ^= (a >> 4); a *= 0x27d4eb2du; a ^= (a >> 15); return a;
}
__device__ __forceinline__ uint32_t semantic_rng(uint32_t root, uint32_t sub, uint32_t sample, uint32_t it, uint32_t j, uint32_t draw) {
    uint32_t h = root;
    h = wanghash(h ^ (sub * 0x9E3779B9u)); h = wanghash(h ^ (sample * 0x85EBCA6Bu)); h = wanghash(h ^ (it * 0xC2B2AE35u));
    h = wanghash(h ^ (j * 0x27D4EB2Du));   h = wanghash(h ^ (draw * 0x165667B1u));
    return h;
}
__device__ __forceinline__ unsigned int joint_target_mask(int j) {
    unsigned int m = 0u;
    for (int k = 0; k < K; ++k) m |= ((hjcd_targets::TARGET_ANCESTOR_MASK[k] >> j) & 1u) << k;
    return m;
}
__device__ __forceinline__ bool in_subtree(int i, int j) {        // is joint i == j or a descendant of j
    for (int a = i; a >= 0; a = hjcd_targets::JOINT_PARENT[a]) if (a == j) return true;
    return false;
}
template<typename T>   // bbc53a9 scaled_target_cost: sum_r w_p s^2 e_p^2 + w_R s^2 e_R^2
__device__ __forceinline__ T scaled_target_cost(const T* e_p, const T* e_R, T wp, T wo, const T* s_scale, int k) {
    T c = (T)0;
    for (int r = 0; r < 3; ++r) { const T sp = s_scale[6*k + r], so = s_scale[6*k + 3 + r]; c += wp*sp*sp*e_p[r]*e_p[r] + wo*so*so*e_R[r]*e_R[r]; }
    return c;
}
template<typename T>   // eval of the targets in tmask; full => total = sum, else total -= old ck, += new ck (bbc53a9 order)
__device__ __forceinline__ void coarse_eval(WarpState<T>* st, CoarseExtra<T>* cx, unsigned int tmask, bool full, T* io_total) {
    const int lane = threadIdx.x & 31;
    if (!full && lane == 0) { T sub = (T)0; for (int k = 0; k < K; ++k) if ((tmask >> k) & 1u) sub += cx->ck[k]; *io_total -= sub; }
    __syncwarp(FULL_MASK);
    if (lane < K && (full || ((tmask >> lane) & 1u))) {
        const int k = lane;
        T ep[3] = {0, 0, 0}, eR[3] = {0, 0, 0}, pn = 0, on = 0, ck = 0;
        if ((st->P.active >> k) & 1u) {
            target_residual<T>(&st->targetX[16*k], &st->P.tgt_p[3*k], &st->P.tgt_q[4*k], ep, eR, st->P.mode[k],
                               &st->P.axis[3*k], &st->P.twist[5*k], &on);
            pn = sqrt(ep[0]*ep[0] + ep[1]*ep[1] + ep[2]*ep[2]);
            ck = scaled_target_cost<T>(ep, eR, st->P.wp[k], st->P.wo[k], st->scale, k);
        }
        for (int r = 0; r < 3; ++r) { st->e_pos[3*k + r] = ep[r]; st->e_ori[3*k + r] = eR[r]; }
        st->pn[k] = pn; st->on[k] = on; cx->ck[k] = ck;
    }
    __syncwarp(FULL_MASK);
    if (lane == 0) {
        T c = (T)0; for (int k = 0; k < K; ++k) if (((full ? st->P.active : tmask) >> k) & 1u) c += cx->ck[k];
        if (full) *io_total = c; else *io_total += c;
    }
    __syncwarp(FULL_MASK);
}
template<typename T>
__device__ __forceinline__ void coarse_full_refresh(WarpState<T>* st, CoarseExtra<T>* cx, const T* s_xhom) {
    tree_fk_warp<T>(s_xhom, st->x, st->local, st->world);
    compose_targets_warp<T>(st->world, st->P.tool, st->targetX);
    compute_row_scales_warp<T>(st->scale, st->P, st->world, st->targetX);
    coarse_eval<T>(st, cx, 0u, true, &cx->total);
}
// Subtree update after joint j moved to x[j]: rebuild its local transform, recompose world for the subtree.
template<typename T>
__device__ __forceinline__ void subtree_fk_warp(WarpState<T>* st, const T* s_xhom, int j) {
    const int lane = threadIdx.x & 31;
    if (lane == 0) grid::update_XmatHom_joint<T>(&st->local[16 * j], s_xhom, j, st->x[j]);
    __syncwarp(FULL_MASK);
    const bool mine = lane < N && in_subtree(lane, j);
    for (int d = hjcd_targets::JOINT_DEPTH[j]; d <= hjcd_targets::MAX_DEPTH; ++d) {
        if (mine && hjcd_targets::JOINT_DEPTH[lane] == d) {
            const int p = hjcd_targets::JOINT_PARENT[lane];
            if (p < 0) for (int m = 0; m < 16; ++m) st->world[16 * lane + m] = st->local[16 * lane + m];
            else glass::thread::gemm<T, 4, 4, 4>(T(1), &st->world[16 * p], &st->local[16 * lane], &st->world[16 * lane]);
        }
        __syncwarp(FULL_MASK);
    }
    compose_targets_warp<T>(st->world, st->P.tool, st->targetX);   // unaffected targets recompose to identical values
}
// bbc53a9 coord_proposal for joint j (= the calling lane): out (v, delta, pred); pred <= 0 => invalid.
template<typename T>
__device__ __forceinline__ void coord_proposal(const WarpState<T>* st, int j, const CoarseConfig& cc, T* out_v, T* out_d, T* out_pred) {
    *out_pred = (T)-1; *out_d = (T)0; *out_v = st->x[j];
    const unsigned int affected = joint_target_mask(j) & st->P.active;
    if (affected == 0u) return;
    T g = (T)0, h = (T)0;
    for (int k = 0; k < K; ++k) {
        if (!((affected >> k) & 1u)) continue;
        T c[6]; target_column<T>(st->P, k, st->world, &st->targetX[16*k], j, c);
        const T wp = st->P.wp[k], wo = (st->P.mode[k] == ORI_NONE) ? (T)0 : st->P.wo[k];
        for (int r = 0; r < 3; ++r) {
            const T sp = st->scale[6*k + r], so = st->scale[6*k + 3 + r];
            const T Wp = wp * sp * sp, Wo = wo * so * so;
            g += Wp * c[r] * st->e_pos[3*k + r] + Wo * c[3 + r] * st->e_ori[3*k + r];
            h += Wp * c[r] * c[r]               + Wo * c[3 + r] * c[3 + r];
        }
    }
    if (!(h > (T)cc.h_min)) return;
    T delta = g / (h + (T)cc.lambda_coord);
    delta = fmin(fmax(delta, -(T)cc.max_step), (T)cc.max_step);
    const T lo = (T)hjcd_targets::JOINT_LOWER[j], hi = (T)hjcd_targets::JOINT_UPPER[j];
    const T v = fmin(fmax(st->x[j] + delta, lo), hi);
    delta = v - st->x[j];
    if (delta == (T)0) return;
    const T pred = (T)2 * delta * g - delta * delta * h;
    if (!(pred > (T)0)) return;
    *out_v = v; *out_d = delta; *out_pred = pred;
}

template<typename T>
__global__ void tree_coarse_kernel(T* __restrict__ x, int batch, ProblemIn<T> in, const T* __restrict__ d_tool,
                                   const grid::robotModel<T>* __restrict__ model, LMConfig cfg, CoarseConfig cc,
                                   CoarseIO<T> io) {
    extern __shared__ __align__(16) unsigned char smem[];
    T* s_xhom; block_prologue<T>(smem, model, s_xhom);
    WarpState<T>* st = warp_state<T>(smem);
    CoarseExtra<T>* cx = coarse_extra<T>(smem);
    const int lane = threadIdx.x & 31, warps = blockDim.x >> 5;
    const int b = blockIdx.x * warps + (threadIdx.x >> 5);
    if (b >= batch) return;
    if (io.use_coarse && !io.use_coarse[b]) return;              // auto: this candidate goes to LM as seeded
    const int S = in.seeds_per_problem, pid = b / S;
    const uint32_t root = io.problem_seeds ? io.problem_seeds[pid]
                        : (uint32_t)((cc.seed ^ ((unsigned long long)pid * 0x9E3779B97F4A7C15ull)) ^
                                     ((cc.seed ^ ((unsigned long long)pid * 0x9E3779B97F4A7C15ull)) >> 32));
    const uint32_t sample = io.candidate_ids ? io.candidate_ids[b] : (uint32_t)(b - pid * S);
    const T eps_pos = (T)cfg.eps_pos, eps_ori = (T)cfg.eps_ori;
    load_problem<T>(st->P, in, d_tool, b);
    load_base<T>(st, in, b);                                     // base FIXED here: targets re-expressed once
    if (lane < N) st->x[lane] = x[size_t(b) * N + lane];
    if (lane == 0) cx->stall = 0;
    __syncwarp(FULL_MASK);
    coarse_full_refresh<T>(st, cx, s_xhom);
    if (lane == 0) { cx->best_ephys = e_phys<T>(st->P, st->pn, st->on, eps_pos, eps_ori); cx->best_total = cx->total; cx->cost0 = cx->total; }
    if (lane < N) st->best_x[lane] = st->x[lane];
    __syncwarp(FULL_MASK);
    int iters = 0, kicks = 0;
    for (int it = 0; it < cc.k_max; ++it) {
        ++iters;
        compute_row_scales_warp<T>(st->scale, st->P, st->world, st->targetX);
        coarse_eval<T>(st, cx, 0u, true, &cx->total);
        T v = (T)0, delta = (T)0, pred = (T)-1;
        if (lane < N) coord_proposal<T>(st, lane, cc, &v, &delta, &pred);
        const T cost_before = cx->total;
        int best_j = (pred > (T)0) ? lane : -1; T best_p = pred, best_v = v;
        for (int off = 16; off > 0; off >>= 1) {
            const T op = __shfl_down_sync(FULL_MASK, best_p, off);
            const int oj = __shfl_down_sync(FULL_MASK, best_j, off);
            const T ov = __shfl_down_sync(FULL_MASK, best_v, off);
            const bool take = (oj >= 0) && (best_j < 0 || op > best_p || (op == best_p && oj < best_j));
            if (take) { best_p = op; best_j = oj; best_v = ov; }
        }
        best_j = __shfl_sync(FULL_MASK, best_j, 0); best_v = __shfl_sync(FULL_MASK, best_v, 0);
        int accepted = 0;
        if (best_j >= 0) {
            const unsigned int tm = joint_target_mask(best_j) & st->P.active;
            if (lane == 0) { cx->v_theta = st->x[best_j]; cx->v_total = cx->total; }
            if (lane < K && ((tm >> lane) & 1u)) {
                for (int c = 0; c < 3; ++c) { cx->v_e_pos[3*lane + c] = st->e_pos[3*lane + c]; cx->v_e_ori[3*lane + c] = st->e_ori[3*lane + c]; }
                cx->v_pn[lane] = st->pn[lane]; cx->v_on[lane] = st->on[lane]; cx->v_ck[lane] = cx->ck[lane];
            }
            __syncwarp(FULL_MASK);
            if (lane == 0) st->x[best_j] = best_v;
            __syncwarp(FULL_MASK);
            subtree_fk_warp<T>(st, s_xhom, best_j);
            if (lane == 0) cx->trial_total = cx->total;
            __syncwarp(FULL_MASK);
            coarse_eval<T>(st, cx, tm, false, &cx->trial_total);
            accepted = (cx->trial_total + (T)1e-20 < cost_before) ? 1 : 0;
            if (accepted) {
                if (lane == 0) { cx->total = cx->trial_total; cx->stall = 0; }
            } else {
                if (lane == 0) st->x[best_j] = cx->v_theta;
                __syncwarp(FULL_MASK);
                subtree_fk_warp<T>(st, s_xhom, best_j);
                if (lane < K && ((tm >> lane) & 1u)) {
                    for (int c = 0; c < 3; ++c) { st->e_pos[3*lane + c] = cx->v_e_pos[3*lane + c]; st->e_ori[3*lane + c] = cx->v_e_ori[3*lane + c]; }
                    st->pn[lane] = cx->v_pn[lane]; st->on[lane] = cx->v_on[lane]; cx->ck[lane] = cx->v_ck[lane];
                }
                if (lane == 0) cx->total = cx->v_total;
            }
            __syncwarp(FULL_MASK);
        }
        if (!accepted && lane == 0) ++cx->stall;
        __syncwarp(FULL_MASK);
        if (lane == 0) {
            cx->take_best = 0;
            const T ep = e_phys<T>(st->P, st->pn, st->on, eps_pos, eps_ori);
            if (ep < cx->best_ephys) { cx->best_ephys = ep; cx->take_best = 1; }
            if (cx->total < cx->best_total) cx->best_total = cx->total;
        }
        __syncwarp(FULL_MASK);
        if (lane < N && cx->take_best) st->best_x[lane] = st->x[lane];
        __syncwarp(FULL_MASK);
        if (cx->stall >= cc.stall_lim) {                          // stall kick (no collision gate => first attempt kept)
            if (lane < N) {
                cx->p_x[lane] = st->x[lane];
                const T lo = (T)hjcd_targets::JOINT_LOWER[lane], hi = (T)hjcd_targets::JOINT_UPPER[lane];
                const uint32_t h = semantic_rng(root, 0x2u, sample, (uint32_t)it, (uint32_t)lane, 0u);
                const T u = (T)((h & 0xFFFFFFu) / (T)0x1000000u) - (T)0.5;
                st->x[lane] = fmin(fmax(cx->p_x[lane] + (T)0.1 * (hi - lo) * u, lo), hi);
            }
            __syncwarp(FULL_MASK);
            coarse_full_refresh<T>(st, cx, s_xhom);
            if (lane == 0) cx->stall = 0;
            ++kicks;
            __syncwarp(FULL_MASK);
        }
        if (all_active_converged<T>(st->P, st->pn, st->on, eps_pos, eps_ori)) break;
    }
    if (lane < N) st->x[lane] = st->best_x[lane];
    __syncwarp(FULL_MASK);
    coarse_full_refresh<T>(st, cx, s_xhom);
    if (lane < N) x[size_t(b) * N + lane] = st->x[lane];            // the LM seed (base untouched: never written)
    if (lane < K) { if (io.pn) io.pn[size_t(b) * K + lane] = st->pn[lane]; if (io.on) io.on[size_t(b) * K + lane] = st->on[lane]; }
    if (lane == 0) {
        if (io.cost0) io.cost0[b] = cx->cost0;
        if (io.cost1) io.cost1[b] = cx->total;
        if (io.iters) io.iters[b] = iters;
        if (io.kicks) io.kicks[b] = kicks;
        if (io.success) io.success[b] = all_active_converged<T>(st->P, st->pn, st->on, eps_pos, eps_ori) ? 1 : 0;
    }
}

// Diagnostic: every joint's proposal at q (no state change). out [B, N, 4] = (v, delta, pred, g-sign-check unused=0)
template<typename T>
__global__ void tree_proposal_kernel(const T* __restrict__ q, int batch, ProblemIn<T> in, const T* __restrict__ d_tool,
                                     const grid::robotModel<T>* __restrict__ model, CoarseConfig cc, T* __restrict__ out) {
    extern __shared__ __align__(16) unsigned char smem[];
    T* s_xhom; block_prologue<T>(smem, model, s_xhom);
    WarpState<T>* st = warp_state<T>(smem);
    CoarseExtra<T>* cx = coarse_extra<T>(smem);
    const int lane = threadIdx.x & 31, warps = blockDim.x >> 5;
    const int b = blockIdx.x * warps + (threadIdx.x >> 5);
    if (b >= batch) return;
    load_problem<T>(st->P, in, d_tool, b);
    load_base<T>(st, in, b);
    if (lane < N) st->x[lane] = q[size_t(b) * N + lane];
    __syncwarp(FULL_MASK);
    coarse_full_refresh<T>(st, cx, s_xhom);
    T v = 0, d = 0, pr = -1;
    if (lane < N) {
        coord_proposal<T>(st, lane, cc, &v, &d, &pr);
        T* o = out + (size_t(b) * N + lane) * 6; o[0] = v; o[1] = d; o[2] = pr; o[3] = cx->total; o[4] = o[5] = (T)0;
    }
    // Subtree-FK oracle: for every joint with a valid proposal, apply it through the subtree path + incremental
    // re-score, then a FULL FK + full re-score of the same state; record max |targetX diff| and |cost diff|.
    for (int j = 0; j < N; ++j) {
        const T vj = __shfl_sync(FULL_MASK, v, j), pj = __shfl_sync(FULL_MASK, pr, j);
        if (!(pj > (T)0)) continue;
        const unsigned int tm = joint_target_mask(j) & st->P.active;
        const T x0 = st->x[j], tot0 = cx->total;
        __syncwarp(FULL_MASK);
        if (lane == 0) { st->x[j] = vj; cx->trial_total = cx->total; }
        __syncwarp(FULL_MASK);
        subtree_fk_warp<T>(st, s_xhom, j);
        coarse_eval<T>(st, cx, tm, false, &cx->trial_total);
        T sub[16 * K]; for (int i = 0; i < 16 * K; ++i) sub[i] = st->targetX[i];
        const T inc = cx->trial_total;
        __syncwarp(FULL_MASK);
        tree_fk_warp<T>(s_xhom, st->x, st->local, st->world);
        compose_targets_warp<T>(st->world, st->P.tool, st->targetX);
        coarse_eval<T>(st, cx, 0u, true, &cx->trial_total);
        if (lane == 0) {
            T md = 0; for (int i = 0; i < 16 * K; ++i) md = fmax(md, fabs(sub[i] - st->targetX[i]));
            T* o = out + (size_t(b) * N + j) * 6; o[4] = md; o[5] = fabs(inc - cx->trial_total);
            st->x[j] = x0;
        }
        __syncwarp(FULL_MASK);
        coarse_full_refresh<T>(st, cx, s_xhom);                    // restore (same scales as before: pure function of x)
        if (lane == 0) cx->total = tot0;
        __syncwarp(FULL_MASK);
    }
}

template<typename T>
__global__ void tree_lm_kernel(const T* __restrict__ q0, int batch, ProblemIn<T> in, const T* __restrict__ d_tool,
                               const grid::robotModel<T>* __restrict__ model, LMConfig cfg, LMOut<T> out) {
    extern __shared__ __align__(16) unsigned char smem[];
    T* s_xhom; block_prologue<T>(smem, model, s_xhom);
    WarpState<T>* st = warp_state<T>(smem);
    const int lane = threadIdx.x & 31, warps = blockDim.x >> 5;
    const int b = blockIdx.x * warps + (threadIdx.x >> 5);
    if (b >= batch) return;
    const T eps_pos = (T)cfg.eps_pos, eps_ori = (T)cfg.eps_ori, stag_rel = (T)cfg.stag_rel;
    const T lambda_min = (T)1e-12, lambda_max = (T)1e6;
    const int stall_lim = 5;
    T lambda = (T)cfg.lambda_init;
    int iters = 0, n_accept = 0, n_fail = 0, n_kick = 0;

    load_problem<T>(st->P, in, d_tool, b);
    load_base<T>(st, in, b);
    if (lane < N) st->x[lane] = q0[size_t(b) * N + lane];
    if (lane == 0) { st->s_break = 0; st->stall = 0; st->prev_cost = (T)-1; st->stag_n = 0; st->prev_ephys = (T)-1; }
    if (lane < 6 * K) st->scale[lane] = (T)1;
    __syncwarp(FULL_MASK);
    refresh<T>(st, s_xhom);
    compute_row_scales_warp<T>(st->scale, st->P, st->world, st->targetX);
    if (lane == 0) {
        st->cost = weighted_cost<T>(st->P, st->e_pos, st->e_ori, st->scale);
        worst_active_errors<T>(st->P, st->pn, st->on, &st->max_pn, &st->max_on);
        st->best_cost = st->cost;
        st->best_ephys = e_phys<T>(st->P, st->pn, st->on, eps_pos, eps_ori);
        out.ephys0[b] = st->best_ephys;
        if (all_active_converged<T>(st->P, st->pn, st->on, eps_pos, eps_ori)) st->s_break = 1;
    }
    if (lane < K) { out.pn0[size_t(b) * K + lane] = st->pn[lane]; out.on0[size_t(b) * K + lane] = st->on[lane]; }
    if (lane < N) st->best_x[lane] = st->x[lane];
    __syncwarp(FULL_MASK);

    for (int it = 0; it < cfg.k_max && !st->s_break; ++it) {
        ++iters;
        // Fresh scales for this linearization; the cost is re-expressed in them (frozen for the trials).
        compute_row_scales_warp<T>(st->scale, st->P, st->world, st->targetX);
        if (lane == 0) st->cost = weighted_cost<T>(st->P, st->e_pos, st->e_ori, st->scale);
        __syncwarp(FULL_MASK);
        accumulate_normal_equations_warp<T>(st->A, st->b, st->P, st->world, st->targetX, st->e_pos, st->e_ori,
                                            st->scale, st->colnorm);
        pin_null_columns_warp<T>(st->A, st->b, st->colnorm, st->P.active);
        if (lane < N) { st->diagA[lane] = st->A[lane * N + lane]; st->g[lane] = st->b[lane]; }
        __syncwarp(FULL_MASK);
        glass::warp::posv<T, N, 1, true, true, true>(st->A, st->b, lambda, &st->s_fail);
        __syncwarp(FULL_MASK);
        if (st->s_fail != 0) ++n_fail;
        if (lane < N) st->dq[lane] = (st->s_fail == 0) ? st->b[lane] : (T)0;
        __syncwarp(FULL_MASK);

        if (lane == 0) {
            T R;
            if      (st->max_pn > (T)1e-2 || st->max_on > (T)0.6)  R = (T)0.38;
            else if (st->max_pn > (T)1e-3 || st->max_on > (T)0.25) R = (T)0.22;
            else if (st->max_pn > (T)2e-4 || st->max_on > (T)0.08) R = (T)0.12;
            else                                                   R = (T)0.05;
            T nrm = (T)0;
            for (int i = 0; i < N; ++i) nrm += st->dq[i] * st->dq[i];
            nrm = sqrt(nrm);
            if (nrm > R) { const T s = R / (nrm + (T)1e-18); for (int i = 0; i < N; ++i) st->dq[i] *= s; }
            const T clip = (st->max_pn > (T)1e-2) ? (T)0.30 : (st->max_pn > (T)1e-3) ? (T)0.15 :
                           (st->max_pn > (T)2e-4) ? (T)0.08 : (T)0.03;
            for (int i = 0; i < N; ++i) { st->dq[i] = fmin(fmax(st->dq[i], -clip), clip); st->x_old[i] = st->x[i]; }
            st->accepted = 0;
        }
        __syncwarp(FULL_MASK);

        T best_a = (T)0;
        for (int tries = 0; tries < 4; ++tries) {
            const T a = (T)1 / (T)(1 << tries);
            if (lane < N) {
                const T lo = (T)hjcd_targets::JOINT_LOWER[lane], hi = (T)hjcd_targets::JOINT_UPPER[lane];
                st->x[lane] = fmin(fmax(st->x_old[lane] + a * st->dq[lane], lo), hi);
            }
            __syncwarp(FULL_MASK);
            refresh<T>(st, s_xhom);
            if (lane == 0) {
                st->trial_cost = weighted_cost<T>(st->P, st->e_pos, st->e_ori, st->scale);
                if (st->trial_cost + (T)1e-20 < st->cost) st->accepted = 1;
            }
            __syncwarp(FULL_MASK);
            if (st->accepted) { best_a = a; break; }
        }
        __syncwarp(FULL_MASK);
        if (st->accepted) ++n_accept;
        if (lane == 0) {
            if (st->accepted) {
                const T ared = st->cost - st->trial_cost;
                T pred = (T)0;
                for (int i = 0; i < N; ++i) { const T ad = best_a * st->dq[i]; pred += lambda * st->diagA[i] * ad * ad + ad * st->g[i]; }
                pred = fmax((T)1e-20, pred);
                const T rho = ared / pred;
                if      (rho > (T)0.90) lambda = fmax(lambda * (T)0.3, lambda_min);
                else if (rho > (T)0.50) lambda = fmax(lambda * (T)0.5, lambda_min);
                else if (rho < (T)0.25) lambda = fmin(lambda * (T)3.0, lambda_max);
                st->cost = st->trial_cost;
                if (st->prev_cost > (T)0 && (st->prev_cost - st->cost) / st->prev_cost < (T)1e-9) ++st->stall;
                else st->stall = 0;
                st->prev_cost = st->cost;
            } else {
                lambda = fmin(lambda * (T)3.0, lambda_max);
                ++st->stall;
            }
        }
        // bbc53a9 updated lambda on lane 0 only, but posv's diagonal shift is lane-strided, so its
        // joints 1..N-1 kept lambda_init for the whole solve. uniform_lambda=1 broadcasts
        // the updated lambda; 0 (the default) reproduces bbc53a9 exactly. M3 measured both: uniform +1 on an
        // easy corpus, -3 on a hard one (temp/humanoid_v1_m3/compare), so the legacy behaviour stays the default.
        if (cfg.uniform_lambda) lambda = __shfl_sync(FULL_MASK, lambda, 0);
        __syncwarp(FULL_MASK);
        if (!st->accepted) {
            if (lane < N) st->x[lane] = st->x_old[lane];
            __syncwarp(FULL_MASK);
            refresh<T>(st, s_xhom);
        }
        if (lane == 0) {
            worst_active_errors<T>(st->P, st->pn, st->on, &st->max_pn, &st->max_on);
            if (st->cost < st->best_cost) st->best_cost = st->cost;
            const T ep = e_phys<T>(st->P, st->pn, st->on, eps_pos, eps_ori);
            st->ephys = ep;
            st->take_best = (ep < st->best_ephys) ? 1 : 0;
            if (st->take_best) st->best_ephys = ep;
            if (cfg.stag_patience > 0) {
                const T prev = st->prev_ephys;
                if (prev > (T)0 && (prev - ep) / prev < stag_rel) ++st->stag_n; else st->stag_n = 0;
                if (st->stag_n >= cfg.stag_patience) st->s_break = 1;
            }
            st->prev_ephys = ep;
        }
        __syncwarp(FULL_MASK);
        if (lane < N && st->take_best) st->best_x[lane] = st->x[lane];      // before any kick, as in bbc53a9
        if (lane == 0 && st->take_best && st->has_base) {                     // the base travels with the joints
            for (int i = 0; i < 3; ++i) st->best_base_p[i] = st->base_p[i];
            for (int i = 0; i < 4; ++i) st->best_base_q[i] = st->base_q[i];
        }
        __syncwarp(FULL_MASK);
        if (st->has_base && cfg.base_enabled && !st->s_break && cfg.base_interval > 0 && (it % cfg.base_interval) == 0) {
            base_update_warp<T>(st, cfg, eps_pos, eps_ori, nullptr);
            int improved = 0;
            if (lane == 0) improved = (st->ephys < st->best_ephys) ? 1 : 0;
            improved = __shfl_sync(FULL_MASK, improved, 0);
            if (lane == 0 && improved) {
                st->best_ephys = st->ephys;
                for (int i = 0; i < 3; ++i) st->best_base_p[i] = st->base_p[i];
                for (int i = 0; i < 4; ++i) st->best_base_q[i] = st->base_q[i];
            }
            if (lane < N && improved) st->best_x[lane] = st->x[lane];
            __syncwarp(FULL_MASK);
        }
        if (lane == 0) {
            st->kicked = 0;
            if (st->stall >= stall_lim) {
                for (int i = 0; i < N; ++i) {
                    const T lo = (T)hjcd_targets::JOINT_LOWER[i], hi = (T)hjcd_targets::JOINT_UPPER[i];
                    const unsigned int u = 0x9E3779B9u ^ (unsigned int)(i * 0xC2B2AE35u);
                    const T kick = (T)0.005 * (hi - lo) * ((u & 1u) ? (T)1 : (T)-1);
                    st->x[i] = fmin(fmax(st->x[i] + kick, lo), hi);
                }
                st->stall = 0; st->kicked = 1;
            }
        }
        __syncwarp(FULL_MASK);
        if (st->kicked) { ++n_kick; refresh<T>(st, s_xhom); }
        if (lane == 0 && all_active_converged<T>(st->P, st->pn, st->on, eps_pos, eps_ori)) st->s_break = 1;
        __syncwarp(FULL_MASK);
    }

    if (lane < N) st->x[lane] = st->best_x[lane];
    if (lane == 0 && st->has_base) {
        for (int i = 0; i < 3; ++i) st->base_p[i] = st->best_base_p[i];
        for (int i = 0; i < 4; ++i) st->base_q[i] = st->best_base_q[i];
    }
    __syncwarp(FULL_MASK);
    if (st->has_base && lane < K) world_target_to_base<T>(&st->tgt_wp[3*lane], &st->tgt_wq[4*lane], st->base_p, st->base_q,
                                                          &st->P.tgt_p[3*lane], &st->P.tgt_q[4*lane]);
    __syncwarp(FULL_MASK);
    refresh<T>(st, s_xhom);
    if (lane < K) { out.pn[size_t(b) * K + lane] = st->pn[lane]; out.on[size_t(b) * K + lane] = st->on[lane]; }
    if (lane < N) out.x[size_t(b) * N + lane] = st->x[lane];
    if (lane == 0) {
        out.cost[b] = weighted_cost<T>(st->P, st->e_pos, st->e_ori, st->scale);
        out.ephys[b] = e_phys<T>(st->P, st->pn, st->on, eps_pos, eps_ori);
        out.success[b] = all_active_converged<T>(st->P, st->pn, st->on, eps_pos, eps_ori) ? 1 : 0;
        out.iters[b] = iters; out.accepted[b] = n_accept; out.solve_fail[b] = n_fail; out.kicks[b] = n_kick;
        if (st->has_base && out.base) {
            for (int i = 0; i < 3; ++i) out.base[size_t(b) * 7 + i] = st->base_p[i];
            for (int i = 0; i < 4; ++i) out.base[size_t(b) * 7 + 3 + i] = st->base_q[i];
        }
        if (st->has_base && out.base_diag) {
            out.base_diag[size_t(b) * 4 + 0] = st->base_att; out.base_diag[size_t(b) * 4 + 1] = st->base_acc;
            out.base_diag[size_t(b) * 4 + 2] = st->base_nf;  out.base_diag[size_t(b) * 4 + 3] = st->base_rej;
        }
    }
}

template<typename T>
struct Device {
    grid::robotModel_owner<T> model;
    hjcd::DeviceAllocations mem;
    template<typename X> X* upload(const X* h, size_t n) {
        X* d = nullptr; mem.allocate(d, sizeof(X) * n);
        CUDA_OK(cudaMemcpy(d, h, sizeof(X) * n, cudaMemcpyHostToDevice));
        return d;
    }
    template<typename X> X* alloc(size_t n) { X* d = nullptr; mem.allocate(d, sizeof(X) * (n ? n : 1)); return d; }
    Device() {
        const char* op = nullptr;
        const cudaError_t e = model.init(&op);
        if (e != cudaSuccess) throw std::runtime_error(std::string("GRiD robot model init failed at ") + (op ? op : "?")
                                                       + ": " + cudaGetErrorString(e));
    }
};

template<typename T>
ProblemIn<T> upload_problem(Device<T>& dev, int B, const ProblemIn<T>& in, const T*& d_tool) {
    // B = number of PROBLEMS whose data is uploaded (one row per configuration in the M3 entry points).
    ProblemIn<T> d = in;
    d.target = dev.upload(in.target, size_t(B) * K * 7);
    d.active = dev.upload(in.active, size_t(B));
    d.w_pos = dev.upload(in.w_pos, size_t(B) * K);
    d.w_ori = dev.upload(in.w_ori, size_t(B) * K);
    d.mode = dev.upload(in.mode, size_t(B) * K);
    d.axis = dev.upload(in.axis, size_t(B) * K * 3);
    d.twist = dev.upload(in.twist, size_t(B) * K * 5);
    if (in.tool && in.tool_per_problem) {
        d_tool = dev.upload(in.tool, size_t(B) * K * 16);
    } else {
        T tool[K * 16];
        for (int i = 0; i < K * 16; ++i) tool[i] = in.tool ? in.tool[i] : (T)hjcd_targets::H_TARGET_TOOL[i];
        d_tool = dev.upload(tool, size_t(K) * 16);
        d.tool_per_problem = 0;
    }
    return d;
}

template<typename X> void download(X* h, const X* d, size_t n) { CUDA_OK(cudaMemcpy(h, d, sizeof(X) * n, cudaMemcpyDeviceToHost)); }


// ---- segmented top-M (port of bbc53a9 segmented_topM_kernel without the collision channels) ----
// coll (M6a): 0 unless the env channel ranks (RANK: FREE 0 < COLLIDING 1 < INVALID 2); bbc53a9 order (cls, coll, E, s).
struct Cand { int cls; int coll; double ephys; int seed; };
__device__ __forceinline__ bool cand_better(const Cand& a, const Cand& b) {
    if (a.cls != b.cls) return a.cls < b.cls;
    if (a.coll != b.coll) return a.coll < b.coll;
    if (a.ephys != b.ephys) return a.ephys < b.ephys;
    return a.seed < b.seed;
}

template<typename T>
__global__ void tree_topM_kernel(const T* __restrict__ q, const T* __restrict__ pe, const T* __restrict__ oe,
                                 const T* __restrict__ cost, const unsigned char* __restrict__ succ,
                                 const T* __restrict__ seeds, const unsigned int* __restrict__ active,
                                 const double eps_pos, const double eps_ori, const int S, const int P, const int M,
                                 SelectOut<T> o, const T* __restrict__ base, const T* __restrict__ base_in,
                                 const unsigned char* __restrict__ env_status, const T* __restrict__ env_mincl, int env_sel,
                                 EnvOut<T> eo, const unsigned char* __restrict__ self_status, const T* __restrict__ self_mincl,
                                 int self_sel, SelfOut<T> xo) {
    const int p = blockIdx.x;
    if (p >= P) return;
    const unsigned int mask = active[p];
    const int tid = threadIdx.x, nt = blockDim.x;
    extern __shared__ __align__(16) unsigned char s_raw[];
    Cand* s_best = reinterpret_cast<Cand*>(s_raw);
    int* s_cnt = reinterpret_cast<int*>(s_best + nt);       // NCNT ints per thread (round 0)
    constexpr int NCNT = 11;    // M5: solved, valid, converged | M6a env: free, colliding, invalid, solved&free |
                                // M6b self: free, colliding, invalid | solved & free in every evaluated channel
    int* s_taken = s_cnt + nt * NCNT;                       // M selected seeds (-1 = pad)
    for (int m = 0; m < M; ++m) {
        Cand best{2, 0, INFINITY, 1 << 30};
        int c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0, c6 = 0, c7 = 0, c8 = 0, c9 = 0, c10 = 0;
        for (int s = tid; s < S; s += nt) {
            const size_t b = size_t(p) * S + s;
            bool taken = false;
            for (int t = 0; t < m; ++t) if (s_taken[t] == s) { taken = true; break; }
            bool finite = true;
            for (int j = 0; j < N; ++j) if (!isfinite((double)q[b * N + j])) { finite = false; break; }
            double ephys = 0.0; bool within = true;
            for (int k = 0; k < K; ++k) {
                if (!((mask >> k) & 1u)) continue;
                const double ep = (double)pe[b * K + k], eo = (double)oe[b * K + k];
                if (!isfinite(ep) || !isfinite(eo)) { finite = false; continue; }
                const double rp = ep / eps_pos, ro = eo / eps_ori; ephys += rp * rp + ro * ro;
                if (ep > eps_pos || eo > eps_ori) within = false;
            }
            int cls; if (!finite) { cls = 2; ephys = INFINITY; } else if (within) cls = 0; else cls = 1;
            if (m == 0) { if (cls == 0) ++c0; if (cls <= 1) ++c1; if (succ[b] && finite) ++c2; }
            // environment channel (M6a): counted on the M5 class, then applied per the selection policy
            // M6b: rank key = 3 * env_rank + self_rank (environment first, then self; each FREE < COLLIDING < INVALID,
            // 0 for a channel that is off or not ranking); GATE on either channel => class 2. Counts use the M5 class.
            int coll = 0;
            const int cls5 = cls;
            bool all_free = true;
            if (env_status) {
                const int st = env_status[b];
                all_free = all_free && st == ENV_FREE;
                if (m == 0) { if (st == ENV_FREE) ++c3; else if (st == ENV_COLLIDING) ++c4; else ++c5;
                              if (cls5 == 0 && st == ENV_FREE) ++c6; }
                if (env_sel == ENV_SELECT_GATE && st != ENV_FREE) { cls = 2; ephys = INFINITY; }
                else if (env_sel == ENV_SELECT_RANK) coll = 3 * st;
            }
            if (self_status) {
                const int st = self_status[b];
                all_free = all_free && st == ENV_FREE;
                if (m == 0) { if (st == ENV_FREE) ++c7; else if (st == ENV_COLLIDING) ++c8; else ++c9; }
                if (self_sel == ENV_SELECT_GATE && st != ENV_FREE) { cls = 2; ephys = INFINITY; }
                else if (self_sel == ENV_SELECT_RANK) coll += st;
            }
            if (m == 0 && cls5 == 0 && all_free) ++c10;
            if (taken) continue;
            const Cand cur{cls, coll, ephys, s};
            if (cand_better(cur, best)) best = cur;
        }
        s_best[tid] = best;
        if (m == 0) { s_cnt[NCNT*tid] = c0; s_cnt[NCNT*tid + 1] = c1; s_cnt[NCNT*tid + 2] = c2; s_cnt[NCNT*tid + 3] = c3;
                      s_cnt[NCNT*tid + 4] = c4; s_cnt[NCNT*tid + 5] = c5; s_cnt[NCNT*tid + 6] = c6; s_cnt[NCNT*tid + 7] = c7;
                      s_cnt[NCNT*tid + 8] = c8; s_cnt[NCNT*tid + 9] = c9; s_cnt[NCNT*tid + 10] = c10; }
        __syncthreads();
        for (int stride = nt >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                if (cand_better(s_best[tid + stride], s_best[tid])) s_best[tid] = s_best[tid + stride];
                if (m == 0) for (int i = 0; i < NCNT; ++i) s_cnt[NCNT*tid + i] += s_cnt[NCNT*(tid + stride) + i];
            }
            __syncthreads();
        }
        if (tid == 0) {
            const Cand win = s_best[0];
            const bool invalid = (win.cls == 2);
            if (m == 0) { o.num_solved[p] = s_cnt[0]; o.num_valid[p] = s_cnt[1]; o.num_converged[p] = s_cnt[2];
                          if (env_status) { if (eo.num_free) eo.num_free[p] = s_cnt[3]; if (eo.num_colliding) eo.num_colliding[p] = s_cnt[4];
                                            if (eo.num_invalid) eo.num_invalid[p] = s_cnt[5]; if (eo.num_solved_free) eo.num_solved_free[p] = s_cnt[6]; } }
            if (self_status) {
                if (m == 0) { if (xo.num_free) xo.num_free[p] = s_cnt[7]; if (xo.num_colliding) xo.num_colliding[p] = s_cnt[8];
                              if (xo.num_invalid) xo.num_invalid[p] = s_cnt[9]; if (xo.num_solved_free) xo.num_solved_free[p] = s_cnt[10]; }
                if (xo.sel_status) xo.sel_status[size_t(p) * M + m] = invalid ? (unsigned char)ENV_NOT_EVALUATED : self_status[size_t(p) * S + win.seed];
                if (xo.sel_min_clearance) xo.sel_min_clearance[size_t(p) * M + m] = invalid ? (T)NAN : self_mincl[size_t(p) * S + win.seed];
            }
            if (env_status) {   // the selected candidate's own verdict; a pad was never evaluated
                if (eo.sel_status) eo.sel_status[size_t(p) * M + m] = invalid ? (unsigned char)ENV_NOT_EVALUATED : env_status[size_t(p) * S + win.seed];
                if (eo.sel_min_clearance) eo.sel_min_clearance[size_t(p) * M + m] = invalid ? (T)NAN : env_mincl[size_t(p) * S + win.seed];
            }
            s_taken[m] = invalid ? -1 : win.seed;
            const size_t slot = size_t(p) * M + m, wb = size_t(p) * S + (invalid ? 0 : win.seed);
            if (!invalid) {
                for (int j = 0; j < N; ++j) o.q[slot * N + j] = q[wb * N + j];
                for (int k = 0; k < K; ++k) { o.pe[slot * K + k] = pe[wb * K + k]; o.oe[slot * K + k] = oe[wb * K + k]; }
                o.cost[slot] = cost[wb]; o.ephys[slot] = win.ephys; o.seed[slot] = win.seed;
                o.exact[slot] = (win.cls == 0) ? 1 : 0; o.valid[slot] = 1;
                if (base && o.base) for (int i = 0; i < 7; ++i) o.base[slot * 7 + i] = base[wb * 7 + i];   // same candidate
            } else {
                bool s0f = true;
                for (int j = 0; j < N; ++j) if (!isfinite((double)seeds[size_t(p) * S * N + j])) { s0f = false; break; }
                for (int j = 0; j < N; ++j) o.q[slot * N + j] = s0f ? seeds[size_t(p) * S * N + j] : (T)0;
                for (int k = 0; k < K; ++k) { o.pe[slot * K + k] = (T)INFINITY; o.oe[slot * K + k] = (T)INFINITY; }
                o.cost[slot] = (T)INFINITY; o.ephys[slot] = INFINITY; o.seed[slot] = -1;
                o.exact[slot] = 0; o.valid[slot] = 0;
                if (base_in && o.base) for (int i = 0; i < 7; ++i) o.base[slot * 7 + i] = base_in[size_t(p) * S * 7 + i];   // pad: first seed's input base
            }
        }
        __syncthreads();
    }
}

inline int seg_block_size(int S) { return S <= 64 ? 64 : (S <= 128 ? 128 : 256); }   // bbc53a9 policy

template<typename T>
size_t topM_shared(int nt, int M) { return sizeof(Cand) * nt + sizeof(int) * (11 * nt + M); }

template<typename T>
SelectOut<T> alloc_select(Device<T>& dev, int P, int M) {
    SelectOut<T> d{};
    const size_t PM = size_t(P) * M;
    d.q = dev.template alloc<T>(PM * N); d.pe = dev.template alloc<T>(PM * K); d.oe = dev.template alloc<T>(PM * K);
    d.cost = dev.template alloc<T>(PM); d.ephys = dev.template alloc<double>(PM); d.seed = dev.template alloc<int>(PM);
    d.exact = dev.template alloc<unsigned char>(PM); d.valid = dev.template alloc<unsigned char>(PM);
    d.num_solved = dev.template alloc<int>(P); d.num_valid = dev.template alloc<int>(P); d.num_converged = dev.template alloc<int>(P);
    d.base = dev.template alloc<T>(PM * 7);
    return d;
}
template<typename T>
void download_select(SelectOut<T>& h, const SelectOut<T>& d, int P, int M) {
    const size_t PM = size_t(P) * M;
    download(h.q, d.q, PM * N); download(h.pe, d.pe, PM * K); download(h.oe, d.oe, PM * K); download(h.cost, d.cost, PM);
    download(h.ephys, d.ephys, PM); download(h.seed, d.seed, PM); download(h.exact, d.exact, PM); download(h.valid, d.valid, PM);
    download(h.num_solved, d.num_solved, P); download(h.num_valid, d.num_valid, P); download(h.num_converged, d.num_converged, P);
    if (h.base) download(h.base, d.base, PM * 7);
}
template<typename T>
void launch_topM(const T* q, const T* pe, const T* oe, const T* cost, const unsigned char* succ, const T* seeds,
                 const unsigned int* active, int P, int S, int M, double eps_pos, double eps_ori, SelectOut<T>& d,
                 const T* base = nullptr, const T* base_in = nullptr, const unsigned char* env_status = nullptr,
                 const T* env_mincl = nullptr, int env_sel = ENV_SELECT_GATE, EnvOut<T> eo = EnvOut<T>{},
                 const unsigned char* self_status = nullptr, const T* self_mincl = nullptr, int self_sel = ENV_SELECT_ANNOTATE,
                 SelfOut<T> xo = SelfOut<T>{}) {
    const int nt = seg_block_size(S);
    tree_topM_kernel<T><<<P, nt, topM_shared<T>(nt, M)>>>(q, pe, oe, cost, succ, seeds, active, eps_pos, eps_ori, S, P, M, d,
                                                         base, base_in, env_status, env_mincl, env_sel, eo, self_status, self_mincl,
                                                         self_sel, xo);
    CUDA_OK(cudaGetLastError());
}

// ---- environment collision (M6a) ----
// One warp per candidate (EW per block, prologue shared), lane-strided over the robot spheres. FK is the LM's
// tree_fk_warp on the candidate's FINAL joints; the base is the candidate's FINAL base (re-normalised here only
// for the rotation, never written back). Primitives are GRiD's grid_collision (no HJCD re-implementation).
constexpr int ENV_WARPS = 4;
template<typename T>
struct EnvWarp { T x[N], local[16 * N], world[16 * N], R[9], t[3]; };
template<typename T>
__host__ __device__ constexpr size_t env_shared_bytes(int warps) {
    return ((prologue_bytes<T>() + 15) / 16) * 16 + size_t(warps) * ((sizeof(EnvWarp<T>) + 15) / 16) * 16;
}
template<typename T>
__global__ void tree_env_kernel(const T* __restrict__ q, const T* __restrict__ base, int B,
                                const grid::robotModel<T>* __restrict__ model, const T* __restrict__ rs, int NS,
                                grid_collision::Environment<T> env, T margin, unsigned char* __restrict__ status,
                                T* __restrict__ min_cl, int* __restrict__ argmin, T* __restrict__ sph_world,
                                T* __restrict__ sph_cl) {
    extern __shared__ __align__(16) unsigned char smem[];
    T* s_xhom; block_prologue<T>(smem, model, s_xhom);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    EnvWarp<T>* st = reinterpret_cast<EnvWarp<T>*>(smem + ((prologue_bytes<T>() + 15) / 16) * 16
                                                   + size_t(warp) * ((sizeof(EnvWarp<T>) + 15) / 16) * 16);
    const int b = blockIdx.x * warps + warp;
    if (b >= B) return;                                       // no block barrier after the prologue
    bool bad = false;
    if (lane < N) { const T v = q[size_t(b) * N + lane]; st->x[lane] = v; bad = !isfinite((double)v); }
    if (base && lane < 7) bad = bad || !isfinite((double)base[size_t(b) * 7 + lane]);
    bad = __any_sync(FULL_MASK, bad);
    if (bad) {                                                // not evaluated: never reported FREE
        if (lane == 0) { status[b] = ENV_INVALID; if (min_cl) min_cl[b] = (T)NAN; if (argmin) argmin[b] = -1; }
        for (int i = lane; i < NS; i += 32) {
            if (sph_cl) sph_cl[size_t(b) * NS + i] = (T)NAN;
            if (sph_world) for (int c = 0; c < 3; ++c) sph_world[(size_t(b) * NS + i) * 3 + c] = (T)NAN;
        }
        return;
    }
    if (lane == 0) {
        if (base) {
            T bq[4]; const T* bb = base + size_t(b) * 7;
            const T n = sqrt(bb[3] * bb[3] + bb[4] * bb[4] + bb[5] * bb[5] + bb[6] * bb[6]);
            for (int i = 0; i < 4; ++i) bq[i] = bb[3 + i] / n;
            quat_to_mat3<T>(bq, st->R);                       // column-major, the M4b base convention
            for (int i = 0; i < 3; ++i) st->t[i] = bb[i];
        } else {
            for (int i = 0; i < 9; ++i) st->R[i] = (i % 4 == 0) ? T(1) : T(0);
            st->t[0] = st->t[1] = st->t[2] = T(0);
        }
    }
    __syncwarp(FULL_MASK);
    tree_fk_warp<T>(s_xhom, st->x, st->local, st->world);
    __syncwarp(FULL_MASK);
    const int n_obs = grid_collision::grid_cc_num_obstacles<T>(env);
    T best = (T)INFINITY; int bi = 0x7fffffff; bool hit = false;
    for (int i = lane; i < NS; i += 32) {
        const T* row = rs + size_t(i) * 5;
        const int a = int(row[0]);
        const T ox = row[1], oy = row[2], oz = row[3], r = row[4];
        T cx = ox, cy = oy, cz = oz;
        if (a >= 0) {                                         // X column-major: X[4*col + row]
            const T* X = &st->world[16 * a];
            cx = X[0] * ox + X[4] * oy + X[8] * oz + X[12];
            cy = X[1] * ox + X[5] * oy + X[9] * oz + X[13];
            cz = X[2] * ox + X[6] * oy + X[10] * oz + X[14];
        }
        const T* R = st->R;
        const T wx = R[0] * cx + R[3] * cy + R[6] * cz + st->t[0];
        const T wy = R[1] * cx + R[4] * cy + R[7] * cz + st->t[1];
        const T wz = R[2] * cx + R[5] * cy + R[8] * cz + st->t[2];
        if (grid_collision::grid_cc_sphere_in_environment<T>(env, wx, wy, wz, r + margin)) hit = true;
        T d = (T)INFINITY;
        if (n_obs > 0) { T nx, ny, nz; d = grid_collision::grid_cc_nearest_obstacle<T>(env, wx, wy, wz, r, &nx, &ny, &nz); }
        if (d < best) { best = d; bi = i; }                   // strict: lowest sphere index within a lane
        if (sph_cl) sph_cl[size_t(b) * NS + i] = d;
        if (sph_world) { T* w = sph_world + (size_t(b) * NS + i) * 3; w[0] = wx; w[1] = wy; w[2] = wz; }
    }
    for (int off = 16; off > 0; off >>= 1) {                  // (min clearance, lowest sphere index)
        const T ob = __shfl_down_sync(FULL_MASK, best, off);
        const int oi = __shfl_down_sync(FULL_MASK, bi, off);
        if (ob < best || (ob == best && oi < bi)) { best = ob; bi = oi; }
    }
    hit = __any_sync(FULL_MASK, hit);
    if (lane == 0) {
        status[b] = hit ? ENV_COLLIDING : ENV_FREE;
        if (min_cl) min_cl[b] = best;                         // +inf for an empty environment
        if (argmin) argmin[b] = (n_obs > 0 && bi != 0x7fffffff) ? bi : -1;
    }
}

// Owned per-call device environment + robot table (freed with the Device arena / on scope exit).
template<typename T>
struct EnvDevice {
    grid_collision::DeviceEnvironment<T> env;
    const T* d_robot = nullptr;
    int NS = 0;
    float prep_ms = 0.f, upload_ms = 0.f;
};
template<typename T>
void prepare_env(Device<T>& dev, const EnvCollision<T>& ec, EnvDevice<T>& ed) {
    if (!ec.spheres_robot || ec.n_robot < 1) throw std::invalid_argument("environment collision needs a robot sphere table");
    if (!(ec.margin >= 0.0) || !std::isfinite(ec.margin)) throw std::invalid_argument("environment margin must be finite and >= 0");
    if (ec.selection < ENV_SELECT_GATE || ec.selection > ENV_SELECT_ANNOTATE) throw std::invalid_argument("bad environment selection");
    for (int i = 0; i < ec.n_robot; ++i) {
        const T* r = ec.spheres_robot + 5 * i;
        for (int c = 0; c < 5; ++c) if (!std::isfinite(double(r[c]))) throw std::invalid_argument("robot sphere table must be finite");
        if (!(r[4] > T(0)) || r[0] != std::floor(r[0]) || r[0] < T(-1) || r[0] >= T(N))
            throw std::invalid_argument("robot sphere rows need anchor in [-1, N) and radius > 0");
    }
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    auto host = grid_collision::make_host_environment<T>(ec.spheres, ec.n_spheres, ec.capsules, ec.n_capsules,
                                                         ec.cuboids, ec.n_cuboids, ec.planes, ec.n_planes);   // throws on bad rows
    const auto t1 = clk::now();
    ed.env = grid_collision::DeviceEnvironment<T>(host);
    ed.d_robot = dev.upload(ec.spheres_robot, size_t(ec.n_robot) * 5);
    ed.NS = ec.n_robot;
    CUDA_OK(cudaDeviceSynchronize());
    const auto t2 = clk::now();
    ed.prep_ms = float(std::chrono::duration<double, std::milli>(t1 - t0).count());
    ed.upload_ms = float(std::chrono::duration<double, std::milli>(t2 - t1).count());
}
template<typename T>
void launch_env(Device<T>& dev, const EnvDevice<T>& ed, const T* d_q, const T* d_base, int B, T margin,
                unsigned char* d_status, T* d_mincl, int* d_argmin, T* d_sw = nullptr, T* d_sc = nullptr) {
    const size_t bytes = env_shared_bytes<T>(ENV_WARPS);
    CUDA_OK(cudaFuncSetAttribute(tree_env_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)));
    tree_env_kernel<T><<<(B + ENV_WARPS - 1) / ENV_WARPS, 32 * ENV_WARPS, bytes>>>(
        d_q, d_base, B, dev.model.get(), ed.d_robot, ed.NS, ed.env.view(), margin, d_status, d_mincl, d_argmin, d_sw, d_sc);
    CUDA_OK(cudaGetLastError());
}

// ---- self-collision (M6b) ----
// One warp per candidate. Sphere centres in the BASE frame (the base never enters), then body pairs in order: a body
// pair is skipped iff its bounding-sphere lower bound LB > max(best, margin) + slack, which provably cannot change
// the verdict or the minimum; otherwise its nA x nB sphere pairs are strided over the lanes (GRiD primitives).
constexpr int SELF_WARPS = 2;
template<typename T>
struct SelfHead { T x[N], local[16 * N], world[16 * N]; };
__host__ __device__ constexpr size_t a16(size_t n) { return (n + 15) / 16 * 16; }
template<typename T>
__host__ __device__ constexpr size_t self_warp_bytes(int NS, int NB) {
    return a16(sizeof(SelfHead<T>)) + a16(sizeof(T) * 3 * size_t(NS)) + a16(sizeof(T) * 4 * size_t(NB));
}
template<typename T>
__host__ __device__ constexpr size_t self_shared_bytes(int warps, int NS, int NB) {
    return a16(prologue_bytes<T>()) + size_t(warps) * self_warp_bytes<T>(NS, NB);
}
template<typename T>
__device__ __forceinline__ bool pair_less(T d, int i, int j, T bd, int bi, int bj) {
    return d < bd || (d == bd && (i < bi || (i == bi && j < bj)));
}
template<typename T>
__global__ void tree_self_kernel(const T* __restrict__ q, int B, const grid::robotModel<T>* __restrict__ model,
                                 const T* __restrict__ rs, int NS, const int* __restrict__ perm, const int* __restrict__ bstart,
                                 const int* __restrict__ banchor, const T* __restrict__ bbound, int NB,
                                 const int* __restrict__ bpairs, int NP, T margin, unsigned char* __restrict__ status,
                                 T* __restrict__ min_cl, int* __restrict__ out_i, int* __restrict__ out_j,
                                 long long* __restrict__ n_eval) {
    extern __shared__ __align__(16) unsigned char smem[];
    T* s_xhom; block_prologue<T>(smem, model, s_xhom);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    unsigned char* ws = smem + a16(prologue_bytes<T>()) + size_t(warp) * self_warp_bytes<T>(NS, NB);
    SelfHead<T>* h = reinterpret_cast<SelfHead<T>*>(ws);
    T* sc = reinterpret_cast<T*>(ws + a16(sizeof(SelfHead<T>)));
    T* bc = reinterpret_cast<T*>(ws + a16(sizeof(SelfHead<T>)) + a16(sizeof(T) * 3 * size_t(NS)));
    const int b = blockIdx.x * warps + warp;
    if (b >= B) return;
    bool bad = false;
    if (lane < N) { const T v = q[size_t(b) * N + lane]; h->x[lane] = v; bad = !isfinite((double)v); }
    if (__any_sync(FULL_MASK, bad)) {
        if (lane == 0) { status[b] = ENV_INVALID; min_cl[b] = (T)NAN; out_i[b] = -1; out_j[b] = -1; if (n_eval) n_eval[b] = 0; }
        return;
    }
    __syncwarp(FULL_MASK);
    tree_fk_warp<T>(s_xhom, h->x, h->local, h->world);
    __syncwarp(FULL_MASK);
    auto place = [&](int a, T ox, T oy, T oz, T* o) {          // base-frame point of an anchor-frame offset
        if (a < 0) { o[0] = ox; o[1] = oy; o[2] = oz; return; }
        const T* X = &h->world[16 * a];
        o[0] = X[0] * ox + X[4] * oy + X[8] * oz + X[12];
        o[1] = X[1] * ox + X[5] * oy + X[9] * oz + X[13];
        o[2] = X[2] * ox + X[6] * oy + X[10] * oz + X[14];
    };
    for (int i = lane; i < NS; i += 32) { const T* r = rs + size_t(i) * 5; place(int(r[0]), r[1], r[2], r[3], &sc[3 * i]); }
    for (int k = lane; k < NB; k += 32) { place(banchor[k], bbound[4 * k], bbound[4 * k + 1], bbound[4 * k + 2], &bc[4 * k]); bc[4 * k + 3] = bbound[4 * k + 3]; }
    __syncwarp(FULL_MASK);
    const T slack = T(1e-4);
    T best = (T)INFINITY; int bi = 0x7fffffff, bj = 0x7fffffff; bool hit = false; long long nev = 0;
    for (int p = 0; p < NP; ++p) {
        const int A = bpairs[2 * p], C = bpairs[2 * p + 1];
        const T dx = bc[4 * A] - bc[4 * C], dy = bc[4 * A + 1] - bc[4 * C + 1], dz = bc[4 * A + 2] - bc[4 * C + 2];
        const T lb = sqrt(dx * dx + dy * dy + dz * dz) - bc[4 * A + 3] - bc[4 * C + 3];
        if (lb > (best > margin ? best : margin) + slack) continue;            // warp-uniform: best is uniform here
        const int sA = bstart[A], nA = bstart[A + 1] - sA, sB = bstart[C], nB = bstart[C + 1] - sB, tot = nA * nB;
        nev += tot;
        for (int k = lane; k < tot; k += 32) {
            const int ia = perm[sA + k / nB], ib = perm[sB + k % nB];
            const int i = ia < ib ? ia : ib, j = ia < ib ? ib : ia;
            const T *ci = &sc[3 * i], *cj = &sc[3 * j];
            const T ri = rs[size_t(i) * 5 + 4], rj = rs[size_t(j) * 5 + 4];
            if (grid_collision::grid_cc_sphere_sphere<T>(ci[0], ci[1], ci[2], ri + margin, cj[0], cj[1], cj[2], rj) < T(0)) hit = true;
            T nx, ny, nz;
            const T d = grid_collision::grid_cc_sphere_sphere_signed<T>(ci[0], ci[1], ci[2], ri, cj[0], cj[1], cj[2], rj, &nx, &ny, &nz);
            if (pair_less<T>(d, i, j, best, bi, bj)) { best = d; bi = i; bj = j; }
        }
        for (int off = 16; off > 0; off >>= 1) {                               // all-reduce: best uniform again
            const T od = __shfl_xor_sync(FULL_MASK, best, off);
            const int oi = __shfl_xor_sync(FULL_MASK, bi, off), oj = __shfl_xor_sync(FULL_MASK, bj, off);
            if (pair_less<T>(od, oi, oj, best, bi, bj)) { best = od; bi = oi; bj = oj; }
        }
    }
    hit = __any_sync(FULL_MASK, hit);
    if (lane == 0) {
        status[b] = hit ? ENV_COLLIDING : ENV_FREE; min_cl[b] = best;
        out_i[b] = bi == 0x7fffffff ? -1 : bi; out_j[b] = bj == 0x7fffffff ? -1 : bj;
        if (n_eval) n_eval[b] = nev;
    }
}

template<typename T>
struct SelfDevice {
    const T* rs = nullptr; const int *perm = nullptr, *bstart = nullptr, *banchor = nullptr, *bpairs = nullptr;
    const T* bbound = nullptr;
    int NS = 0, NB = 0, NP = 0; long long eligible = 0;
    float prep_ms = 0.f, upload_ms = 0.f;
};
// Host: group spheres by anchor (body), bounding spheres in the anchor frame, body-pair list. Validates the table.
template<typename T>
void prepare_self(Device<T>& dev, const SelfCollision<T>& sc, SelfDevice<T>& sd) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    if (!sc.spheres_robot || sc.n_robot < 1) throw std::invalid_argument("self-collision needs a robot sphere table");
    if (!(sc.margin >= 0.0) || !std::isfinite(sc.margin)) throw std::invalid_argument("self-collision margin must be finite and >= 0");
    if (sc.selection < ENV_SELECT_GATE || sc.selection > ENV_SELECT_ANNOTATE) throw std::invalid_argument("bad self-collision selection");
    if (sc.n_pairs < 0 || (sc.n_pairs > 0 && !sc.pairs)) throw std::invalid_argument("self-collision pairs missing");
    const int NS = sc.n_robot;
    std::vector<int> anc(NS);
    for (int i = 0; i < NS; ++i) {
        const T* r = sc.spheres_robot + 5 * i;
        for (int c = 0; c < 5; ++c) if (!std::isfinite(double(r[c]))) throw std::invalid_argument("robot sphere table must be finite");
        if (!(r[4] > T(0)) || r[0] != std::floor(r[0]) || r[0] < T(-1) || r[0] >= T(N))
            throw std::invalid_argument("robot sphere rows need anchor in [-1, N) and radius > 0");
        anc[i] = int(r[0]);
    }
    std::vector<int> body_of_anchor(N + 1, -1), banchor, bstart, perm;
    for (int a = -1; a < N; ++a) {
        bool any = false;
        for (int i = 0; i < NS; ++i) if (anc[i] == a) { if (!any) { body_of_anchor[a + 1] = int(banchor.size()); banchor.push_back(a); bstart.push_back(int(perm.size())); any = true; } perm.push_back(i); }
    }
    const int NB = int(banchor.size()); bstart.push_back(NS);
    std::vector<T> bbound(4 * size_t(NB));
    for (int k = 0; k < NB; ++k) {                                   // centre = mean offset, radius covers every sphere
        double c[3] = {0, 0, 0}; const int n = bstart[k + 1] - bstart[k];
        for (int m = bstart[k]; m < bstart[k + 1]; ++m) for (int d = 0; d < 3; ++d) c[d] += double(sc.spheres_robot[5 * perm[m] + 1 + d]) / n;
        double R = 0;
        for (int m = bstart[k]; m < bstart[k + 1]; ++m) {
            const T* r = sc.spheres_robot + 5 * perm[m];
            const double e = std::sqrt((r[1] - c[0]) * (r[1] - c[0]) + (r[2] - c[1]) * (r[2] - c[1]) + (r[3] - c[2]) * (r[3] - c[2])) + double(r[4]);
            R = e > R ? e : R;
        }
        for (int d = 0; d < 3; ++d) bbound[4 * k + d] = T(c[d]);
        bbound[4 * k + 3] = T(R * (1 + 1e-6) + 1e-6);
    }
    std::vector<int> bp; long long elig = 0;
    for (int p = 0; p < sc.n_pairs; ++p) {
        const int a = sc.pairs[2 * p], b = sc.pairs[2 * p + 1];
        if (a < -1 || a >= N || b < -1 || b >= N || a == b) throw std::invalid_argument("self-collision pairs need distinct anchors in [-1, N)");
        const int A = body_of_anchor[a + 1], C = body_of_anchor[b + 1];
        if (A < 0 || C < 0) continue;                                // a body with no spheres: nothing to check
        bp.push_back(A); bp.push_back(C);
        elig += (long long)(bstart[A + 1] - bstart[A]) * (bstart[C + 1] - bstart[C]);
    }
    const auto t1 = clk::now();
    sd.rs = dev.upload(sc.spheres_robot, size_t(NS) * 5); sd.perm = dev.upload(perm.data(), perm.size());
    sd.bstart = dev.upload(bstart.data(), bstart.size()); sd.banchor = dev.upload(banchor.data(), banchor.size());
    sd.bbound = dev.upload(bbound.data(), bbound.size());
    sd.bpairs = bp.empty() ? dev.template alloc<int>(2) : dev.upload(bp.data(), bp.size());
    sd.NS = NS; sd.NB = NB; sd.NP = int(bp.size() / 2); sd.eligible = elig;
    CUDA_OK(cudaDeviceSynchronize());
    const auto t2 = clk::now();
    sd.prep_ms = float(std::chrono::duration<double, std::milli>(t1 - t0).count());
    sd.upload_ms = float(std::chrono::duration<double, std::milli>(t2 - t1).count());
}
template<typename T>
void launch_self(Device<T>& dev, const SelfDevice<T>& sd, const T* d_q, int B, T margin, unsigned char* st, T* mc,
                 int* pi, int* pj, long long* ne) {
    const size_t bytes = self_shared_bytes<T>(SELF_WARPS, sd.NS, sd.NB);
    CUDA_OK(cudaFuncSetAttribute(tree_self_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)));
    tree_self_kernel<T><<<(B + SELF_WARPS - 1) / SELF_WARPS, 32 * SELF_WARPS, bytes>>>(
        d_q, B, dev.model.get(), sd.rs, sd.NS, sd.perm, sd.bstart, sd.banchor, sd.bbound, sd.NB, sd.bpairs, sd.NP, margin,
        st, mc, pi, pj, ne);
    CUDA_OK(cudaGetLastError());
}

}  // namespace

template<typename T>
void evaluate(const T* q, int B, const ProblemIn<T>& in, T lambda, EvalOut<T>& out) {
    if (B < 1) throw std::invalid_argument("batch must be >= 1");
    Device<T> dev;
    const T* d_tool = nullptr;
    ProblemIn<T> din = upload_problem<T>(dev, B, in, d_tool);
    const T* d_q = dev.upload(q, size_t(B) * N);
    if (in.base) din.base = dev.upload(in.base, size_t(B) * 7);
    EvalOut<T> d{};
    d.pose = dev.template alloc<T>(size_t(B) * K * 16); d.e = dev.template alloc<T>(size_t(B) * K * 6);
    d.pn = dev.template alloc<T>(size_t(B) * K); d.on = dev.template alloc<T>(size_t(B) * K);
    d.J = dev.template alloc<T>(size_t(B) * K * 6 * N); d.scale = dev.template alloc<T>(size_t(B) * K * 6);
    d.A_raw = dev.template alloc<T>(size_t(B) * N * N); d.b_raw = dev.template alloc<T>(size_t(B) * N);
    d.A = dev.template alloc<T>(size_t(B) * N * N); d.b = dev.template alloc<T>(size_t(B) * N);
    d.colnorm2 = dev.template alloc<T>(size_t(B) * N); d.Ap = dev.template alloc<T>(size_t(B) * N * N);
    d.bp = dev.template alloc<T>(size_t(B) * N); d.dq = dev.template alloc<T>(size_t(B) * N);
    d.pinned = dev.template alloc<unsigned int>(size_t(B)); d.fail = dev.template alloc<int>(size_t(B));
    const int warps = 1;
    const size_t bytes = shared_bytes<T>(warps);
    CUDA_OK(cudaFuncSetAttribute(tree_eval_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)));
    tree_eval_kernel<T><<<B, 32 * warps, bytes>>>(d_q, B, din, d_tool, dev.model.get(), lambda, d);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    download(out.pose, d.pose, size_t(B) * K * 16); download(out.e, d.e, size_t(B) * K * 6);
    download(out.pn, d.pn, size_t(B) * K); download(out.on, d.on, size_t(B) * K);
    download(out.J, d.J, size_t(B) * K * 6 * N); download(out.scale, d.scale, size_t(B) * K * 6);
    download(out.A_raw, d.A_raw, size_t(B) * N * N); download(out.b_raw, d.b_raw, size_t(B) * N);
    download(out.A, d.A, size_t(B) * N * N); download(out.b, d.b, size_t(B) * N);
    download(out.colnorm2, d.colnorm2, size_t(B) * N); download(out.Ap, d.Ap, size_t(B) * N * N);
    download(out.bp, d.bp, size_t(B) * N); download(out.dq, d.dq, size_t(B) * N);
    download(out.pinned, d.pinned, size_t(B)); download(out.fail, d.fail, size_t(B));
}

template<typename T>
void lm(const T* q0, int B, const ProblemIn<T>& in, const LMConfig& cfg, LMOut<T>& out, float* kernel_ms) {
    if (B < 1) throw std::invalid_argument("batch must be >= 1");
    if (cfg.k_max < 0 || cfg.warps_per_block < 1 || cfg.warps_per_block > 32) throw std::invalid_argument("bad LM config");
    Device<T> dev;
    const T* d_tool = nullptr;
    ProblemIn<T> din = upload_problem<T>(dev, B, in, d_tool);
    const T* d_q = dev.upload(q0, size_t(B) * N);
    LMOut<T> d{};
    d.x = dev.template alloc<T>(size_t(B) * N);
    d.pn0 = dev.template alloc<T>(size_t(B) * K); d.on0 = dev.template alloc<T>(size_t(B) * K);
    d.pn = dev.template alloc<T>(size_t(B) * K); d.on = dev.template alloc<T>(size_t(B) * K);
    d.cost = dev.template alloc<T>(B); d.ephys0 = dev.template alloc<T>(B); d.ephys = dev.template alloc<T>(B);
    d.success = dev.template alloc<unsigned char>(B); d.iters = dev.template alloc<int>(B);
    d.accepted = dev.template alloc<int>(B); d.solve_fail = dev.template alloc<int>(B); d.kicks = dev.template alloc<int>(B);
    const int warps = cfg.warps_per_block;
    const size_t bytes = shared_bytes<T>(warps);
    CUDA_OK(cudaFuncSetAttribute(tree_lm_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)));
    const dim3 grid_dim((B + warps - 1) / warps), block_dim(32 * warps);
    tree_lm_kernel<T><<<grid_dim, block_dim, bytes>>>(d_q, B, din, d_tool, dev.model.get(), cfg, d);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    if (cfg.repeats > 0 && kernel_ms) {
        cudaEvent_t s, e; CUDA_OK(cudaEventCreate(&s)); CUDA_OK(cudaEventCreate(&e));
        CUDA_OK(cudaEventRecord(s));
        for (int i = 0; i < cfg.repeats; ++i) tree_lm_kernel<T><<<grid_dim, block_dim, bytes>>>(d_q, B, din, d_tool, dev.model.get(), cfg, d);
        CUDA_OK(cudaEventRecord(e)); CUDA_OK(cudaEventSynchronize(e));
        float ms = 0.f; CUDA_OK(cudaEventElapsedTime(&ms, s, e));
        (void)cudaEventDestroy(s); (void)cudaEventDestroy(e);
        *kernel_ms = ms / float(cfg.repeats);
    }
    download(out.x, d.x, size_t(B) * N); download(out.pn0, d.pn0, size_t(B) * K); download(out.on0, d.on0, size_t(B) * K);
    download(out.pn, d.pn, size_t(B) * K); download(out.on, d.on, size_t(B) * K);
    download(out.cost, d.cost, B); download(out.ephys0, d.ephys0, B); download(out.ephys, d.ephys, B);
    download(out.success, d.success, B); download(out.iters, d.iters, B); download(out.accepted, d.accepted, B);
    download(out.solve_fail, d.solve_fail, B); download(out.kicks, d.kicks, B);
}

template<typename T>
void solve_batch(const T* seeds, int P, int S, int M, const ProblemIn<T>& in, const LMConfig& cfg,
                 LMOut<T>* raw, SelectOut<T>& sel, BatchTiming* timing, const CoarseConfig* coarse, CoarseIO<T>* cio,
                 const EnvCollision<T>* env, EnvOut<T>* eout, const SelfCollision<T>* self, SelfOut<T>* sout) {
    if (P < 1 || S < 1 || M < 1 || M > S) throw std::invalid_argument("need P >= 1, S >= 1, 1 <= M <= S");
    if (cfg.k_max < 0 || cfg.warps_per_block < 1 || cfg.warps_per_block > 32) throw std::invalid_argument("bad LM config");
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    Device<T> dev;
    const auto t1 = clk::now();
    ProblemIn<T> hin = in; hin.seeds_per_problem = S;
    const T* d_tool = nullptr;
    ProblemIn<T> din = upload_problem<T>(dev, P, hin, d_tool);
    const int B = P * S;
    const T* d_seeds = dev.upload(seeds, size_t(B) * N);
    if (in.base) din.base = dev.upload(in.base, size_t(B) * 7);
    LMOut<T> d{};
    d.x = dev.template alloc<T>(size_t(B) * N);
    d.pn0 = dev.template alloc<T>(size_t(B) * K); d.on0 = dev.template alloc<T>(size_t(B) * K);
    d.pn = dev.template alloc<T>(size_t(B) * K); d.on = dev.template alloc<T>(size_t(B) * K);
    d.cost = dev.template alloc<T>(B); d.ephys0 = dev.template alloc<T>(B); d.ephys = dev.template alloc<T>(B);
    d.success = dev.template alloc<unsigned char>(B); d.iters = dev.template alloc<int>(B);
    d.accepted = dev.template alloc<int>(B); d.solve_fail = dev.template alloc<int>(B); d.kicks = dev.template alloc<int>(B);
    if (in.base) { d.base = dev.template alloc<T>(size_t(B) * 7); d.base_diag = dev.template alloc<int>(size_t(B) * 4); }
    SelectOut<T> ds = alloc_select<T>(dev, P, M);
    if (!in.base) ds.base = nullptr;
    // environment collision (M6a): requested => prepared or the call fails (never silently skipped)
    EnvDevice<T> ed;
    unsigned char* d_est = nullptr; T* d_emin = nullptr; int* d_earg = nullptr;
    EnvOut<T> deo{};
    if (env) {
        prepare_env<T>(dev, *env, ed);
        d_est = dev.template alloc<unsigned char>(B); d_emin = dev.template alloc<T>(B); d_earg = dev.template alloc<int>(B);
        deo.num_free = dev.template alloc<int>(P); deo.num_colliding = dev.template alloc<int>(P);
        deo.num_invalid = dev.template alloc<int>(P); deo.num_solved_free = dev.template alloc<int>(P);
        deo.sel_status = dev.template alloc<unsigned char>(size_t(P) * M); deo.sel_min_clearance = dev.template alloc<T>(size_t(P) * M);
    }
    // self-collision (M6b): independent channel, same contract (requested => prepared or the call fails)
    SelfDevice<T> sdv;
    unsigned char* d_sst = nullptr; T* d_smin = nullptr; int *d_si = nullptr, *d_sj = nullptr; long long* d_sne = nullptr;
    SelfOut<T> dxo{};
    if (self) {
        prepare_self<T>(dev, *self, sdv);
        d_sst = dev.template alloc<unsigned char>(B); d_smin = dev.template alloc<T>(B); d_si = dev.template alloc<int>(B);
        d_sj = dev.template alloc<int>(B); d_sne = dev.template alloc<long long>(B);
        dxo.num_free = dev.template alloc<int>(P); dxo.num_colliding = dev.template alloc<int>(P);
        dxo.num_invalid = dev.template alloc<int>(P); dxo.num_solved_free = dev.template alloc<int>(P);
        dxo.sel_status = dev.template alloc<unsigned char>(size_t(P) * M); dxo.sel_min_clearance = dev.template alloc<T>(size_t(P) * M);
    }
    CUDA_OK(cudaDeviceSynchronize());
    const auto t2 = clk::now();
    cudaEvent_t e0, e1, e2;
    CUDA_OK(cudaEventCreate(&e0)); CUDA_OK(cudaEventCreate(&e1)); CUDA_OK(cudaEventCreate(&e2));
    const int warps = cfg.warps_per_block;
    const size_t bytes = shared_bytes<T>(warps);
    CUDA_OK(cudaFuncSetAttribute(tree_lm_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)));
    // coarse stage (M5): seeds -> d_xc (in place on a copy); candidates with use_coarse == 0 keep their seed
    const T* lm_seeds = d_seeds;
    float coarse_ms = 0.f;
    CoarseIO<T> dio{};
    if (coarse && cio) {
        T* d_xc = dev.template alloc<T>(size_t(B) * N);
        CUDA_OK(cudaMemcpy(d_xc, d_seeds, sizeof(T) * size_t(B) * N, cudaMemcpyDeviceToDevice));
        if (cio->use_coarse) dio.use_coarse = dev.upload(cio->use_coarse, size_t(B));
        if (cio->problem_seeds) dio.problem_seeds = dev.upload(cio->problem_seeds, size_t(P));
        if (cio->candidate_ids) dio.candidate_ids = dev.upload(cio->candidate_ids, size_t(B));
        dio.cost0 = dev.template alloc<T>(B); dio.cost1 = dev.template alloc<T>(B); dio.pn = dev.template alloc<T>(size_t(B) * K);
        dio.on = dev.template alloc<T>(size_t(B) * K); dio.iters = dev.template alloc<int>(B); dio.kicks = dev.template alloc<int>(B);
        dio.success = dev.template alloc<unsigned char>(B);
        CUDA_OK(cudaMemset(dio.iters, 0, sizeof(int) * B)); CUDA_OK(cudaMemset(dio.kicks, 0, sizeof(int) * B));
        CUDA_OK(cudaMemset(dio.success, 0, B)); CUDA_OK(cudaMemset(dio.cost0, 0, sizeof(T) * B)); CUDA_OK(cudaMemset(dio.cost1, 0, sizeof(T) * B));
        CUDA_OK(cudaMemset(dio.pn, 0, sizeof(T) * B * K)); CUDA_OK(cudaMemset(dio.on, 0, sizeof(T) * B * K));
        const int cw = 1;                                               // bbc53a9: one warp per block
        const size_t cbytes = coarse_shared_bytes<T>(cw);
        CUDA_OK(cudaFuncSetAttribute(tree_coarse_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(cbytes)));
        cudaEvent_t c0, c1; CUDA_OK(cudaEventCreate(&c0)); CUDA_OK(cudaEventCreate(&c1));
        CUDA_OK(cudaEventRecord(c0));
        tree_coarse_kernel<T><<<B, 32 * cw, cbytes>>>(d_xc, B, din, d_tool, dev.model.get(), cfg, *coarse, dio);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaEventRecord(c1)); CUDA_OK(cudaEventSynchronize(c1));
        CUDA_OK(cudaEventElapsedTime(&coarse_ms, c0, c1));
        (void)cudaEventDestroy(c0); (void)cudaEventDestroy(c1);
        dio.x = d_xc; lm_seeds = d_xc;
    }
    CUDA_OK(cudaEventRecord(e0));
    tree_lm_kernel<T><<<(B + warps - 1) / warps, 32 * warps, bytes>>>(lm_seeds, B, din, d_tool, dev.model.get(), cfg, d);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaEventRecord(e1));
    cudaEvent_t e1b; CUDA_OK(cudaEventCreate(&e1b));
    // post-LM: the FINAL joints d.x paired with the same candidate's FINAL base d.base (null => fixed base)
    if (env) launch_env<T>(dev, ed, d.x, in.base ? d.base : nullptr, B, (T)env->margin, d_est, d_emin, d_earg);
    cudaEvent_t e1c; CUDA_OK(cudaEventCreate(&e1c));
    CUDA_OK(cudaEventRecord(e1c));
    // self-collision on the same FINAL joints (base-frame: the base cannot affect it)
    if (self) launch_self<T>(dev, sdv, d.x, B, (T)self->margin, d_sst, d_smin, d_si, d_sj, d_sne);
    CUDA_OK(cudaEventRecord(e1b));
    launch_topM<T>(d.x, d.pn, d.on, d.cost, d.success, d_seeds, din.active, P, S, M, cfg.eps_pos, cfg.eps_ori, ds,
                   d.base, din.base, d_est, d_emin, env ? env->selection : ENV_SELECT_GATE, deo,
                   d_sst, d_smin, self ? self->selection : ENV_SELECT_ANNOTATE, dxo);
    CUDA_OK(cudaEventRecord(e2));
    CUDA_OK(cudaEventSynchronize(e2));
    float lm_ms = 0, sel_ms = 0, env_ms = 0;
    float self_ms = 0;
    CUDA_OK(cudaEventElapsedTime(&lm_ms, e0, e1)); CUDA_OK(cudaEventElapsedTime(&env_ms, e1, e1c));
    CUDA_OK(cudaEventElapsedTime(&self_ms, e1c, e1b)); CUDA_OK(cudaEventElapsedTime(&sel_ms, e1b, e2));
    (void)cudaEventDestroy(e0); (void)cudaEventDestroy(e1); (void)cudaEventDestroy(e1b); (void)cudaEventDestroy(e1c); (void)cudaEventDestroy(e2);
    if (self && sout) {
        sout->prep_ms = sdv.prep_ms; sout->upload_ms = sdv.upload_ms; sout->kernel_ms = self_ms;
        sout->eligible_pairs = sdv.eligible; sout->n_bodies = sdv.NB; sout->n_body_pairs = sdv.NP;
        if (sout->status) download(sout->status, d_sst, size_t(B));
        if (sout->min_clearance) download(sout->min_clearance, d_smin, size_t(B));
        if (sout->pair_i) download(sout->pair_i, d_si, size_t(B));
        if (sout->pair_j) download(sout->pair_j, d_sj, size_t(B));
        if (sout->evaluated_pairs) download(sout->evaluated_pairs, d_sne, size_t(B));
        if (sout->num_free) download(sout->num_free, dxo.num_free, P);
        if (sout->num_colliding) download(sout->num_colliding, dxo.num_colliding, P);
        if (sout->num_invalid) download(sout->num_invalid, dxo.num_invalid, P);
        if (sout->num_solved_free) download(sout->num_solved_free, dxo.num_solved_free, P);
        if (sout->sel_status) download(sout->sel_status, dxo.sel_status, size_t(P) * M);
        if (sout->sel_min_clearance) download(sout->sel_min_clearance, dxo.sel_min_clearance, size_t(P) * M);
    }
    if (env && eout) {
        eout->prep_ms = ed.prep_ms; eout->upload_ms = ed.upload_ms; eout->kernel_ms = env_ms;
        if (eout->status) download(eout->status, d_est, size_t(B));
        if (eout->min_clearance) download(eout->min_clearance, d_emin, size_t(B));
        if (eout->argmin_sphere) download(eout->argmin_sphere, d_earg, size_t(B));
        if (eout->num_free) download(eout->num_free, deo.num_free, P);
        if (eout->num_colliding) download(eout->num_colliding, deo.num_colliding, P);
        if (eout->num_invalid) download(eout->num_invalid, deo.num_invalid, P);
        if (eout->num_solved_free) download(eout->num_solved_free, deo.num_solved_free, P);
        if (eout->sel_status) download(eout->sel_status, deo.sel_status, size_t(P) * M);
        if (eout->sel_min_clearance) download(eout->sel_min_clearance, deo.sel_min_clearance, size_t(P) * M);
    }
    const auto t3 = clk::now();
    if (!in.base) sel.base = nullptr;
    download_select<T>(sel, ds, P, M);
    if (raw) {
        if (in.base && raw->base) download(raw->base, d.base, size_t(B) * 7);
        if (in.base && raw->base_diag) download(raw->base_diag, d.base_diag, size_t(B) * 4);
        download(raw->x, d.x, size_t(B) * N); download(raw->pn0, d.pn0, size_t(B) * K); download(raw->on0, d.on0, size_t(B) * K);
        download(raw->pn, d.pn, size_t(B) * K); download(raw->on, d.on, size_t(B) * K); download(raw->cost, d.cost, B);
        download(raw->ephys0, d.ephys0, B); download(raw->ephys, d.ephys, B); download(raw->success, d.success, B);
        download(raw->iters, d.iters, B); download(raw->accepted, d.accepted, B);
        download(raw->solve_fail, d.solve_fail, B); download(raw->kicks, d.kicks, B);
    }
    const auto t4 = clk::now();
    if (coarse && cio) {
        if (cio->x) download(cio->x, dio.x, size_t(B) * N);
        if (cio->cost0) download(cio->cost0, dio.cost0, B); if (cio->cost1) download(cio->cost1, dio.cost1, B);
        if (cio->pn) download(cio->pn, dio.pn, size_t(B) * K); if (cio->on) download(cio->on, dio.on, size_t(B) * K);
        if (cio->iters) download(cio->iters, dio.iters, B); if (cio->kicks) download(cio->kicks, dio.kicks, B);
        if (cio->success) download(cio->success, dio.success, B);
    }
    const auto t5 = clk::now();
    if (timing) {
        auto ms = [](clk::time_point a, clk::time_point b) { return float(std::chrono::duration<double, std::milli>(b - a).count()); };
        *timing = BatchTiming{ms(t1, t2), lm_ms, sel_ms, ms(t3, t5), ms(t0, t1), ms(t0, t5), coarse_ms};
    }
}

template<typename T>
void select_only(const T* q, const T* pe, const T* oe, const T* cost, const unsigned char* succ, const T* seeds,
                 const unsigned int* active, int P, int S, int M, double eps_pos, double eps_ori, SelectOut<T>& sel,
                 const unsigned char* env_status, const T* env_min_clearance, int env_selection, EnvOut<T>* eout,
                 const unsigned char* self_status, int self_selection, SelfOut<T>* sout) {
    if (self_selection < ENV_SELECT_GATE || self_selection > ENV_SELECT_ANNOTATE) throw std::invalid_argument("bad self-collision selection");
    if (P < 1 || S < 1 || M < 1 || M > S) throw std::invalid_argument("need P >= 1, S >= 1, 1 <= M <= S");
    if (env_selection < ENV_SELECT_GATE || env_selection > ENV_SELECT_ANNOTATE) throw std::invalid_argument("bad environment selection");
    Device<T> dev;
    const size_t B = size_t(P) * S;
    SelectOut<T> ds = alloc_select<T>(dev, P, M);
    EnvOut<T> deo{};
    const unsigned char* d_est = nullptr; const T* d_emin = nullptr;
    if (env_status) {
        d_est = dev.upload(env_status, B);
        d_emin = env_min_clearance ? dev.upload(env_min_clearance, B) : dev.template alloc<T>(B);
        deo.num_free = dev.template alloc<int>(P); deo.num_colliding = dev.template alloc<int>(P);
        deo.num_invalid = dev.template alloc<int>(P); deo.num_solved_free = dev.template alloc<int>(P);
        deo.sel_status = dev.template alloc<unsigned char>(size_t(P) * M); deo.sel_min_clearance = dev.template alloc<T>(size_t(P) * M);
    }
    SelfOut<T> dxo{};
    const unsigned char* d_xst = nullptr; const T* d_xmin = nullptr;
    if (self_status) {
        d_xst = dev.upload(self_status, B); d_xmin = dev.template alloc<T>(B);
        dxo.num_free = dev.template alloc<int>(P); dxo.num_colliding = dev.template alloc<int>(P);
        dxo.num_invalid = dev.template alloc<int>(P); dxo.num_solved_free = dev.template alloc<int>(P);
        dxo.sel_status = dev.template alloc<unsigned char>(size_t(P) * M); dxo.sel_min_clearance = dev.template alloc<T>(size_t(P) * M);
    }
    launch_topM<T>(dev.upload(q, B * N), dev.upload(pe, B * K), dev.upload(oe, B * K), dev.upload(cost, B),
                   dev.upload(succ, B), dev.upload(seeds, B * N), dev.upload(active, size_t(P)), P, S, M, eps_pos, eps_ori, ds,
                   nullptr, nullptr, d_est, d_emin, env_selection, deo, d_xst, d_xmin, self_selection, dxo);
    CUDA_OK(cudaDeviceSynchronize());
    download_select<T>(sel, ds, P, M);
    if (self_status && sout) {
        if (sout->num_free) download(sout->num_free, dxo.num_free, P);
        if (sout->num_colliding) download(sout->num_colliding, dxo.num_colliding, P);
        if (sout->num_invalid) download(sout->num_invalid, dxo.num_invalid, P);
        if (sout->num_solved_free) download(sout->num_solved_free, dxo.num_solved_free, P);
        if (sout->sel_status) download(sout->sel_status, dxo.sel_status, size_t(P) * M);
    }
    if (env_status && eout) {
        if (eout->num_free) download(eout->num_free, deo.num_free, P);
        if (eout->num_colliding) download(eout->num_colliding, deo.num_colliding, P);
        if (eout->num_invalid) download(eout->num_invalid, deo.num_invalid, P);
        if (eout->num_solved_free) download(eout->num_solved_free, deo.num_solved_free, P);
        if (eout->sel_status) download(eout->sel_status, deo.sel_status, size_t(P) * M);
    }
}

template<typename T>
void proposals(const T* q, int B, const ProblemIn<T>& in, const CoarseConfig& cc, T* out) {
    if (B < 1) throw std::invalid_argument("batch must be >= 1");
    Device<T> dev;
    const T* d_tool = nullptr;
    ProblemIn<T> din = upload_problem<T>(dev, B, in, d_tool);
    if (in.base) din.base = dev.upload(in.base, size_t(B) * 7);
    const T* d_q = dev.upload(q, size_t(B) * N);
    T* d_out = dev.template alloc<T>(size_t(B) * N * 6);
    const size_t bytes = coarse_shared_bytes<T>(1);
    CUDA_OK(cudaFuncSetAttribute(tree_proposal_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)));
    tree_proposal_kernel<T><<<B, 32, bytes>>>(d_q, B, din, d_tool, dev.model.get(), cc, d_out);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    download(out, d_out, size_t(B) * N * 6);
}

template<typename T>
void env_collision(const T* q, const T* base, int B, const EnvCollision<T>& ec, EnvOut<T>& out) {
    if (B < 1) throw std::invalid_argument("batch must be >= 1");
    Device<T> dev;
    EnvDevice<T> ed; prepare_env<T>(dev, ec, ed);
    const T* d_q = dev.upload(q, size_t(B) * N);
    const T* d_b = base ? dev.upload(base, size_t(B) * 7) : nullptr;
    unsigned char* d_st = dev.template alloc<unsigned char>(B); T* d_mc = dev.template alloc<T>(B); int* d_am = dev.template alloc<int>(B);
    T* d_sw = out.sphere_world ? dev.template alloc<T>(size_t(B) * ed.NS * 3) : nullptr;
    T* d_sc = out.sphere_clearance ? dev.template alloc<T>(size_t(B) * ed.NS) : nullptr;
    cudaEvent_t a, z; CUDA_OK(cudaEventCreate(&a)); CUDA_OK(cudaEventCreate(&z));
    CUDA_OK(cudaEventRecord(a));
    launch_env<T>(dev, ed, d_q, d_b, B, (T)ec.margin, d_st, d_mc, d_am, d_sw, d_sc);
    CUDA_OK(cudaEventRecord(z)); CUDA_OK(cudaEventSynchronize(z));
    CUDA_OK(cudaEventElapsedTime(&out.kernel_ms, a, z));
    (void)cudaEventDestroy(a); (void)cudaEventDestroy(z);
    out.prep_ms = ed.prep_ms; out.upload_ms = ed.upload_ms;
    if (out.status) download(out.status, d_st, size_t(B));
    if (out.min_clearance) download(out.min_clearance, d_mc, size_t(B));
    if (out.argmin_sphere) download(out.argmin_sphere, d_am, size_t(B));
    if (d_sw) download(out.sphere_world, d_sw, size_t(B) * ed.NS * 3);
    if (d_sc) download(out.sphere_clearance, d_sc, size_t(B) * ed.NS);
}

template<typename T>
void self_collision(const T* q, int B, const SelfCollision<T>& sc, SelfOut<T>& out) {
    if (B < 1) throw std::invalid_argument("batch must be >= 1");
    Device<T> dev;
    SelfDevice<T> sd; prepare_self<T>(dev, sc, sd);
    const T* d_q = dev.upload(q, size_t(B) * N);
    unsigned char* st = dev.template alloc<unsigned char>(B); T* mc = dev.template alloc<T>(B);
    int* pi = dev.template alloc<int>(B); int* pj = dev.template alloc<int>(B); long long* ne = dev.template alloc<long long>(B);
    cudaEvent_t a, z; CUDA_OK(cudaEventCreate(&a)); CUDA_OK(cudaEventCreate(&z));
    CUDA_OK(cudaEventRecord(a));
    launch_self<T>(dev, sd, d_q, B, (T)sc.margin, st, mc, pi, pj, ne);
    CUDA_OK(cudaEventRecord(z)); CUDA_OK(cudaEventSynchronize(z));
    CUDA_OK(cudaEventElapsedTime(&out.kernel_ms, a, z));
    (void)cudaEventDestroy(a); (void)cudaEventDestroy(z);
    out.prep_ms = sd.prep_ms; out.upload_ms = sd.upload_ms; out.eligible_pairs = sd.eligible; out.n_bodies = sd.NB; out.n_body_pairs = sd.NP;
    if (out.status) download(out.status, st, size_t(B));
    if (out.min_clearance) download(out.min_clearance, mc, size_t(B));
    if (out.pair_i) download(out.pair_i, pi, size_t(B));
    if (out.pair_j) download(out.pair_j, pj, size_t(B));
    if (out.evaluated_pairs) download(out.evaluated_pairs, ne, size_t(B));
}

template<typename T>
LMKernelInfo self_kernel_info() {
    cudaFuncAttributes a{};
    CUDA_OK(cudaFuncGetAttributes(&a, tree_self_kernel<T>));
    return LMKernelInfo{a.numRegs, int(a.localSizeBytes), int(a.sharedSizeBytes), int(self_shared_bytes<T>(SELF_WARPS, 621, 28)), a.maxThreadsPerBlock};
}

template<typename T>
LMKernelInfo env_kernel_info() {
    cudaFuncAttributes a{};
    CUDA_OK(cudaFuncGetAttributes(&a, tree_env_kernel<T>));
    return LMKernelInfo{a.numRegs, int(a.localSizeBytes), int(a.sharedSizeBytes), int(env_shared_bytes<T>(ENV_WARPS)), a.maxThreadsPerBlock};
}

template<typename T>
LMKernelInfo coarse_kernel_info() {
    cudaFuncAttributes a{};
    CUDA_OK(cudaFuncGetAttributes(&a, tree_coarse_kernel<T>));
    return LMKernelInfo{a.numRegs, int(a.localSizeBytes), int(a.sharedSizeBytes), int(coarse_shared_bytes<T>(1)), a.maxThreadsPerBlock};
}

template<typename T>
void base_step(const T* q, int B, const ProblemIn<T>& in, const LMConfig& cfg, T* out) {
    if (B < 1 || !in.base) throw std::invalid_argument("base_step needs batch >= 1 and base poses");
    Device<T> dev;
    const T* d_tool = nullptr;
    ProblemIn<T> din = upload_problem<T>(dev, B, in, d_tool);
    din.base = dev.upload(in.base, size_t(B) * 7);
    const T* d_q = dev.upload(q, size_t(B) * N);
    T* d_out = dev.template alloc<T>(size_t(B) * 65);
    const size_t bytes = shared_bytes<T>(1);
    CUDA_OK(cudaFuncSetAttribute(tree_base_step_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)));
    tree_base_step_kernel<T><<<B, 32, bytes>>>(d_q, B, din, d_tool, dev.model.get(), cfg, d_out);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    download(out, d_out, size_t(B) * 65);
}

template<typename T>
LMKernelInfo lm_kernel_info(int warps) {
    cudaFuncAttributes a{};
    CUDA_OK(cudaFuncGetAttributes(&a, tree_lm_kernel<T>));
    return LMKernelInfo{a.numRegs, int(a.localSizeBytes), int(a.sharedSizeBytes), int(shared_bytes<T>(warps)), a.maxThreadsPerBlock};
}

template void evaluate<float>(const float*, int, const ProblemIn<float>&, float, EvalOut<float>&);
template void evaluate<double>(const double*, int, const ProblemIn<double>&, double, EvalOut<double>&);
template void lm<float>(const float*, int, const ProblemIn<float>&, const LMConfig&, LMOut<float>&, float*);
template void lm<double>(const double*, int, const ProblemIn<double>&, const LMConfig&, LMOut<double>&, float*);
template void solve_batch<float>(const float*, int, int, int, const ProblemIn<float>&, const LMConfig&, LMOut<float>*, SelectOut<float>&, BatchTiming*, const CoarseConfig*, CoarseIO<float>*, const EnvCollision<float>*, EnvOut<float>*, const SelfCollision<float>*, SelfOut<float>*);
template void solve_batch<double>(const double*, int, int, int, const ProblemIn<double>&, const LMConfig&, LMOut<double>*, SelectOut<double>&, BatchTiming*, const CoarseConfig*, CoarseIO<double>*, const EnvCollision<double>*, EnvOut<double>*, const SelfCollision<double>*, SelfOut<double>*);
template void env_collision<float>(const float*, const float*, int, const EnvCollision<float>&, EnvOut<float>&);
template void env_collision<double>(const double*, const double*, int, const EnvCollision<double>&, EnvOut<double>&);
template LMKernelInfo env_kernel_info<float>();
template LMKernelInfo self_kernel_info<float>();
template void self_collision<float>(const float*, int, const SelfCollision<float>&, SelfOut<float>&);
template LMKernelInfo env_kernel_info<double>();
template LMKernelInfo self_kernel_info<double>();
template void self_collision<double>(const double*, int, const SelfCollision<double>&, SelfOut<double>&);
template void proposals<float>(const float*, int, const ProblemIn<float>&, const CoarseConfig&, float*);
template void proposals<double>(const double*, int, const ProblemIn<double>&, const CoarseConfig&, double*);
template LMKernelInfo coarse_kernel_info<float>();
template LMKernelInfo coarse_kernel_info<double>();
template void select_only<float>(const float*, const float*, const float*, const float*, const unsigned char*, const float*, const unsigned int*, int, int, int, double, double, SelectOut<float>&, const unsigned char*, const float*, int, EnvOut<float>*, const unsigned char*, int, SelfOut<float>*);
template void select_only<double>(const double*, const double*, const double*, const double*, const unsigned char*, const double*, const unsigned int*, int, int, int, double, double, SelectOut<double>&, const unsigned char*, const double*, int, EnvOut<double>*, const unsigned char*, int, SelfOut<double>*);
template void base_step<float>(const float*, int, const ProblemIn<float>&, const LMConfig&, float*);
template void base_step<double>(const double*, int, const ProblemIn<double>&, const LMConfig&, double*);
template LMKernelInfo lm_kernel_info<float>(int);
template LMKernelInfo lm_kernel_info<double>(int);

}  // namespace hjcd_tree
