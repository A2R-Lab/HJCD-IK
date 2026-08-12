// Standalone CUDA collision sidecar -- kernels + host wrappers (Checkpoint 2).
// Behaviorally isolated: not referenced by the HJCD solver, pybind solve dispatch, or grid.cuh.
#include "collision_sidecar.cuh"
#include "env_sidecar.cuh"
#include <cstdio>

namespace g1sc {

// ---- Stage 1: FK debug kernel. One thread per config; writes all N_LINKS*16 transforms. ----
__global__ void fk_debug_kernel(const float* __restrict__ q_batch, float* __restrict__ T_batch, int B) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    sidecar_fk(&q_batch[b * N_JOINTS], &T_batch[b * N_LINKS * 16]);
}

// Host wrapper: q_batch [B*N_JOINTS] -> T_batch [B*N_LINKS*16] (column-major, root=identity).
extern "C" void sidecar_fk_batch(const float* q_host, float* T_host, int B) {
    float *dq, *dT;
    cudaMalloc(&dq, (size_t)B * N_JOINTS * sizeof(float));
    cudaMalloc(&dT, (size_t)B * N_LINKS * 16 * sizeof(float));
    cudaMemcpy(dq, q_host, (size_t)B * N_JOINTS * sizeof(float), cudaMemcpyHostToDevice);
    int tpb = 128, blocks = (B + tpb - 1) / tpb;
    fk_debug_kernel<<<blocks, tpb>>>(dq, dT, B);
    cudaMemcpy(T_host, dT, (size_t)B * N_LINKS * 16 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(dq); cudaFree(dT);
}

// ---- Stage 2 focused probes: raw distance functions on synthetic geometry ----
__global__ void seg_seg_probe_kernel(const float* __restrict__ in, float* __restrict__ out, int B) {
    int b = blockIdx.x * blockDim.x + threadIdx.x; if (b >= B) return;
    const float* a = &in[b * 12];
    out[b] = seg_seg_dist(&a[0], &a[3], &a[6], &a[9]);
}
__global__ void pt_seg_probe_kernel(const float* __restrict__ in, float* __restrict__ out, int B) {
    int b = blockIdx.x * blockDim.x + threadIdx.x; if (b >= B) return;
    const float* a = &in[b * 9];
    out[b] = pt_seg_dist(&a[0], &a[3], &a[6]);
}
extern "C" void sidecar_seg_seg_probe(const float* in, float* out, int B) {
    float *din, *dout;
    cudaMalloc(&din, (size_t)B * 12 * sizeof(float)); cudaMalloc(&dout, (size_t)B * sizeof(float));
    cudaMemcpy(din, in, (size_t)B * 12 * sizeof(float), cudaMemcpyHostToDevice);
    int tpb = 128, blk = (B + tpb - 1) / tpb;
    seg_seg_probe_kernel<<<blk, tpb>>>(din, dout, B);
    cudaMemcpy(out, dout, (size_t)B * sizeof(float), cudaMemcpyDeviceToHost); cudaDeviceSynchronize();
    cudaFree(din); cudaFree(dout);
}
extern "C" void sidecar_pt_seg_probe(const float* in, float* out, int B) {
    float *din, *dout;
    cudaMalloc(&din, (size_t)B * 9 * sizeof(float)); cudaMalloc(&dout, (size_t)B * sizeof(float));
    cudaMemcpy(din, in, (size_t)B * 9 * sizeof(float), cudaMemcpyHostToDevice);
    int tpb = 128, blk = (B + tpb - 1) / tpb;
    pt_seg_probe_kernel<<<blk, tpb>>>(din, dout, B);
    cudaMemcpy(out, dout, (size_t)B * sizeof(float), cudaMemcpyDeviceToHost); cudaDeviceSynchronize();
    cudaFree(din); cudaFree(dout);
}

// ---- Stage 2: per-config min gap over each PRIMITIVE checked link-pair ----
// out_gap [B*N_CHECKED_PAIRS]: primitive pairs filled with min cross-product gap; others = +inf.
__global__ void prim_gaps_kernel(const float* __restrict__ q_batch, float* __restrict__ out_gap, int B) {
    int b = blockIdx.x * blockDim.x + threadIdx.x; if (b >= B) return;
    float T[N_LINKS * 16];
    sidecar_fk(&q_batch[b * N_JOINTS], T);
    for (int g = 0; g < N_CHECKED_PAIRS; ++g) {
        if (PAIR_TYPE[g] != PAIR_PRIMITIVE) { out_gap[b * N_CHECKED_PAIRS + g] = 1e30f; continue; }
        int any; out_gap[b * N_CHECKED_PAIRS + g] = prim_linkpair_gap(PAIR_LINK_A[g], PAIR_LINK_B[g], T, 0.0f, &any);
    }
}
extern "C" void sidecar_prim_gaps(const float* q_host, float* gap_host, int B) {
    float *dq, *dg;
    cudaMalloc(&dq, (size_t)B * N_JOINTS * sizeof(float));
    cudaMalloc(&dg, (size_t)B * N_CHECKED_PAIRS * sizeof(float));
    cudaMemcpy(dq, q_host, (size_t)B * N_JOINTS * sizeof(float), cudaMemcpyHostToDevice);
    int tpb = 64, blk = (B + tpb - 1) / tpb;
    prim_gaps_kernel<<<blk, tpb>>>(dq, dg, B);
    cudaMemcpy(gap_host, dg, (size_t)B * N_CHECKED_PAIRS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize(); cudaFree(dq); cudaFree(dg);
}

// ---- Stage 3: upload cluster SDF grids (int16, C-order) to device ----
// The device allocations are ALSO handed to the solver TU (hjcd_kernel.cu) at bind time: with
// SEPARABLE_COMPILATION OFF each TU owns a private copy of the g_sdf / g_cverts pointer symbols,
// and hard mode dereferences the solver TU's copy. Recording the raw pointers here is what lets
// both symbols name the SAME memory instead of the sidecar's upload silently not reaching the
// solver. Same bytes, two symbols -- not two uploads.
namespace { short* g_sdf_dev[8] = {nullptr}; double* g_cverts_dev = nullptr; }
extern "C" const void* sidecar_device_sdf_ptr(int cid) {
    return (cid >= 0 && cid < 8) ? (const void*)g_sdf_dev[cid] : nullptr;
}
extern "C" const void* sidecar_device_convex_ptr() { return (const void*)g_cverts_dev; }

// Has the bulk model data (SDF grids + convex vertices) actually been uploaded?
//
// WHY THIS EXISTS: every narrow phase dereferences g_sdf / g_cverts unconditionally. Before the
// upload those symbols are NULL, and a check whose configurations reach the SDF or GJK stage then
// performs an illegal device read. CUDA reports that asynchronously, so the crash surfaces in
// whatever unrelated kernel runs NEXT -- observed as an "illegal memory access" attributed to
// grid.cuh in the following solve. Callers are gated on this instead of on luck.
extern "C" int sidecar_model_uploaded() {
    for (int c = 0; c < N_CLUSTERS && c < 8; ++c) if (!g_sdf_dev[c]) return 0;
    return g_cverts_dev ? 1 : 0;
}

extern "C" void sidecar_upload_sdf(int cid, const short* host_grid, int n) {
    short* d;
    cudaMalloc(&d, (size_t)n * sizeof(short));
    cudaMemcpy(d, host_grid, (size_t)n * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(g_sdf, &d, sizeof(short*), (size_t)cid * sizeof(short*));
    if (cid >= 0 && cid < 8) g_sdf_dev[cid] = d;
}

// Per-config gap + SDF eval count over each CLUSTER checked link-pair.
__global__ void cluster_gaps_kernel(const float* __restrict__ q_batch, float* __restrict__ out_gap,
                                    int* __restrict__ out_ev, int B) {
    int b = blockIdx.x * blockDim.x + threadIdx.x; if (b >= B) return;
    float T[N_LINKS * 16];
    sidecar_fk(&q_batch[b * N_JOINTS], T);
    for (int g = 0; g < N_CHECKED_PAIRS; ++g) {
        if (PAIR_TYPE[g] != PAIR_CLUSTER_SDF) { out_gap[b * N_CHECKED_PAIRS + g] = 1e30f;
                                                out_ev[b * N_CHECKED_PAIRS + g] = 0; continue; }
        int a = PAIR_LINK_A[g], bb = PAIR_LINK_B[g];
        int cid, cl_link, limb;
        if (LINK_CLUSTER[a] >= 0) { cid = LINK_CLUSTER[a]; cl_link = a; limb = bb; }
        else                      { cid = LINK_CLUSTER[bb]; cl_link = bb; limb = a; }
        int any, ev = 0;
        out_gap[b * N_CHECKED_PAIRS + g] = cluster_linkpair_gap(cid, cl_link, limb, T, 0.0f, &any, &ev);
        out_ev[b * N_CHECKED_PAIRS + g] = ev;
    }
}
extern "C" void sidecar_cluster_gaps(const float* q_host, float* gap_host, int* ev_host, int B) {
    float *dq, *dg; int* de;
    cudaMalloc(&dq, (size_t)B * N_JOINTS * sizeof(float));
    cudaMalloc(&dg, (size_t)B * N_CHECKED_PAIRS * sizeof(float));
    cudaMalloc(&de, (size_t)B * N_CHECKED_PAIRS * sizeof(int));
    cudaMemcpy(dq, q_host, (size_t)B * N_JOINTS * sizeof(float), cudaMemcpyHostToDevice);
    int tpb = 64, blk = (B + tpb - 1) / tpb;
    cluster_gaps_kernel<<<blk, tpb>>>(dq, dg, de, B);
    cudaMemcpy(gap_host, dg, (size_t)B * N_CHECKED_PAIRS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(ev_host, de, (size_t)B * N_CHECKED_PAIRS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize(); cudaFree(dq); cudaFree(dg); cudaFree(de);
}

// ---- Stage 4: upload exact convex vertices (double3-packed) for the f64 GJK ----
extern "C" void sidecar_upload_convex(const double* host_verts, int n_verts) {
    double* d;
    cudaMalloc(&d, (size_t)n_verts * 3 * sizeof(double));
    cudaMemcpy(d, host_verts, (size_t)n_verts * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(g_cverts, &d, sizeof(double*));
    g_cverts_dev = d;
}

// f64 FK prepass: one thread per config -> double link transforms.
__global__ void fk_debug_kernel_d(const float* __restrict__ q_batch, double* __restrict__ T_batch, int B) {
    int b = blockIdx.x * blockDim.x + threadIdx.x; if (b >= B) return;
    sidecar_fk_d(&q_batch[b * N_JOINTS], &T_batch[b * N_LINKS * 16]);
}

// One warp per (config, gjk-ordinal). out_gap/out_iters: [B * N_GJK_PAIRS]. f64 transforms.
__global__ void gjk_gaps_kernel(const double* __restrict__ Td_batch, float* __restrict__ out_gap,
                                int* __restrict__ out_iters, int B) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= B * N_GJK_PAIRS) return;
    int b = warp / N_GJK_PAIRS, ord = warp % N_GJK_PAIRS;
    int g = -1, cnt = 0;
    for (int k = 0; k < N_CHECKED_PAIRS; ++k)
        if (PAIR_TYPE[k] == PAIR_CONVEX_GJK) { if (cnt == ord) { g = k; break; } ++cnt; }
    const double* Td = &Td_batch[b * N_LINKS * 16];
    int iters = 0;
    double gap = gjk_linkpair_gap_d(PAIR_LINK_A[g], PAIR_LINK_B[g], Td, 0.0, lane, &iters);
    if (lane == 0) { out_gap[b * N_GJK_PAIRS + ord] = (float)gap; out_iters[b * N_GJK_PAIRS + ord] = iters; }
}
extern "C" void sidecar_gjk_gaps(const float* q_host, float* gap_host, int* iter_host, int B) {
    float *dq, *dg; double* dT; int* di;
    cudaMalloc(&dq, (size_t)B * N_JOINTS * sizeof(float));
    cudaMalloc(&dT, (size_t)B * N_LINKS * 16 * sizeof(double));
    cudaMalloc(&dg, (size_t)B * N_GJK_PAIRS * sizeof(float));
    cudaMalloc(&di, (size_t)B * N_GJK_PAIRS * sizeof(int));
    cudaMemcpy(dq, q_host, (size_t)B * N_JOINTS * sizeof(float), cudaMemcpyHostToDevice);
    fk_debug_kernel_d<<<(B + 127) / 128, 128>>>(dq, dT, B);
    int warps = B * N_GJK_PAIRS, tpb = 256, blk = (warps * 32 + tpb - 1) / tpb;
    gjk_gaps_kernel<<<blk, tpb>>>(dT, dg, di, B);
    cudaMemcpy(gap_host, dg, (size_t)B * N_GJK_PAIRS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(iter_host, di, (size_t)B * N_GJK_PAIRS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize(); cudaFree(dq); cudaFree(dT); cudaFree(dg); cudaFree(di);
}

// ================= Stage 5: full + incremental checker (one warp per config) =================
#define WPB 2   // warps per block (shared FK budget: WPB*(f32+f64) link transforms)

// FULL sparse check: per config, colliding byte for every checked pair (all 4 narrow phases).
__global__ void full_check_kernel(const float* __restrict__ q, unsigned char* __restrict__ out,
                                  int B, float margin) {
    __shared__ float  shT[WPB * N_LINKS * 16];
    __shared__ double shTd[WPB * N_LINKS * 16];
    int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int b = blockIdx.x * WPB + w; if (b >= B) return;
    float* T = &shT[w * N_LINKS * 16]; double* Td = &shTd[w * N_LINKS * 16];
    if (lane == 0) { sidecar_fk(&q[b * N_JOINTS], T); sidecar_fk_d(&q[b * N_JOINTS], Td); }
    __syncwarp();
    for (int g = lane; g < N_CHECKED_PAIRS; g += 32) {          // prim + cluster: lane-parallel
        if (PAIR_TYPE[g] == PAIR_CONVEX_GJK) continue;
        out[b * N_CHECKED_PAIRS + g] = (unsigned char)linkpair_colliding_nongjk(g, T, margin);
    }
    __syncwarp();
    for (int g = 0; g < N_CHECKED_PAIRS; ++g) {                 // gjk: full-warp cooperative
        if (PAIR_TYPE[g] != PAIR_CONVEX_GJK) continue;
        int c = linkpair_colliding_gjk(g, Td, margin, lane);
        if (lane == 0) out[b * N_CHECKED_PAIRS + g] = (unsigned char)c;
    }
}
// FUSED final-mode path (Checkpoint 3C.1): persistent grow-on-demand device buffers (no per-call
// cudaMalloc/cudaFree), reusing the persistent immutable model data (g_sdf/g_cverts uploaded once).
// One q H2D + one verdict D2H remain (the LM result leaves the untouchable refine path on the host,
// so a fully device-resident hook would require modifying hjcd_kernel.cu -- deferred to hard mode).
namespace { float* g_ws_q = nullptr; unsigned char* g_ws_out = nullptr; int g_ws_cap = 0, g_ws_nalloc = 0; }
extern "C" void sidecar_full_check(const float* q_host, unsigned char* out_host, int B, float margin) {
    if (B > g_ws_cap) {                                   // grow (and count) only when a bigger batch arrives
        if (g_ws_q) { cudaFree(g_ws_q); cudaFree(g_ws_out); }
        cudaMalloc(&g_ws_q, (size_t)B * N_JOINTS * sizeof(float));
        cudaMalloc(&g_ws_out, (size_t)B * N_CHECKED_PAIRS);
        g_ws_cap = B; ++g_ws_nalloc;
    }
    cudaMemcpy(g_ws_q, q_host, (size_t)B * N_JOINTS * sizeof(float), cudaMemcpyHostToDevice);
    full_check_kernel<<<(B + WPB - 1) / WPB, WPB * 32>>>(g_ws_q, g_ws_out, B, margin);
    cudaMemcpy(out_host, g_ws_out, (size_t)B * N_CHECKED_PAIRS, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}
// how many times the persistent full-check workspace has (re)allocated -- 0 growth after warm-up.
extern "C" int sidecar_ws_nalloc() { return g_ws_nalloc; }

// INCREMENTAL check: trial verdict = committed base overlaid by the joint's affected pairs at q_new.
// The committed `base` buffer is never mutated; only affected verdict bytes are recomputed (no full-state clone).
__global__ void incr_check_kernel(const float* __restrict__ qbase, const unsigned char* __restrict__ base,
                                  const int* __restrict__ jidx, const float* __restrict__ newval,
                                  unsigned char* __restrict__ out, int J, float margin) {
    __shared__ float  shT[WPB * N_LINKS * 16];
    __shared__ double shTd[WPB * N_LINKS * 16];
    __shared__ float  shq[WPB * N_JOINTS];
    int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int j = blockIdx.x * WPB + w; if (j >= J) return;
    float* q = &shq[w * N_JOINTS];
    for (int i = lane; i < N_JOINTS; i += 32) q[i] = qbase[j * N_JOINTS + i];
    __syncwarp();
    if (lane == 0) q[jidx[j]] = newval[j];
    __syncwarp();
    float* T = &shT[w * N_LINKS * 16]; double* Td = &shTd[w * N_LINKS * 16];
    if (lane == 0) { sidecar_fk(q, T); sidecar_fk_d(q, Td); }
    __syncwarp();
    for (int g = lane; g < N_CHECKED_PAIRS; g += 32)            // committed base -> trial
        out[j * N_CHECKED_PAIRS + g] = base[j * N_CHECKED_PAIRS + g];
    __syncwarp();
    int off0 = JOINT_AFFPAIR_OFF[jidx[j]], off1 = JOINT_AFFPAIR_OFF[jidx[j] + 1];
    for (int k = off0 + lane; k < off1; k += 32) {             // affected prim/cluster lane-parallel
        int g = JOINT_AFFPAIR[k];
        if (PAIR_TYPE[g] == PAIR_CONVEX_GJK) continue;
        out[j * N_CHECKED_PAIRS + g] = (unsigned char)linkpair_colliding_nongjk(g, T, margin);
    }
    __syncwarp();
    for (int k = off0; k < off1; ++k) {                        // affected gjk full-warp
        int g = JOINT_AFFPAIR[k];
        if (PAIR_TYPE[g] != PAIR_CONVEX_GJK) continue;
        int c = linkpair_colliding_gjk(g, Td, margin, lane);
        if (lane == 0) out[j * N_CHECKED_PAIRS + g] = (unsigned char)c;
    }
}
extern "C" void sidecar_incr_check(const float* qbase_host, const unsigned char* base_host,
                                   const int* jidx_host, const float* newval_host,
                                   unsigned char* out_host, int J, float margin) {
    float *dqb, *dnv; unsigned char *dbase, *dout; int* djx;
    cudaMalloc(&dqb, (size_t)J * N_JOINTS * sizeof(float));
    cudaMalloc(&dbase, (size_t)J * N_CHECKED_PAIRS);
    cudaMalloc(&djx, (size_t)J * sizeof(int));
    cudaMalloc(&dnv, (size_t)J * sizeof(float));
    cudaMalloc(&dout, (size_t)J * N_CHECKED_PAIRS);
    cudaMemcpy(dqb, qbase_host, (size_t)J * N_JOINTS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dbase, base_host, (size_t)J * N_CHECKED_PAIRS, cudaMemcpyHostToDevice);
    cudaMemcpy(djx, jidx_host, (size_t)J * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dnv, newval_host, (size_t)J * sizeof(float), cudaMemcpyHostToDevice);
    incr_check_kernel<<<(J + WPB - 1) / WPB, WPB * 32>>>(dqb, dbase, djx, dnv, dout, J, margin);
    cudaMemcpy(out_host, dout, (size_t)J * N_CHECKED_PAIRS, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(dqb); cudaFree(dbase); cudaFree(djx); cudaFree(dnv); cudaFree(dout);
}

// ================= Stage 6: kernel-only timing (SDF/convex must be uploaded first) =================
// Times `iters` launches of the full-check kernel with device-resident buffers (no H2D/D2H in the
// timed region). Returns average kernel milliseconds. FK(f32+f64) + all four narrow phases.
extern "C" float sidecar_bench_full(const float* q_host, int B, int iters) {
    float* dq; unsigned char* dout;
    cudaMalloc(&dq, (size_t)B * N_JOINTS * sizeof(float));
    cudaMalloc(&dout, (size_t)B * N_CHECKED_PAIRS);
    cudaMemcpy(dq, q_host, (size_t)B * N_JOINTS * sizeof(float), cudaMemcpyHostToDevice);
    int grid = (B + WPB - 1) / WPB, blk = WPB * 32;
    full_check_kernel<<<grid, blk>>>(dq, dout, B, 0.0f);   // warm-up
    cudaDeviceSynchronize();
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < iters; ++i) full_check_kernel<<<grid, blk>>>(dq, dout, B, 0.0f);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1); cudaFree(dq); cudaFree(dout);
    return ms / iters;
}
// Times the GJK-only kernel (f64 FK prepass excluded from the timed loop: it is done once).
extern "C" float sidecar_bench_gjk(const float* q_host, int B, int iters) {
    float* dq; double* dT; float* dg; int* di;
    cudaMalloc(&dq, (size_t)B * N_JOINTS * sizeof(float));
    cudaMalloc(&dT, (size_t)B * N_LINKS * 16 * sizeof(double));
    cudaMalloc(&dg, (size_t)B * N_GJK_PAIRS * sizeof(float));
    cudaMalloc(&di, (size_t)B * N_GJK_PAIRS * sizeof(int));
    cudaMemcpy(dq, q_host, (size_t)B * N_JOINTS * sizeof(float), cudaMemcpyHostToDevice);
    fk_debug_kernel_d<<<(B + 127) / 128, 128>>>(dq, dT, B);
    int warps = B * N_GJK_PAIRS, tpb = 256, blk = (warps * 32 + tpb - 1) / tpb;
    gjk_gaps_kernel<<<blk, tpb>>>(dT, dg, di, B); cudaDeviceSynchronize();
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1); cudaEventRecord(t0);
    for (int i = 0; i < iters; ++i) gjk_gaps_kernel<<<blk, tpb>>>(dT, dg, di, B);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(dq); cudaFree(dT); cudaFree(dg); cudaFree(di);
    return ms / iters;
}

// ================= CUDA ENVIRONMENT checker: scene upload + check (Task A) =================
// Immutable scene uploaded ONCE (device pointers via cudaMemcpyToSymbol, like g_cverts). Persistent
// grow-on-demand candidate workspace -- no per-call scene re-upload, no per-candidate malloc.
namespace {
    bool  g_env_ready = false;
    int*    d_otype=nullptr; double* d_broad=nullptr; double* d_box=nullptr; double* d_plane=nullptr;
    double* d_sph=nullptr;   double* d_reg=nullptr;   int*    d_rallow=nullptr;
    int*    d_plink=nullptr; double* d_poff=nullptr;  double* d_ptype=nullptr;
    // candidate workspace
    double* d_q=nullptr; int* d_asg=nullptr; unsigned char* d_flags=nullptr;
    int g_env_cap=0, g_env_nalloc=0;
    template<class T> void up(T** dp, const T* h, size_t n){ cudaMalloc(dp,n*sizeof(T));
        cudaMemcpy(*dp,h,n*sizeof(T),cudaMemcpyHostToDevice); }
}
// Upload the compiled native scene as SoA. Called once; re-callable (frees + re-uploads) if the
// scene changes. Sets every device symbol the env kernel reads.
extern "C" void sidecar_upload_scene(int nobj,
        const int* otype, const double* broad, const double* box, const double* plane,
        const double* sph, const double* reg, const int* rallow,
        const int* plink, const double* poff, const double* ptype){
    if (g_env_ready){ cudaFree(d_otype);cudaFree(d_broad);cudaFree(d_box);cudaFree(d_plane);
        cudaFree(d_sph);cudaFree(d_reg);cudaFree(d_rallow);cudaFree(d_plink);cudaFree(d_poff);cudaFree(d_ptype); }
    up(&d_otype,otype,nobj); up(&d_broad,broad,(size_t)nobj*4); up(&d_box,box,(size_t)nobj*10);
    up(&d_plane,plane,(size_t)nobj*6); up(&d_sph,sph,(size_t)nobj*4); up(&d_reg,reg,(size_t)nobj*16);
    up(&d_rallow,rallow,nobj); up(&d_plink,plink,4); up(&d_poff,poff,12); up(&d_ptype,ptype,4);
    cudaMemcpyToSymbol(g1sc::g_env_nobj,&nobj,sizeof(int));
    cudaMemcpyToSymbol(g1sc::g_env_otype,&d_otype,sizeof(void*));
    cudaMemcpyToSymbol(g1sc::g_env_broad,&d_broad,sizeof(void*));
    cudaMemcpyToSymbol(g1sc::g_env_box,&d_box,sizeof(void*));
    cudaMemcpyToSymbol(g1sc::g_env_plane,&d_plane,sizeof(void*));
    cudaMemcpyToSymbol(g1sc::g_env_sph,&d_sph,sizeof(void*));
    cudaMemcpyToSymbol(g1sc::g_env_reg,&d_reg,sizeof(void*));
    cudaMemcpyToSymbol(g1sc::g_env_rallow,&d_rallow,sizeof(void*));
    cudaMemcpyToSymbol(g1sc::g_env_plink,&d_plink,sizeof(void*));
    cudaMemcpyToSymbol(g1sc::g_env_poff,&d_poff,sizeof(void*));
    cudaMemcpyToSymbol(g1sc::g_env_ptype,&d_ptype,sizeof(void*));
    g_env_ready=true;
}
extern "C" int sidecar_env_ready(){ return g_env_ready?1:0; }
extern "C" int sidecar_env_nalloc(){ return g_env_nalloc; }

// Batched env check. q_host [B*36] (base pos3+quat4+joints29, f64); assign_host [B*4] (object idx or
// -1). flags_host [B*6] out. Persistent workspace grows only when a bigger batch arrives.
extern "C" void sidecar_env_check(const double* q_host, const int* assign_host,
                                  unsigned char* flags_host, int B){
    if (B>g_env_cap){
        if (d_q){ cudaFree(d_q); cudaFree(d_asg); cudaFree(d_flags); }
        cudaMalloc(&d_q,(size_t)B*36*sizeof(double));
        cudaMalloc(&d_asg,(size_t)B*4*sizeof(int));
        cudaMalloc(&d_flags,(size_t)B*6);
        g_env_cap=B; ++g_env_nalloc;
    }
    cudaMemcpy(d_q,q_host,(size_t)B*36*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_asg,assign_host,(size_t)B*4*sizeof(int),cudaMemcpyHostToDevice);
    g1sc::env_check_kernel<<<(B+ENV_WPB-1)/ENV_WPB, ENV_WPB*32>>>(d_q,d_asg,d_flags,B);
    cudaMemcpy(flags_host,d_flags,(size_t)B*6,cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}
// Timed variant: same as sidecar_env_check, but records H2D / kernel / D2H phase times via CUDA
// events into ms3[3] = {h2d_ms, kernel_ms, d2h_ms}. Used by the profiling path only; identical result.
extern "C" void sidecar_env_check_timed(const double* q_host, const int* assign_host,
                                        unsigned char* flags_host, int B, float* ms3){
    if (B>g_env_cap){
        if (d_q){ cudaFree(d_q); cudaFree(d_asg); cudaFree(d_flags); }
        cudaMalloc(&d_q,(size_t)B*36*sizeof(double));
        cudaMalloc(&d_asg,(size_t)B*4*sizeof(int));
        cudaMalloc(&d_flags,(size_t)B*6);
        g_env_cap=B; ++g_env_nalloc;
    }
    cudaEvent_t e0,e1,e2,e3; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3);
    cudaEventRecord(e0);
    cudaMemcpy(d_q,q_host,(size_t)B*36*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_asg,assign_host,(size_t)B*4*sizeof(int),cudaMemcpyHostToDevice);
    cudaEventRecord(e1);
    g1sc::env_check_kernel<<<(B+ENV_WPB-1)/ENV_WPB, ENV_WPB*32>>>(d_q,d_asg,d_flags,B);
    cudaEventRecord(e2);
    cudaMemcpy(flags_host,d_flags,(size_t)B*6,cudaMemcpyDeviceToHost);
    cudaEventRecord(e3); cudaEventSynchronize(e3);
    cudaEventElapsedTime(&ms3[0],e0,e1); cudaEventElapsedTime(&ms3[1],e1,e2); cudaEventElapsedTime(&ms3[2],e2,e3);
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3);
}
// kernel-only timing (scene + workspace resident): average ms over `iters` launches.
extern "C" float sidecar_env_bench(int B, int iters){
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    g1sc::env_check_kernel<<<(B+ENV_WPB-1)/ENV_WPB, ENV_WPB*32>>>(d_q,d_asg,d_flags,B);
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for (int i=0;i<iters;++i) g1sc::env_check_kernel<<<(B+ENV_WPB-1)/ENV_WPB, ENV_WPB*32>>>(d_q,d_asg,d_flags,B);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms=0; cudaEventElapsedTime(&ms,t0,t1); cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms/iters;
}

}  // namespace g1sc

// ---- model-info accessors (global extern "C"; readable from the host-only pybind TU) ----
// which: 0 urdf,1 joint_order,2 proxy_yaml,3 torso_sdf,4 pelvis_sdf,5 convex,6 pair_policy
extern "C" const char* sidecar_hash_str(int which) {
    using namespace g1_sidecar;
    switch (which) {
        case 0: return HASH_URDF;       case 1: return HASH_JOINT_ORDER; case 2: return HASH_PROXY_YAML;
        case 3: return HASH_TORSO_SDF;  case 4: return HASH_PELVIS_SDF;  case 5: return HASH_CONVEX;
        case 6: return HASH_PAIR_POLICY; case 7: return HASH_TYPED_PIECE;
    }
    return "";
}
// which: 0 n_joints,1 n_links,2 n_checked_pairs,3 n_prim_pairs,4 n_cluster_pairs,5 n_gjk_pairs,6 n_clusters,7 n_convex_verts
extern "C" int sidecar_model_int(int which) {
    using namespace g1_sidecar;
    switch (which) {
        case 0: return N_JOINTS;        case 1: return N_LINKS;         case 2: return N_CHECKED_PAIRS;
        case 3: return N_PRIM_PAIRS;    case 4: return N_CLUSTER_PAIRS; case 5: return N_GJK_PAIRS;
        case 6: return N_CLUSTERS;      case 7: return N_CONVEX_VERTS;
    }
    return -1;
}
