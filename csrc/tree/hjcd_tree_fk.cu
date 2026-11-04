// Fixed-base TREE target forward kinematics (humanoid-v1 M2b): verification entry point only.
//
// One warp per configuration, lane j owns joint j (N <= 32). GRiD supplies the per-joint local
// transforms (load_update_XmatsHom_helpers for the q-independent template incl. every fixed frame,
// update_XmatHom_joint for joint j at theta_j); HJCD composes them over the generated parent table
// level by level (a parent always has a smaller depth), then applies each target's tool to its anchor.
// No serial-chain assumption, no single-target kernels, no IK, no collision, no persistent state.
#include "grid.cuh"
#include "hjcd_joint_model.h"
#include "hjcd_targets.cuh"
#include "glass.cuh"
#include "kernel/cuda_memory.h"
#include "tree/hjcd_tree_fk.h"

static_assert(hjcd_model::NUM_JOINTS == grid::NUM_JOINTS, "joint model / GRiD header mismatch");
static_assert(hjcd_targets::NUM_JOINTS == grid::NUM_JOINTS, "target metadata / GRiD header mismatch");
static_assert(grid::NUM_JOINTS >= 1 && grid::NUM_JOINTS <= 32, "one joint per warp lane");
static_assert(hjcd_targets::NUM_TARGETS >= 1 && hjcd_targets::NUM_TARGETS <= 32, "one target per lane");

namespace hjcd_tree {
namespace {

constexpr int N = grid::NUM_JOINTS;
constexpr int K = hjcd_targets::NUM_TARGETS;
constexpr unsigned FULL = 0xFFFFFFFFu;

template<typename T>
__host__ __device__ constexpr size_t shared_bytes(int warps) {
    // T: template XmatsHom | q0 (N) | sin/cos scratch (2N) | per warp: q (32) + local (16N) + world (16N); then int topology
    return sizeof(T) * (size_t(grid::XHOM_T_COUNT) + 3 * N + size_t(warps) * (32 + 32 * N))
         + sizeof(int) * size_t(grid::TOPOLOGY_HELPERS_COUNT);
}

template<typename T, int SOURCE>
__global__ void tree_target_fk_kernel(const T* __restrict__ q, int batch, T* __restrict__ out,
                                      const grid::robotModel<T>* __restrict__ model) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int warps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    T* s_xhom = reinterpret_cast<T*>(smem);
    T* s_q0 = s_xhom + grid::XHOM_T_COUNT;
    T* s_tmp = s_q0 + N;
    T* s_warp = s_tmp + 2 * N + size_t(warp) * (32 + 32 * N);
    int* s_topo = reinterpret_cast<int*>(reinterpret_cast<T*>(smem) + grid::XHOM_T_COUNT + 3 * N
                                         + size_t(warps) * (32 + 32 * N));
    T* s_q = s_warp;
    T* s_local = s_warp + 32;
    T* s_world = s_local + 16 * N;

    // Block prologue: the q-independent template (all fixed frames included) at q = 0.
    for (int i = threadIdx.x; i < N; i += blockDim.x) s_q0[i] = T(0);
    __syncthreads();
    grid::load_update_XmatsHom_helpers<T>(s_xhom, s_topo, s_q0, model, s_tmp);
    __syncthreads();

    const int b = blockIdx.x * warps + warp;
    if (b >= batch) return;                                   // warp-uniform; no block barrier follows

    if (lane < N) {
        s_q[lane] = q[size_t(b) * N + lane];
        grid::update_XmatHom_joint<T>(&s_local[16 * lane], s_xhom, lane, s_q[lane]);
    }
    __syncwarp(FULL);
    for (int d = 0; d <= hjcd_targets::MAX_DEPTH; ++d) {
        if (lane < N && hjcd_targets::JOINT_DEPTH[lane] == d) {
            const int p = hjcd_targets::JOINT_PARENT[lane];
            if (p < 0) {
                #pragma unroll
                for (int m = 0; m < 16; ++m) s_world[16 * lane + m] = s_local[16 * lane + m];
            } else {
                glass::thread::gemm<T, 4, 4, 4>(T(1), &s_world[16 * p], &s_local[16 * lane], &s_world[16 * lane]);
            }
        }
        __syncwarp(FULL);
    }
    if (lane < K) {
        T tool[16], pose[16];
        #pragma unroll
        for (int m = 0; m < 16; ++m)
            tool[m] = SOURCE == 0 ? hjcd_targets::tool<T>(16 * lane + m)
                                  : s_xhom[16 * hjcd_targets::TARGET_FIXED_FRAME_IDX[lane] + m];
        glass::thread::gemm<T, 4, 4, 4>(T(1), &s_world[16 * hjcd_targets::TARGET_ANCHOR_JID[lane]], tool, pose);
        T* dst = out + (size_t(b) * K + lane) * 16;
        #pragma unroll
        for (int r = 0; r < 4; ++r)
            #pragma unroll
            for (int c = 0; c < 4; ++c) dst[4 * r + c] = pose[4 * c + r];   // column-major -> row-major
    }
}

template<typename T>
void check_args(int batch, int warps, int source) {
    if (batch < 1) throw std::invalid_argument("batch must be >= 1");
    if (warps < 1 || warps > 32) throw std::invalid_argument("warps_per_block must be in [1, 32]");
    if (source != 0 && source != 1) throw std::invalid_argument("source must be 0 (metadata) or 1 (grid)");
}

template<typename T>
void configure(int warps, int source) {
    const int bytes = int(shared_bytes<T>(warps));
    if (source == 0) CUDA_OK(cudaFuncSetAttribute(tree_target_fk_kernel<T, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    else             CUDA_OK(cudaFuncSetAttribute(tree_target_fk_kernel<T, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

}  // namespace

template<typename T>
void target_poses(const T* q, int batch, T* out, int warps, int source, int repeats, float* kernel_ms) {
    check_args<T>(batch, warps, source);
    grid::robotModel_owner<T> model;
    const char* op = nullptr;
    const cudaError_t init = model.init(&op);
    if (init != cudaSuccess)
        throw std::runtime_error(std::string("GRiD robot model initialization failed at ") + (op ? op : "?")
                                 + ": " + cudaGetErrorString(init));
    hjcd::DeviceAllocations memory;
    T *d_q = nullptr, *d_out = nullptr;
    memory.allocate(d_q, sizeof(T) * size_t(batch) * N);
    memory.allocate(d_out, sizeof(T) * size_t(batch) * K * 16);
    CUDA_OK(cudaMemcpy(d_q, q, sizeof(T) * size_t(batch) * N, cudaMemcpyHostToDevice));
    configure<T>(warps, source);
    const dim3 grid_dim((batch + warps - 1) / warps), block_dim(32 * warps);
    const size_t bytes = shared_bytes<T>(warps);
    auto launch = [&] {
        if (source == 0) tree_target_fk_kernel<T, 0><<<grid_dim, block_dim, bytes>>>(d_q, batch, d_out, model.get());
        else             tree_target_fk_kernel<T, 1><<<grid_dim, block_dim, bytes>>>(d_q, batch, d_out, model.get());
        CUDA_OK(cudaGetLastError());
    };
    launch();
    CUDA_OK(cudaDeviceSynchronize());
    if (repeats > 0 && kernel_ms) {
        cudaEvent_t start, stop;
        CUDA_OK(cudaEventCreate(&start));
        CUDA_OK(cudaEventCreate(&stop));
        CUDA_OK(cudaEventRecord(start));
        for (int i = 0; i < repeats; ++i) launch();
        CUDA_OK(cudaEventRecord(stop));
        CUDA_OK(cudaEventSynchronize(stop));
        float ms = 0.f;
        CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
        (void)cudaEventDestroy(start);
        (void)cudaEventDestroy(stop);
        *kernel_ms = ms / float(repeats);
    }
    CUDA_OK(cudaMemcpy(out, d_out, sizeof(T) * size_t(batch) * K * 16, cudaMemcpyDeviceToHost));
    memory.clear();
    const cudaError_t freed = model.free(&op);
    if (freed != cudaSuccess) throw std::runtime_error(std::string("GRiD robot model release failed: ") + cudaGetErrorString(freed));
}

template<typename T>
KernelInfo kernel_info(int warps) {
    cudaFuncAttributes a{};
    CUDA_OK(cudaFuncGetAttributes(&a, tree_target_fk_kernel<T, 0>));
    return KernelInfo{a.numRegs, int(a.localSizeBytes), int(a.sharedSizeBytes), int(shared_bytes<T>(warps)),
                      a.maxThreadsPerBlock, a.ptxVersion, a.binaryVersion};
}

template void target_poses<float>(const float*, int, float*, int, int, int, float*);
template void target_poses<double>(const double*, int, double*, int, int, int, float*);
template KernelInfo kernel_info<float>(int);
template KernelInfo kernel_info<double>(int);

}  // namespace hjcd_tree
