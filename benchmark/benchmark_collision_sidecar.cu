// Standalone CUDA micro-benchmark for the collision sidecar (Checkpoint 2, Stage 6).
// Self-contained (no artifacts needed): times FK (f32 + f64) and the warp-cooperative convex
// support scan in isolation ("benchmark support scans separately"). Batched full/incremental
// throughput is measured by benchmark_collision_sidecar.py via the kernel-only bench entry points.
//   nvcc -std=c++17 -arch=sm_89 -O3 -I generated -I src benchmark/benchmark_collision_sidecar.cu -o bench
#include "collision_sidecar.cu"
#include <cstdio>
#include <vector>

using namespace g1sc;

// warp-cooperative argmax over `nverts` synthetic vertices, `nscans` times per warp.
__global__ void support_bench(const double* __restrict__ verts, int nverts, int nscans,
                              double* __restrict__ out) {
    int lane = threadIdx.x & 31;
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    double acc = 0.0;
    for (int s = 0; s < nscans; ++s) {
        double d[3] = {cos(s * 0.017), sin(s * 0.017), 0.3};
        double bd = -1e300; int bi = -1;
        for (int i = lane; i < nverts; i += 32) {
            const double* v = &verts[i * 3];
            double dp = v[0]*d[0] + v[1]*d[1] + v[2]*d[2];
            if (dp > bd || (dp == bd && i < bi)) { bd = dp; bi = i; }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            double od = __shfl_down_sync(0xffffffffu, bd, off);
            int    oi = __shfl_down_sync(0xffffffffu, bi, off);
            if (od > bd || (od == bd && oi < bi)) { bd = od; bi = oi; }
        }
        acc += bd;
    }
    if (lane == 0) out[warp] = acc;
}

int main() {
    const int B = 4096;
    std::vector<float> q(B * N_JOINTS, 0.1f);
    float* dq; float* dT; double* dTd;
    cudaMalloc(&dq, q.size() * sizeof(float));
    cudaMalloc(&dT, (size_t)B * N_LINKS * 16 * sizeof(float));
    cudaMalloc(&dTd, (size_t)B * N_LINKS * 16 * sizeof(double));
    cudaMemcpy(dq, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice);

    // FK f32
    { cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
      fk_debug_kernel<<<(B+127)/128,128>>>(dq,dT,B); cudaDeviceSynchronize();
      cudaEventRecord(a); for(int i=0;i<200;++i) fk_debug_kernel<<<(B+127)/128,128>>>(dq,dT,B);
      cudaEventRecord(b); cudaEventSynchronize(b); float ms=0; cudaEventElapsedTime(&ms,a,b); ms/=200;
      printf("FK f32   : %7.3f us / %d configs  = %6.1f ns/config  (%.1f M configs/s)\n",
             ms*1e3, B, ms*1e6/B, B/ms/1e3); }
    // FK f64
    { cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
      fk_debug_kernel_d<<<(B+127)/128,128>>>(dq,dTd,B); cudaDeviceSynchronize();
      cudaEventRecord(a); for(int i=0;i<200;++i) fk_debug_kernel_d<<<(B+127)/128,128>>>(dq,dTd,B);
      cudaEventRecord(b); cudaEventSynchronize(b); float ms=0; cudaEventElapsedTime(&ms,a,b); ms/=200;
      printf("FK f64   : %7.3f us / %d configs  = %6.1f ns/config  (%.1f M configs/s)\n",
             ms*1e3, B, ms*1e6/B, B/ms/1e3); }

    // support scan micro-benchmark at several hull sizes (base_link hull = 5584 verts)
    for (int nverts : {673, 1637, 5584}) {
        std::vector<double> verts(nverts * 3);
        for (int i = 0; i < nverts * 3; ++i) verts[i] = (double)((i * 2654435761u) % 1000) / 1000.0;
        double* dv; cudaMalloc(&dv, verts.size() * sizeof(double));
        cudaMemcpy(dv, verts.data(), verts.size() * sizeof(double), cudaMemcpyHostToDevice);
        int warps = 2048, nscans = 32; double* dout; cudaMalloc(&dout, warps * sizeof(double));
        int tpb = 256, blocks = (warps * 32 + tpb - 1) / tpb;
        support_bench<<<blocks, tpb>>>(dv, nverts, nscans, dout); cudaDeviceSynchronize();
        cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a);
        for (int i = 0; i < 100; ++i) support_bench<<<blocks, tpb>>>(dv, nverts, nscans, dout);
        cudaEventRecord(b); cudaEventSynchronize(b); float ms=0; cudaEventElapsedTime(&ms,a,b);
        double per_scan_ns = (double)ms/100.0/warps/nscans*1e6;
        printf("support  : %5d verts -> %6.1f ns/scan  (warp-cooperative argmax)\n", nverts, per_scan_ns);
        cudaFree(dv); cudaFree(dout);
    }
    cudaFree(dq); cudaFree(dT); cudaFree(dTd);
    return 0;
}
