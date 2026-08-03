"""Batch throughput + resource benchmark for the GPU collision sidecar (Checkpoint 2, Stage 6).

Reports:
  * kernel-only full-check throughput (configs/s) across batch sizes
  * kernel-only GJK-phase throughput
  * end-to-end (H2D + kernel + D2H) full-check latency
  * ptxas register / smem usage + theoretical occupancy (sm_89)
"""
from __future__ import annotations
import ctypes, json, os, re, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HJCD = os.path.dirname(HERE)
GEN = os.path.join(HJCD, "generated")
SC = os.path.join(HJCD, "collision_sidecar")
SCR = os.environ.get("SIDECAR_SCRATCH", "/tmp/sidecar_build")
sys.path.insert(0, SC)
from sidecar import Sidecar  # noqa: E402
from parity_fk import build_lib  # noqa: E402

# sm_89 (RTX 4060): 1536 threads, 65536 regs, up to 100 KB smem per SM
SM_THREADS, SM_REGS, SM_SMEM = 1536, 65536, 100 * 1024


def occupancy(regs, smem_per_block, threads_per_block):
    warps_blk = threads_per_block // 32
    by_reg = SM_REGS // (regs * threads_per_block)
    by_smem = SM_SMEM // smem_per_block if smem_per_block else 999
    by_warp = (SM_THREADS // 32) // warps_blk
    blocks = max(1, min(by_reg, by_smem, by_warp))
    return blocks, blocks * threads_per_block / SM_THREADS


def parse_ptxas(log):
    out = {}
    cur = None
    for ln in open(log):
        m = re.search(r"entry function '(\w+)'", ln)
        if m: cur = m.group(1)
        m = re.search(r"Used (\d+) registers", ln)
        if cur and m:
            sm = re.search(r"(\d+) bytes smem", ln)
            out[cur] = (int(m.group(1)), int(sm.group(1)) if sm else 0)
    return out


def main():
    sc = Sidecar()
    # bench entry points live in the ctypes lib (kernel-only, device-resident)
    lib = ctypes.CDLL(build_lib())
    # re-upload SDF/convex into THIS lib instance (separate from the module's)
    from parity_sdf import upload_sdfs
    from parity_gjk import upload_convex
    upload_sdfs(lib); upload_convex(lib)
    fp = ctypes.POINTER(ctypes.c_float)
    lib.sidecar_bench_full.restype = ctypes.c_float
    lib.sidecar_bench_full.argtypes = [fp, ctypes.c_int, ctypes.c_int]
    lib.sidecar_bench_gjk.restype = ctypes.c_float
    lib.sidecar_bench_gjk.argtypes = [fp, ctypes.c_int, ctypes.c_int]

    rng = np.random.default_rng(0)
    print("=== kernel-only throughput (full check: FK f32+f64 + 4 narrow phases) ===")
    for B in (256, 1024, 4096, 16384):
        q = np.ascontiguousarray((rng.standard_normal((B, 29)) * 0.3).astype(np.float32))
        ms = lib.sidecar_bench_full(q.ctypes.data_as(fp), B, 50)
        print(f"  B={B:6d}: {ms*1e3:8.1f} us/batch  = {ms*1e3/B:6.2f} us/config  "
              f"({B/ms:7.1f} K configs/s)")
    print("=== kernel-only GJK phase (14 pairs/config, warp-cooperative f64) ===")
    for B in (1024, 4096):
        q = np.ascontiguousarray((rng.standard_normal((B, 29)) * 0.3).astype(np.float32))
        ms = lib.sidecar_bench_gjk(q.ctypes.data_as(fp), B, 50)
        print(f"  B={B:6d}: {ms*1e3:8.1f} us/batch  = {ms*1e3/B:6.2f} us/config")

    print("=== end-to-end full check (H2D + kernel + D2H, via module) ===")
    for B in (1024, 4096):
        q = np.ascontiguousarray((rng.standard_normal((B, 29)) * 0.3).astype(np.float32))
        sc.full_check(q, 0.0)  # warm
        t = time.perf_counter()
        for _ in range(20): sc.full_check(q, 0.0)
        dt = (time.perf_counter() - t) / 20
        print(f"  B={B:6d}: {dt*1e6:8.1f} us/batch  = {dt*1e6/B:6.2f} us/config")

    print("=== CUDA resource usage / theoretical occupancy (sm_89) ===")
    log = os.path.join(GEN, "sidecar_ptxas.log")
    if not os.path.exists(log):
        subprocess.run(["bash", os.path.join(SC, "build_sidecar_module.sh")], check=True,
                       capture_output=True)
    info = parse_ptxas(log)
    for k in ("full_check_kernel", "incr_check_kernel", "gjk_gaps_kernel", "fk_debug_kernel_d"):
        hit = next((v for name, v in info.items() if k.replace("_kernel", "") in name), None)
        if not hit: continue
        regs, smem = hit
        tpb = 64 if "check" in k else 128
        blks, occ = occupancy(regs, smem, tpb)
        print(f"  {k:20s}: {regs:3d} regs, {smem:6d} B smem/block, tpb={tpb} "
              f"-> {blks} blocks/SM, {occ*100:4.1f}% occupancy")

    print("=== support-scan micro-benchmark (standalone .cu) ===")
    exe = os.path.join(SCR, "bench_sidecar")
    r = subprocess.run(["/usr/local/cuda/bin/nvcc", "-std=c++17", "-arch=sm_89", "-O3",
                        "-I", os.path.join(HJCD, "generated"), "-I", os.path.join(HJCD, "src"),
                        os.path.join(HERE, "benchmark_collision_sidecar.cu"), "-o", exe],
                       capture_output=True, text=True)
    if r.returncode:
        print("  (build failed)\n" + r.stderr[-800:])
    else:
        print(subprocess.run([exe], capture_output=True, text=True).stdout.rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
