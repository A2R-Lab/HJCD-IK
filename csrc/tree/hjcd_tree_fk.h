#pragma once
// Fixed-base TREE target forward kinematics (humanoid-v1 M2b). Host API only; no CUDA types.
//
// Separate from the single-target solver (hjcd_kernel.cu): compiled only into tree-model runtime
// artifacts (hjcdik/_tree_build.py), never into the packaged Panda module. No IK, no collision.

namespace hjcd_tree {

struct KernelInfo {
    int num_regs;
    int local_bytes;           // per-thread local memory (spills + stack arrays)
    int static_shared_bytes;
    int dynamic_shared_bytes;  // what target_poses launches with for this warps_per_block
    int max_threads_per_block;
    int ptx_version;
    int binary_version;
};

// Base-frame poses of every target for B joint configurations.
//   q   : host, B x NUM_JOINTS, row-major, joint order = GRiD id order.
//   out : host, B x NUM_TARGETS x 4 x 4, each 4x4 ROW-major (numpy-natural); batch-major, then target.
//   source 0: target = X_world[anchor] * generated tool table (hjcd_targets.cuh)
//   source 1: target = X_world[anchor] * GRiD s_XmatsHom[fixed-frame slot]   (cross-check of the two sources)
//   repeats > 0 additionally times `repeats` back-to-back launches on device; *kernel_ms = mean ms/launch.
template<typename T>
void target_poses(const T* q, int batch, T* out, int warps_per_block, int source, int repeats, float* kernel_ms);

template<typename T>
KernelInfo kernel_info(int warps_per_block);

}  // namespace hjcd_tree
