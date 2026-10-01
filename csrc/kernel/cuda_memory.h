#pragma once
#include "kernel/util.h"
#include <utility>
#include <vector>

namespace hjcd {

// Own temporary device allocations for one host operation. Kernel arguments stay raw
// pointers; CUDA, Thrust, and host exceptions release earlier allocations on unwind.
// Cached environments move this owner and are freed on their original device.
class DeviceAllocations {
public:
    DeviceAllocations() { CUDA_OK(cudaGetDevice(&device_)); }
    ~DeviceAllocations() noexcept { release_noexcept(); }
    DeviceAllocations(const DeviceAllocations&) = delete;
    DeviceAllocations& operator=(const DeviceAllocations&) = delete;
    DeviceAllocations(DeviceAllocations&& other) noexcept
        : device_(other.device_), pointers_(std::exchange(other.pointers_, {})) {}
    DeviceAllocations& operator=(DeviceAllocations&& other) noexcept {
        if (this != &other) {
            release_noexcept();
            device_ = other.device_;
            pointers_ = std::exchange(other.pointers_, {});
        }
        return *this;
    }

    template<typename T>
    void allocate(T*& pointer, size_t bytes) {
        // Reserve ownership before CUDA allocation: vector growth can throw too.
        pointers_.push_back(nullptr);
        CUDA_OK(cudaMalloc(&pointers_.back(), bytes));
        pointer = static_cast<T*>(pointers_.back());
    }

    template<typename T>
    T* adopt(T* pointer) {
        try { pointers_.push_back(pointer); }
        catch (...) { if (pointer) (void)cudaFree(pointer); throw; }
        return pointer;
    }

    // Checked cleanup on the allocating device; the destructor retries without throwing.
    void clear() {
        while (!pointers_.empty()) {
            CUDA_OK(cudaFree(pointers_.back()));
            pointers_.pop_back();
        }
    }

private:
    void release_noexcept() noexcept {
        if (pointers_.empty()) return;
        int previous = -1;
        if (cudaGetDevice(&previous) != cudaSuccess) return;
        if (previous != device_ && cudaSetDevice(device_) != cudaSuccess) return;
        for (auto pointer : pointers_) {
            if (pointer) (void)cudaFree(pointer);
        }
        pointers_.clear();
        if (previous != device_) (void)cudaSetDevice(previous);
    }

    int device_ = 0;
    std::vector<void*> pointers_;
};

}  // namespace hjcd
