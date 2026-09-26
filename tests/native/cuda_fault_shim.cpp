// Linux-only test interposer: inject recoverable failures into the REAL consumer.
// Loaded only by isolated pytest subprocesses, never linked into hjcdik.
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <cstdlib>

namespace {
int fail_at = 0, calls = 0, injected = 0, resets = 0;
void* live[256] = {};
int outstanding = 0;

template<class F> F next(const char* name) {
    auto fn = reinterpret_cast<F>(dlsym(RTLD_NEXT, name));
    if (!fn) std::abort();  // a broken harness must not report a passing test
    return fn;
}
bool fail() {
    if (fail_at && ++calls == fail_at) {
        fail_at = 0;
        ++injected;
        return true;
    }
    return false;
}
void remember(void* p) {
    for (auto& entry : live) if (!entry) { entry = p; ++outstanding; return; }
    std::abort();
}
}
extern "C" void hjcd_test_arm(int nth) { fail_at = nth; calls = 0; injected = 0; }
extern "C" int hjcd_test_injected() { return injected; }
extern "C" int hjcd_test_outstanding() { return outstanding; }
extern "C" int hjcd_test_resets() { return resets; }

extern "C" cudaError_t cudaMalloc(void** p, size_t n) {
    if (fail()) { *p = nullptr; return cudaErrorMemoryAllocation; }
    const auto e = next<decltype(&cudaMalloc)>("cudaMalloc")(p, n);
    if (e == cudaSuccess) remember(*p);
    return e;
}
extern "C" cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind kind) {
    if (kind == cudaMemcpyHostToDevice && fail()) return cudaErrorInvalidValue;
    return next<decltype(&cudaMemcpy)>("cudaMemcpy")(dst, src, n, kind);
}
extern "C" cudaError_t cudaFree(void* p) {
    const auto e = next<decltype(&cudaFree)>("cudaFree")(p);
    if (e == cudaSuccess && p)
        for (auto& entry : live) if (entry == p) { entry = nullptr; --outstanding; break; }
    return e;
}
extern "C" cudaError_t cudaDeviceReset() {
    ++resets;
    return cudaErrorNotPermitted;  // never reset even the test process's context
}
