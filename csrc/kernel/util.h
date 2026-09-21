#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#ifndef CUDA_OK
#define CUDA_OK(stmt)                                                         \
    do {                                                                      \
        cudaError_t __err = (stmt);                                           \
        if (__err != cudaSuccess) {                                           \
            throw std::runtime_error(std::string("CUDA error: ") +            \
                cudaGetErrorString(__err) + " in " #stmt + " at " +           \
                __FILE__ + ":" + std::to_string(__LINE__));                    \
        }                                                                     \
    } while (0)
#endif
