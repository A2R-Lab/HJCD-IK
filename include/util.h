#pragma once
#include <array>
#include <vector>
#include <string>

#ifndef CUDA_OK
#define CUDA_OK(stmt)                                                         \
    do {                                                                      \
        cudaError_t __err = (stmt);                                           \
        if (__err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(__err), __FILE__, __LINE__);           \
            std::abort();                                                     \
        }                                                                     \
    } while (0)
#endif
