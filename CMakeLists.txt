cmake_minimum_required(VERSION 3.23)
project(hjcdik LANGUAGES CXX CUDA)

option(HJCDIK_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
option(BUILD_PYTHON "Build the Python extension module" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Prefer static CUDA runtime to avoid cudart64_*.dll at import time
# (CMake 3.18+)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

# set(CMAKE_CUDA_ARCHITECTURES 75;86;90)

find_package(CUDAToolkit REQUIRED)

# Use modern Python discovery in pybind11 (important for Python 3.13)
set(PYBIND11_FINDPYTHON ON)
if(BUILD_PYTHON)
  find_package(pybind11 CONFIG REQUIRED)
endif()

# ----- Optional GRiD subdir -----
set(GRID_DIR ${CMAKE_SOURCE_DIR}/external/GRiD)
if(EXISTS "${GRID_DIR}/CMakeLists.txt")
  add_subdirectory(${GRID_DIR} EXCLUDE_FROM_ALL)
  set(HAVE_GRID_TARGETS TRUE)
else()
  set(HAVE_GRID_TARGETS FALSE)
endif()

# ----- Sources -----
set(HJCDIK_COMMON_SOURCES
  src/hjcd_kernel.cu
  src/util.cpp
)

# ===== Native executable =====
add_executable(hjcdik
  src/main.cpp
  ${HJCDIK_COMMON_SOURCES}
)

target_include_directories(hjcdik PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/external/GRiD
  ${CMAKE_SOURCE_DIR}/external/GRiD/include
)

# Link STATIC CUDA runtime
target_link_libraries(hjcdik PRIVATE CUDA::cudart_static)
if(HAVE_GRID_TARGETS)
  if(TARGET grid::grid)
    target_link_libraries(hjcdik PRIVATE grid::grid)
  elseif(TARGET GRiD)
    target_link_libraries(hjcdik PRIVATE GRiD)
  endif()
endif()

# Warnings: scope to CXX only (avoid leaking into NVCC)
if(MSVC)
  target_compile_options(hjcdik PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/permissive- /W4>)
else()
  target_compile_options(hjcdik PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
endif()

# CUDA-specific flags
target_compile_options(hjcdik PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3>
)

# ===== Python extension =====
if(BUILD_PYTHON)
  pybind11_add_module(_hjcdik MODULE
    src/pybind_module.cpp
    ${HJCDIK_COMMON_SOURCES}
  )
  target_include_directories(_hjcdik PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external/GRiD
    ${CMAKE_SOURCE_DIR}/external/GRiD/include
  )

  # Link STATIC CUDA runtime for the extension too
  target_link_libraries(_hjcdik PRIVATE CUDA::cudart_static)
  if(HAVE_GRID_TARGETS)
    if(TARGET grid::grid)
      target_link_libraries(_hjcdik PRIVATE grid::grid)
      set(GRID_DLL_TARGET grid::grid)
    elseif(TARGET GRiD)
      target_link_libraries(_hjcdik PRIVATE GRiD)
      set(GRID_DLL_TARGET GRiD)
    endif()
  endif()

  if(MSVC)
    target_compile_options(_hjcdik PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/permissive- /W4>)
  else()
    target_compile_options(_hjcdik PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
  endif()

  target_compile_options(_hjcdik PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3>
  )

  # Place the extension into the package directory on install
  install(TARGETS _hjcdik DESTINATION hjcdik)

  # --- If GRiD is a shared library, copy its DLL next to the .pyd and install it ---
  if(HAVE_GRID_TARGETS AND GRID_DLL_TARGET)
    get_target_property(_grid_type ${GRID_DLL_TARGET} TYPE)
    if(_grid_type STREQUAL "SHARED_LIBRARY" OR _grid_type STREQUAL "MODULE_LIBRARY")
      add_custom_command(TARGET _hjcdik POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:${GRID_DLL_TARGET}>
                $<TARGET_FILE_DIR:_hjcdik>)

      install(FILES $<TARGET_FILE:${GRID_DLL_TARGET}> DESTINATION hjcdik)
    endif()
  endif()
endif()
