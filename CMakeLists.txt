cmake_minimum_required(VERSION 3.23)

if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
  find_program(HJCDIK_GXX10 g++-10)
  if(HJCDIK_GXX10)
    message(STATUS "HJCD-IK: using g++-10 as CUDA host compiler: ${HJCDIK_GXX10}")
    set(CMAKE_CUDA_HOST_COMPILER
        "${HJCDIK_GXX10}"
        CACHE FILEPATH "CUDA host compiler")
  else()
    message(STATUS "HJCD-IK: g++-10 not found; using default CUDA host compiler")
  endif()
endif()

project(hjcdik LANGUAGES CXX CUDA)

option(HJCDIK_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
option(BUILD_PYTHON "Build the Python extension module" ON)
option(HJCDIK_AUTO_CODEGEN "Regenerate grid.cuh from the URDF via GRiD codegen before building" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # Build for the GPU present at configure time. Override with
  # -DCMAKE_CUDA_ARCHITECTURES=86;89 (or a specific list) for cross-compiling / CI without a GPU.
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

# Note: the CLI (src/main.cpp) emits its results YAML by hand via std::ofstream and
# nothing in the project includes the yaml-cpp library, so it is not a dependency.

set(PYBIND11_FINDPYTHON ON)
if(BUILD_PYTHON)
  find_package(pybind11 CONFIG REQUIRED)
endif()

# GRiD kinematics header. The committed, build-default header lives at
# include/test_cuh/grid.cuh and is #included as "grid.cuh" (that directory is on the
# include path below). Regenerate it with scripts/codegen/generate_grid.py, or opt into a build-time
# regenerate with -DHJCDIK_AUTO_CODEGEN=ON (uses the external/GRiD submodule's codegen).
set(GRID_HEADER         "${CMAKE_SOURCE_DIR}/include/test_cuh/grid.cuh")
set(GRID_CODEGEN_SCRIPT "${CMAKE_SOURCE_DIR}/scripts/codegen/generate_grid.py")
set(GRID_CODEGEN_URDF   "${CMAKE_SOURCE_DIR}/include/test_urdf/panda.urdf")

if(HJCDIK_AUTO_CODEGEN)
  if(NOT EXISTS "${GRID_CODEGEN_SCRIPT}")
    message(FATAL_ERROR "HJCD-IK: codegen requested, but ${GRID_CODEGEN_SCRIPT} is missing")
  endif()
  message(STATUS "HJCD-IK: regenerating grid.cuh via ${GRID_CODEGEN_SCRIPT}")
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} "${GRID_CODEGEN_SCRIPT}" "${GRID_CODEGEN_URDF}"
            -t panda_grasptarget_hand -o "${GRID_HEADER}"
    RESULT_VARIABLE _codegen_result
  )
  if(NOT _codegen_result EQUAL 0)
    message(FATAL_ERROR "GRiD codegen failed")
  endif()
endif()

if(NOT EXISTS "${GRID_HEADER}")
  message(FATAL_ERROR
    "HJCD-IK: ${GRID_HEADER} is missing. Run scripts/codegen/generate_grid.py "
    "(or configure with -DHJCDIK_AUTO_CODEGEN=ON).")
endif()
message(STATUS "HJCD-IK: using GRiD header ${GRID_HEADER}")

# Sources
set(HJCDIK_HEADERS
  include/hjcd_settings.h
  include/device_utils.cuh
  src/collision/prrtc_collision.cuh
  src/collision/prrtc_env_from_json.hpp
  src/collision/prrtc_env_cache.hpp
)

set(HJCDIK_COMMON_SOURCES
  src/hjcd_kernel.cu
  src/collision/prrtc_env_from_json.cpp
  ${HJCDIK_HEADERS}
)

set(HJCDIK_INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}/external/GRiD
  ${CMAKE_SOURCE_DIR}/external/GRiD/include
  ${CMAKE_SOURCE_DIR}/external/GLASS
  ${CMAKE_BINARY_DIR}
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include/test_cuh
  ${CMAKE_SOURCE_DIR}/src
)

if(MSVC)
  set(HJCDIK_WARN_FLAGS $<$<COMPILE_LANGUAGE:CXX>:/permissive- /W4>)
else()
  set(HJCDIK_WARN_FLAGS $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
endif()

set(HJCDIK_CUDA_FLAGS
  $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fcommon>
)

# Native Executable
add_executable(hjcdik
  src/main.cpp
  ${HJCDIK_COMMON_SOURCES}
)
target_sources(hjcdik PRIVATE ${GRID_HEADER})
target_include_directories(hjcdik PRIVATE ${HJCDIK_INCLUDE_DIRS})
target_compile_options(hjcdik PRIVATE ${HJCDIK_WARN_FLAGS} ${HJCDIK_CUDA_FLAGS})

set_target_properties(hjcdik PROPERTIES CUDA_RUNTIME_LIBRARY Static)

target_link_libraries(hjcdik PRIVATE
  CUDA::cudart
  Eigen3::Eigen
)

# Python
if(BUILD_PYTHON)
  pybind11_add_module(_hjcdik MODULE
    src/pybind_module.cpp
    ${HJCDIK_COMMON_SOURCES}
  )
  target_sources(_hjcdik PRIVATE ${GRID_HEADER})
  target_include_directories(_hjcdik PRIVATE ${HJCDIK_INCLUDE_DIRS})
  target_compile_options(_hjcdik PRIVATE ${HJCDIK_WARN_FLAGS} ${HJCDIK_CUDA_FLAGS})

  set_target_properties(_hjcdik PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    CUDA_RUNTIME_LIBRARY Shared
  )

  target_link_libraries(_hjcdik PRIVATE
    CUDA::cudart
    Eigen3::Eigen
  )

  install(TARGETS _hjcdik
    LIBRARY DESTINATION hjcdik
    RUNTIME DESTINATION hjcdik
    ARCHIVE DESTINATION hjcdik
  )
endif()

