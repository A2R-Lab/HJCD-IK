cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND "$ENV{CUDAHOSTCXX}" STREQUAL "")
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

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
endif()

project(hjcdik LANGUAGES CXX CUDA)
message(STATUS "HJCD-IK: CUDA architectures = ${CMAKE_CUDA_ARCHITECTURES}")

option(HJCDIK_WARNINGS_AS_ERRORS "Treat C++ front-end warnings as errors" OFF)
option(BUILD_PYTHON "Build the Python extension module" ON)
option(HJCDIK_BUILD_NATIVE_TESTS "Build native API correctness tests" OFF)
option(HJCDIK_AUTO_CODEGEN "Regenerate grid.cuh from the URDF via GRiD codegen before building" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# Note: the CLI (csrc/kernel/main.cpp) emits its results YAML by hand via std::ofstream and
# nothing in the project includes the yaml-cpp library, so it is not a dependency.

set(PYBIND11_FINDPYTHON ON)
if(BUILD_PYTHON)
  find_package(pybind11 CONFIG REQUIRED)
endif()

# GRiD kinematics header. The committed, build-default header lives at
# csrc/generated/grid.cuh and is #included as "grid.cuh" (that directory is on the
# include path below). Regenerate it with scripts/codegen/generate_grid.py, or opt into a build-time
# regenerate with -DHJCDIK_AUTO_CODEGEN=ON (uses the external/GRiD submodule's codegen).
set(HJCDIK_GRID_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/csrc/generated/grid.cuh"
    CACHE FILEPATH "Generated robot header (must be named grid.cuh)")
get_filename_component(GRID_HEADER "${HJCDIK_GRID_HEADER}" ABSOLUTE
    BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
get_filename_component(GRID_HEADER_DIR "${GRID_HEADER}" DIRECTORY)
get_filename_component(GRID_HEADER_NAME "${GRID_HEADER}" NAME)
if(NOT GRID_HEADER_NAME STREQUAL "grid.cuh")
  message(FATAL_ERROR "HJCDIK_GRID_HEADER must name a generated grid.cuh file")
endif()
if(HJCDIK_AUTO_CODEGEN AND NOT GRID_HEADER STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/csrc/generated/grid.cuh")
  message(FATAL_ERROR "Generate custom headers explicitly; HJCDIK_AUTO_CODEGEN is the default Panda workflow")
endif()
set(GRID_CODEGEN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/scripts/codegen/generate_grid.py")
set(GRID_CODEGEN_URDF   "${CMAKE_CURRENT_SOURCE_DIR}/csrc/urdf/panda.urdf")
# Panda's collision meshes don't resolve on-disk, so the grid_collision spheres come from the
# checked-in foam spherized URDF (the paper's 59-sphere model) rather than the mesh spherizer.
set(GRID_CODEGEN_SPHERIZED "${CMAKE_CURRENT_SOURCE_DIR}/external/foam/assets/panda/smaller_panda_spherized.urdf")

if(HJCDIK_AUTO_CODEGEN)
  if(NOT EXISTS "${GRID_CODEGEN_SCRIPT}")
    message(FATAL_ERROR "HJCD-IK: codegen requested, but ${GRID_CODEGEN_SCRIPT} is missing")
  endif()
  message(STATUS "HJCD-IK: regenerating grid.cuh via ${GRID_CODEGEN_SCRIPT}")
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} "${GRID_CODEGEN_SCRIPT}" "${GRID_CODEGEN_URDF}"
            -t panda_grasptarget_hand -o "${GRID_HEADER}"
            --collision --spherized-urdf "${GRID_CODEGEN_SPHERIZED}"
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
  csrc/kernel/hjcd_settings.h
  csrc/kernel/device_utils.cuh
  csrc/kernel/grid_env.cuh
)

set(HJCDIK_COMMON_SOURCES
  csrc/kernel/hjcd_kernel.cu
  ${HJCDIK_HEADERS}
)

set(HJCDIK_INCLUDE_DIRS
  ${GRID_HEADER_DIR}   # custom robot header takes precedence over the committed Panda header
  ${CMAKE_CURRENT_SOURCE_DIR}/external/GRiD
  ${CMAKE_CURRENT_SOURCE_DIR}/external/GRiD/grid_codegen/collision   # grid_collision_geometry.cuh (SDF primitives, bare-included by grid.cuh)
  ${CMAKE_CURRENT_SOURCE_DIR}/external/GLASS
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/generated
)

if(MSVC)
  set(HJCDIK_WARN_FLAGS $<$<COMPILE_LANGUAGE:CXX>:/permissive- /W4>)
else()
  set(HJCDIK_WARN_FLAGS $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
endif()
if(HJCDIK_WARNINGS_AS_ERRORS)
  if(MSVC)
    list(APPEND HJCDIK_WARN_FLAGS $<$<COMPILE_LANGUAGE:CXX>:/WX>)
  else()
    list(APPEND HJCDIK_WARN_FLAGS $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
  endif()
endif()

set(HJCDIK_CUDA_FLAGS
  $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3>
)

# Compile the CUDA solver once and reuse the object code in both front ends. Keeping bindings and
# CLI code out of this target means ordinary wrapper edits do not trigger the expensive CUDA compile.
add_library(hjcdik_core OBJECT ${HJCDIK_COMMON_SOURCES})
target_sources(hjcdik_core PRIVATE ${GRID_HEADER})
target_include_directories(hjcdik_core PUBLIC ${HJCDIK_INCLUDE_DIRS})
target_compile_options(hjcdik_core PRIVATE ${HJCDIK_WARN_FLAGS} ${HJCDIK_CUDA_FLAGS})
target_link_libraries(hjcdik_core PUBLIC nlohmann_json::nlohmann_json CUDA::cudart Threads::Threads)
set_target_properties(hjcdik_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF
  CUDA_RUNTIME_LIBRARY Shared
  POSITION_INDEPENDENT_CODE ON
)

# Native executable
add_executable(hjcdik csrc/kernel/main.cpp)
target_compile_options(hjcdik PRIVATE ${HJCDIK_WARN_FLAGS})
target_link_libraries(hjcdik PRIVATE hjcdik_core)
set_target_properties(hjcdik PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

if(HJCDIK_BUILD_NATIVE_TESTS)
  enable_testing()
  add_executable(hjcdik_native_tests tests/native/test_api.cpp)
  target_compile_options(hjcdik_native_tests PRIVATE ${HJCDIK_WARN_FLAGS})
  target_link_libraries(hjcdik_native_tests PRIVATE hjcdik_core)
  set_target_properties(hjcdik_native_tests PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
  add_test(NAME native_api COMMAND hjcdik_native_tests)
endif()

# Python extension
if(BUILD_PYTHON)
  # Reconfigure when the selected model changes so Python metadata cannot retain a stale hash.
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${GRID_HEADER}")
  file(SHA256 "${GRID_HEADER}" HJCDIK_GRID_SHA256)
  pybind11_add_module(_hjcdik MODULE csrc/bindings/pybind_module.cpp)
  target_compile_definitions(_hjcdik PRIVATE
    HJCDIK_GRID_SHA256="${HJCDIK_GRID_SHA256}"
    HJCDIK_CUDA_COMPILER_VERSION="${CMAKE_CUDA_COMPILER_VERSION}"
  )
  target_compile_options(_hjcdik PRIVATE ${HJCDIK_WARN_FLAGS})
  target_link_libraries(_hjcdik PRIVATE hjcdik_core)
  set_target_properties(_hjcdik PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

  install(TARGETS _hjcdik
    LIBRARY DESTINATION hjcdik
    RUNTIME DESTINATION hjcdik
    ARCHIVE DESTINATION hjcdik
  )
endif()
