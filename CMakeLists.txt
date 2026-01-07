cmake_minimum_required(VERSION 3.23)
project(hjcdik LANGUAGES CXX CUDA)

option(HJCDIK_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
option(BUILD_PYTHON "Build the Python extension module" ON)
option(HJCDIK_AUTO_CODEGEN "Automatically generate grid.cuh using GRiD if missing" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86;89)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)

set(PYBIND11_FINDPYTHON ON)
if(BUILD_PYTHON)
  find_package(pybind11 CONFIG REQUIRED)
endif()

# GRiD
set(GRID_DIR              ${CMAKE_SOURCE_DIR}/external/GRiD)
set(GRID_HEADER_GENERATED ${GRID_DIR}/grid.cuh)
set(GRID_HEADER_FALLBACK  ${CMAKE_SOURCE_DIR}/include/test_cuh/panda_grid.cuh)

if(EXISTS "${GRID_DIR}/CMakeLists.txt" AND EXISTS "${GRID_HEADER_GENERATED}")
  add_subdirectory(${GRID_DIR} EXCLUDE_FROM_ALL)
  set(HAVE_GRID_TARGETS TRUE)
else()
  set(HAVE_GRID_TARGETS FALSE)
endif()

if(NOT EXISTS "${GRID_HEADER_GENERATED}" AND HJCDIK_AUTO_CODEGEN)
  message(STATUS "HJCD-IK: grid.cuh missing — running GRiD codegen")
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(HJCDIK_CODEGEN_URDF "${CMAKE_SOURCE_DIR}/include/test_urdf/panda.urdf")

  execute_process(
    COMMAND ${Python3_EXECUTABLE} generateGRiD.py "${HJCDIK_CODEGEN_URDF}" panda
    WORKING_DIRECTORY ${GRID_DIR}
    RESULT_VARIABLE _codegen_result
  )
  if(NOT _codegen_result EQUAL 0)
    message(FATAL_ERROR "GRiD codegen failed")
  endif()
endif()

if(EXISTS "${GRID_HEADER_GENERATED}")
  message(STATUS "HJCD-IK: using generated GRiD header")
  set(GRID_HEADER "${GRID_HEADER_GENERATED}")
  add_compile_definitions(HJCDIK_HAS_GENERATED_GRID_CUH=1)
else()
  message(WARNING "HJCD-IK: grid.cuh missing — using fallback")
  set(GRID_HEADER "${GRID_HEADER_FALLBACK}")
  add_compile_definitions(HJCDIK_HAS_GENERATED_GRID_CUH=0)
endif()

if(NOT EXISTS "${GRID_HEADER_GENERATED}")
  set(GRID_HEADER_SHIM "${CMAKE_BINARY_DIR}/grid.cuh")
  file(WRITE "${GRID_HEADER_SHIM}"
"#pragma once\n"
"#include \"test_cuh/panda_grid.cuh\"\n"
  )
endif()

# Sources
set(HJCDIK_HEADERS
  include/hjcd_settings.h
  include/device_utils.cuh
  include/math.cuh
)

set(HJCDIK_COMMON_SOURCES
  src/hjcd_kernel.cu
  src/util.cpp
  ${HJCDIK_HEADERS}
)

set(HJCDIK_INCLUDE_DIRS
  ${CMAKE_BINARY_DIR}
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/external/GRiD
  ${CMAKE_SOURCE_DIR}/external/GRiD/include
)

if(MSVC)
  set(HJCDIK_WARN_FLAGS $<$<COMPILE_LANGUAGE:CXX>:/permissive- /W4>)
else()
  set(HJCDIK_WARN_FLAGS $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
endif()

set(HJCDIK_CUDA_FLAGS
  $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3>
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
  yaml-cpp::yaml-cpp
)

if(HAVE_GRID_TARGETS)
  if(TARGET grid::grid)
    target_link_libraries(hjcdik PRIVATE grid::grid)
  elseif(TARGET GRiD)
    target_link_libraries(hjcdik PRIVATE GRiD)
  endif()
endif()

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
    yaml-cpp::yaml-cpp
  )

  if(HAVE_GRID_TARGETS)
    if(TARGET grid::grid)
      target_link_libraries(_hjcdik PRIVATE grid::grid)
    elseif(TARGET GRiD)
      target_link_libraries(_hjcdik PRIVATE GRiD)
    endif()
  endif()

  install(TARGETS _hjcdik
    LIBRARY DESTINATION hjcdik
    RUNTIME DESTINATION hjcdik
    ARCHIVE DESTINATION hjcdik
  )
endif()

