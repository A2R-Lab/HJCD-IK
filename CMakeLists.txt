cmake_minimum_required(VERSION 3.23)
project(hjcdik LANGUAGES CXX CUDA)

option(HJCDIK_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
option(BUILD_PYTHON "Build the Python extension module" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# (CMake 3.18+)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

# set(CMAKE_CUDA_ARCHITECTURES 75;86;90)

find_package(CUDAToolkit REQUIRED)

set(PYBIND11_FINDPYTHON ON)
if(BUILD_PYTHON)
  find_package(pybind11 CONFIG REQUIRED)
endif()

# GRiD subdir
set(GRID_DIR ${CMAKE_SOURCE_DIR}/external/GRiD)
if(EXISTS "${GRID_DIR}/CMakeLists.txt")
  add_subdirectory(${GRID_DIR} EXCLUDE_FROM_ALL)
  set(HAVE_GRID_TARGETS TRUE)
else()
  set(HAVE_GRID_TARGETS FALSE)
endif()

# Sources
set(HJCDIK_COMMON_SOURCES
  src/hjcd_kernel.cu
  src/util.cpp
)

set(GRID_HEADER_GENERATED "${CMAKE_SOURCE_DIR}/external/GRiD/grid.cuh")
set(GRID_HEADER_FALLBACK  "${CMAKE_SOURCE_DIR}/include/test_cuh/panda_grid.cuh")

if(EXISTS "${GRID_HEADER_GENERATED}")
  message(STATUS "HJCD-IK: using generated GRiD header: ${GRID_HEADER_GENERATED}")
  set(GRID_HEADER "${GRID_HEADER_GENERATED}")
  add_compile_definitions(HJCDIK_HAS_GENERATED_GRID_CUH=1)
else()
  message(WARNING "HJCD-IK: external/GRiD/grid.cuh not found; using fallback: ${GRID_HEADER_FALLBACK}")
  if(NOT EXISTS "${GRID_HEADER_FALLBACK}")
    message(FATAL_ERROR "HJCD-IK: fallback grid header not found at ${GRID_HEADER_FALLBACK}")
  endif()
  set(GRID_HEADER "${GRID_HEADER_FALLBACK}")
  add_compile_definitions(HJCDIK_HAS_GENERATED_GRID_CUH=0)
endif()

# Native Executable
add_executable(hjcdik
  src/main.cpp
  ${HJCDIK_COMMON_SOURCES}
)

target_sources(hjcdik 
  PRIVATE ${GRID_HEADER}
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

# Python
if(BUILD_PYTHON)
  pybind11_add_module(_hjcdik MODULE
    src/pybind_module.cpp
    ${HJCDIK_COMMON_SOURCES}
  )
  target_sources(_hjcdik PRIVATE ${GRID_HEADER})
  target_include_directories(_hjcdik PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external/GRiD
    ${CMAKE_SOURCE_DIR}/external/GRiD/include
  )

  # Link STATIC CUDA runtime
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

  install(TARGETS _hjcdik DESTINATION hjcdik)

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
