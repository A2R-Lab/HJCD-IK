cmake_minimum_required(VERSION 3.22)
project(globeik LANGUAGES CXX CUDA)

# ---------- Options ----------
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(GLOBEIK_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

# ---------- Standards ----------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Prevent MSVC host flags from leaking into CUDA (extra safety)
# (Not strictly required after per-source fix, but good hygiene with VS+CUDA.)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

# ---------- Dependencies ----------
# yaml-cpp: prefer modern namespaced target if available
find_package(yaml-cpp QUIET CONFIG)
if(TARGET yaml-cpp::yaml-cpp)
  set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
else()
  find_package(yaml-cpp REQUIRED)  # legacy 'yaml-cpp' target on some systems
  set(YAML_CPP_TARGET yaml-cpp)
endif()

# CUDA Toolkit (cudart)
find_package(CUDAToolkit REQUIRED)

# ---------- GRiD submodule ----------
set(GRID_DIR ${CMAKE_SOURCE_DIR}/external/GRiD)
if(EXISTS "${GRID_DIR}/CMakeLists.txt")
  add_subdirectory(${GRID_DIR} EXCLUDE_FROM_ALL)
  set(HAVE_GRID_TARGETS TRUE)
else()
  set(HAVE_GRID_TARGETS FALSE)
endif()

# ---------- Sources ----------
set(GLOBEIK_SOURCES
  src/main.cpp
  src/util.cpp
  src/globeik_kernel.cu
)

add_executable(globeik ${GLOBEIK_SOURCES})

# ---------- Include paths ----------
# You include headers as "include/xyz.h", so expose project root and include/
target_include_directories(globeik
  PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${GRID_DIR}/include
)

# ---------- Per-language compile options (apply ONLY to C++ files) ----------
if(MSVC)
  set(CXX_WARNINGS /permissive- /W4 /source-charset:utf-8 /execution-charset:utf-8)
  if(GLOBEIK_WARNINGS_AS_ERRORS)
    list(APPEND CXX_WARNINGS /WX)
  endif()
else()
  set(CXX_WARNINGS -Wall -Wextra -Wpedantic)
  if(GLOBEIK_WARNINGS_AS_ERRORS)
    list(APPEND CXX_WARNINGS -Werror)
  endif()
endif()

# Apply to just the C++ files (NOT the .cu file)
set_source_files_properties(
  src/main.cpp src/util.cpp
  PROPERTIES COMPILE_OPTIONS "${CXX_WARNINGS}"
)

# ---------- Link libraries ----------
target_link_libraries(globeik PRIVATE ${YAML_CPP_TARGET} CUDA::cudart)

# If GRiD exports a target, link it too (best-effort names)
if(HAVE_GRID_TARGETS)
  if(TARGET grid::grid)
    target_link_libraries(globeik PRIVATE grid::grid)
  elseif(TARGET GRiD)
    target_link_libraries(globeik PRIVATE GRiD)
  endif()
endif()

# ---------- CUDA architectures (optional) ----------
# Uncomment and set for your GPUs, otherwise the default is fine:
# set_property(TARGET globeik PROPERTY CUDA_ARCHITECTURES 75 86)

# ---------- Post-build: copy tests next to binary ----------
add_custom_command(TARGET globeik POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_SOURCE_DIR}/tests $<TARGET_FILE_DIR:globeik>/tests
  COMMENT "Copying test YAML files to build output directory."
)