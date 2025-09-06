cmake_minimum_required(VERSION 3.22)
project(hjcdik LANGUAGES CXX CUDA)

# ---------- Options ----------
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(HJCDIK_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

# ---------- Standards ----------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

# ---------- Dependencies ----------
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
set(HJCDIK_SOURCES
  src/main.cpp
  src/util.cpp
  src/hjcd_kernel.cu
)

add_executable(hjcdik ${HJCDIK_SOURCES})

# ---------- Include paths ----------
target_include_directories(hjcdik
  PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${GRID_DIR}/include
)

# ---------- Per-language compile options ----------
if(MSVC)
  set(CXX_WARNINGS /permissive- /W4 /source-charset:utf-8 /execution-charset:utf-8)
  if(HJCDIK_WARNINGS_AS_ERRORS)
    list(APPEND CXX_WARNINGS /WX)
  endif()
else()
  set(CXX_WARNINGS -Wall -Wextra -Wpedantic)
  if(HJCDIK_WARNINGS_AS_ERRORS)
    list(APPEND CXX_WARNINGS -Werror)
  endif()
endif()

set_source_files_properties(
  src/main.cpp src/util.cpp
  PROPERTIES COMPILE_OPTIONS "${CXX_WARNINGS}"
)

# ---------- Link libraries ----------
target_link_libraries(hjcdik PRIVATE ${YAML_CPP_TARGET} CUDA::cudart)

if(HAVE_GRID_TARGETS)
  if(TARGET grid::grid)
    target_link_libraries(globeik PRIVATE grid::grid)
  elseif(TARGET GRiD)
    target_link_libraries(globeik PRIVATE GRiD)
  endif()
endif()

# ---------- Post-build: copy tests next to binary ----------
add_custom_command(TARGET hjcdik POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_SOURCE_DIR}/tests $<TARGET_FILE_DIR:hjcdik>/tests
  COMMENT "Copying test YAML files to build output directory."
)