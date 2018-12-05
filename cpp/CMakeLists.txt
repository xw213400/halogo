project (halogo)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++11)

set(TARGET_TYPE "record")
set(TARGET_PATH "tensorflow")
set(CMAKE_BUILD_TYPE "Release")

include_directories (/usr/local/include/eigen3 include third_party)
include_directories(../../tensorflow ../../tensorflow/bazel-genfiles ../../tensorflow/tensorflow)
#include_directories(third_party/libtorch/include)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")  
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

if (${TARGET_PATH} STREQUAL "random")
    add_subdirectory(src/random)
elseif (${TARGET_PATH} STREQUAL "tensorflow")
    add_subdirectory(src/tensorflow)
elseif (${TARGET_PATH} STREQUAL "torch")
    add_subdirectory(src/torch)
endif()
