PROJECT (halogo)

CMAKE_MINIMUM_REQUIRED (VERSION 3.5)

ADD_DEFINITIONS(-std=c++11)

LINK_DIRECTORIES(./lib)

INCLUDE_DIRECTORIES (
    include 
    third_party
    ../../tensorflow
    ../../tensorflow/bazel-genfiles)

AUX_SOURCE_DIRECTORY(src DIR_SRCS)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

SET(CMAKE_BUILD_TYPE "Release")

SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")  
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

set(xrecord_sources ${DIR_SRCS})
list(REMOVE_ITEM xrecord_sources src/xdebug.cpp src/xtesttf.cpp)
message(STATUS "xrecord_sources = ${xrecord_sources}")

set(xdebug_sources ${DIR_SRCS})
list(REMOVE_ITEM xdebug_sources src/xrecord.cpp src/xtesttf.cpp)
message(STATUS "xdebug_sources = ${xdebug_sources}")

set(xtesttf_sources ${DIR_SRCS})
list(REMOVE_ITEM xtesttf_sources src/xdebug.cpp src/xrecord.cpp)
message(STATUS "xtesttf_sources = ${xtesttf_sources}")

ADD_EXECUTABLE (xtesttf ${xtesttf_sources})
TARGET_LINK_LIBRARIES(xtesttf tensorflow_cc tensorflow_framework)

ADD_EXECUTABLE (xrecord ${xrecord_sources})
TARGET_LINK_LIBRARIES(xrecord tensorflow_cc tensorflow_framework)

ADD_EXECUTABLE (xdebug ${xdebug_sources})
TARGET_LINK_LIBRARIES(xdebug tensorflow_cc tensorflow_framework)