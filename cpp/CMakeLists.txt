PROJECT (halogo)

CMAKE_MINIMUM_REQUIRED (VERSION 3.13)

ADD_DEFINITIONS(-std=c++11)

INCLUDE_DIRECTORIES (include)

AUX_SOURCE_DIRECTORY(src DIR_SRCS)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

SET(CMAKE_BUILD_TYPE "Release")

SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")  
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

set(xrecord_sources ${DIR_SRCS})
list(REMOVE_ITEM xrecord_sources src/xdebug.cpp)
message(STATUS "xrecord_sources = ${xrecord_sources}")

set(xdebug_sources ${DIR_SRCS})
list(REMOVE_ITEM xdebug_sources src/xrecord.cpp)
message(STATUS "xdebug_sources = ${xdebug_sources}")

ADD_EXECUTABLE (xrecord ${xrecord_sources})
ADD_EXECUTABLE (xdebug ${xdebug_sources})