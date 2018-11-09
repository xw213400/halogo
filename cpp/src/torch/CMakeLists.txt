aux_source_directory(. TORCH_SRCS)
aux_source_directory(.. SHARE_SRCS)

list(APPEND CMAKE_PREFIX_PATH ../../third_party/libtorch)
find_package(Torch REQUIRED)

link_directories(../../third_party/libtorch/lib)

message(STATUS "TORCH_SRCS = ${TORCH_SRCS}")
message(STATUS "TORCH_LIBRARIES = ${TORCH_LIBRARIES}")

add_executable (xtorch ${SHARE_SRCS} ${TORCH_SRCS})
target_link_libraries(xtorch "${TORCH_LIBRARIES}")