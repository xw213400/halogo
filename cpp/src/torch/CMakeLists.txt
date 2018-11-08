aux_source_directory(. TORCH_SRCS)
aux_source_directory(.. SHARE_SRCS)

link_directories(../../third_party/libtorch/lib)

message(STATUS "TORCH_SRCS = ${TORCH_SRCS}")

add_executable (xtorch ${SHARE_SRCS} ${TORCH_SRCS})
TARGET_LINK_LIBRARIES(xtorch libtorch.dylib)