aux_source_directory(. TF_SRCS)
aux_source_directory(.. SHARE_SRCS)

link_directories(../../lib)

message(STATUS "TF_SRCS = ${TF_SRCS}")

add_executable (xrecordtf ${SHARE_SRCS} ${TF_SRCS})
TARGET_LINK_LIBRARIES(xrecordtf tensorflow_cc tensorflow_framework)