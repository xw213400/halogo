aux_source_directory(. TF_SRCS)
aux_source_directory(.. SHARE_SRCS)

link_directories(../lib)

message(STATUS "TF_SRCS = ${TF_SRCS}")

add_executable (xtftest ${SHARE_SRCS} ${TF_SRCS})
TARGET_LINK_LIBRARIES(xtftest tensorflow_cc tensorflow_framework)