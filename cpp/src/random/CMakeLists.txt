aux_source_directory(. RANDOM_SRCS)
aux_source_directory(.. SHARE_SRCS)

if (${TARGET_TYPE} STREQUAL "record")
    list(REMOVE_ITEM RANDOM_SRCS ./xdebug.cpp)
else()
    list(REMOVE_ITEM RANDOM_SRCS ./xrecord.cpp)
endif()

message(STATUS "RANDOM_SRCS = ${RANDOM_SRCS}")
add_executable (xrecord ${SHARE_SRCS} ${RANDOM_SRCS})