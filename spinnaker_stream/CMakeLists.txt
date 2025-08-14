cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)

add_library(spinnaker_stream
        STATIC
        spinnaker_stream.cpp
)

target_include_directories(spinnaker_stream
        PUBLIC
        /usr/include/spinnaker
        .
        ${CMAKE_SOURCE_DIR}/cuda
)

set_target_properties(spinnaker_stream PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)
