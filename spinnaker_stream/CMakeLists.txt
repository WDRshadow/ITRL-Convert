cmake_minimum_required(VERSION 3.18)

project(spinnaker_stream)

set(CMAKE_CXX_STANDARD 17)

add_library(spinnaker_stream
        STATIC
        src/spinnaker_stream.cpp
)

target_include_directories(spinnaker_stream
        PUBLIC
        include
        /usr/include/spinnaker
        ${CMAKE_SOURCE_DIR}/sensor/include
        ${CMAKE_SOURCE_DIR}/rgb2yuyv/include
        ${CMAKE_SOURCE_DIR}/camera/include
)

set_target_properties(spinnaker_stream PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)
