cmake_minimum_required(VERSION 3.18)

project(spinnaker_stream)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_library(spinnaker_stream
        STATIC
        spinnaker_stream.cpp
)

target_include_directories(spinnaker_stream
        PUBLIC
        .
        /opt/spinnaker/include
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/sensor/include
        ${CMAKE_SOURCE_DIR}/cuda
        ${CMAKE_SOURCE_DIR}/camera/include
)

set_target_properties(spinnaker_stream PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)
