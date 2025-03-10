cmake_minimum_required(VERSION 3.18)

project(sensor)

set(CMAKE_CXX_STANDARD 17)

add_library(sensor
        STATIC
        src/sensor.cpp
)

target_include_directories(sensor
        PUBLIC
        include
)

set_target_properties(sensor PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)
