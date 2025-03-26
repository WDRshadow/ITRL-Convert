cmake_minimum_required(VERSION 3.18)

project(motion)

enable_testing()

set(CMAKE_CXX_STANDARD 17)

add_library(motion
        STATIC
        cyra.cpp
)

add_subdirectory(tests)

set_target_properties(sensor PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)
