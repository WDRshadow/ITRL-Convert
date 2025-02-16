cmake_minimum_required(VERSION 3.29)
project(ITRL_Convert)

set(CMAKE_CXX_STANDARD 11)

add_executable(ITRL_Convert test.cpp
        convert_rgb24_to_yuyv.h
        convert_rgb24_to_yuyv_neon.h
)
