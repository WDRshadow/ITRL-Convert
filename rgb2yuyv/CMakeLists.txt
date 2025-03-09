cmake_minimum_required(VERSION 3.18)

project(rgb2yuyv LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)


add_library(rgb2yuyv
        STATIC
        src/rbg2yuyv.cu
)

target_include_directories(rgb2yuyv
        PUBLIC
        include
)

set_target_properties(rgb2yuyv PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)
