cmake_minimum_required(VERSION 3.18)

project(cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 72)

add_library(cuda
        STATIC
        src/rgb2yuyv.cu
        src/cuda_stream.cu
        src/bayerRG2rgb.cu
)

target_include_directories(cuda
        PUBLIC
        include
)

set_target_properties(cuda PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)
