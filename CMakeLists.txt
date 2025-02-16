cmake_minimum_required(VERSION 3.10)
project(ITRL_Convert LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

add_executable(ITRL_Convert test.cpp
        convert_rgb24_to_yuyv.h
        convert_rgb24_to_yuyv_neon.h
        convert_rgb24_to_yuyv_cuda.h
        convert_rgb24_to_yuyv_cuda.cu
)

target_link_libraries(test PRIVATE CUDA::cudart)
