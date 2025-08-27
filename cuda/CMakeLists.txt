cmake_minimum_required(VERSION 3.18)

project(cuda LANGUAGES CXX CUDA)

enable_testing()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 72)

add_library(cuda
        STATIC
        formatting.cu
        resolution.cu
)

add_subdirectory(tests)

target_include_directories(cuda
        PUBLIC
        .
)

set_target_properties(cuda PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)
