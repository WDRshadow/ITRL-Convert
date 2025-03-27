cmake_minimum_required(VERSION 3.18)

project(cuda LANGUAGES CXX CUDA)

enable_testing()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 72)

add_subdirectory(tests)

add_library(cuda
        STATIC
        src/bayer2yuyv.cu
        src/rgb2yuyv.cu
        src/cuda_stream.cu
        src/bayer2rgb.cu
        src/bayer2nv12.cu
)

target_include_directories(cuda
        PUBLIC
        include
)

set_target_properties(cuda PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

add_executable(cuda_encoder
        cuda_encoder.cu
        src/cuda_stream.cu
        src/bayer2nv12.cu
)

target_link_directories(cuda_encoder
        PRIVATE
        /home/rcve/FleetMQ/Video_Codec_SDK/Lib/linux/stubs/aarch64
)

target_include_directories(cuda_encoder
        PRIVATE
        include
        /home/rcve/FleetMQ/Video_Codec_SDK/Interface
        /usr/include/spinnaker
)

target_link_libraries(cuda_encoder cuda Spinnaker)
