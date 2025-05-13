cmake_minimum_required(VERSION 3.18)

project(cuda LANGUAGES CXX CUDA)

enable_testing()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 72)

add_subdirectory(tests)

add_library(cuda_l
        STATIC
        src/formatting.cu
)

target_include_directories(cuda_l
        PUBLIC
        include
)

set_target_properties(cuda_l PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

# cuda_encoder executable file ==============================

find_package(OpenCV REQUIRED)

set(TOP_DIR "/usr/src/jetson_multimedia_api")
set(TEGRA_ARMABI "aarch64-linux-gnu")
file(GLOB COMMON_SOURCES ${TOP_DIR}/samples/common/classes/*.cpp)

add_executable(cuda_encoder
        cuda_encoder.cu
        src/h265_encoder.cpp
        src/formatting.cu
        src/bayer2nv12.cu
        ${COMMON_SOURCES}
)

target_link_directories(cuda_encoder
        PRIVATE
        /usr/lib/${TEGRA_ARMABI}
        /usr/lib/${TEGRA_ARMABI}/nvidia
        /usr/lib/${TEGRA_ARMABI}/tegra
)

target_include_directories(cuda_encoder
        PRIVATE
        include
        /usr/include/spinnaker
        /usr/include/${TEGRA_ARMABI}
        /usr/include/libdrm
        ${OpenCV_INCLUDE_DIRS}
        ${TOP_DIR}/include
        ${TOP_DIR}/include/libjpeg-8b
)

target_link_libraries(cuda_encoder 
        Spinnaker 
        ${OpenCV_LIBS}
        pthread
        nvv4l2
        EGL
        GLESv2
        X11
        nvbufsurface
        nvbufsurftransform
        nvjpeg
        nvosd
        drm
        cuda
        cudart
        vulkan
)
