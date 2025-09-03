cmake_minimum_required(VERSION 3.30)

project(SVEA-Stream LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "CMake is in Debug mode — enabling CUDA debug flags (-G -g -O0)")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -O0")
endif()

add_subdirectory(cuda)
add_subdirectory(motion)
add_subdirectory(camera)
add_subdirectory(sensor)
add_subdirectory(zed_stream)

add_executable(svea_stream main.cpp)

target_include_directories(svea_stream
        PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/camera/include
)

find_library(NVJPEG_LIB nvjpeg
    PATHS /usr/lib/aarch64-linux-gnu/nvidia /usr/lib/aarch64-linux-gnu
)

if(NVJPEG_LIB)
    message(STATUS "Found nvjpeg: ${NVJPEG_LIB}")
else()
    message(FATAL_ERROR "nvjpeg library not found")
endif()

target_link_libraries(svea_stream
    camera
    zed_stream
    cuda
    sensor
    motion
    ${NVJPEG_LIB}
)
