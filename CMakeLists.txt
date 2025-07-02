cmake_minimum_required(VERSION 3.31)

project(SVEA-Stream LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "CMake is in Debug mode — enabling CUDA debug flags (-G -g -O0)")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -O0")
endif()

find_package(OpenCV REQUIRED)

add_subdirectory(zed_stream)
add_subdirectory(camera)
add_subdirectory(cuda)
add_subdirectory(sensor)
add_subdirectory(motion)

add_executable(svea_stream main.cpp)

target_include_directories(svea_stream
        PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/camera/include
)

target_link_libraries(svea_stream camera zed_stream cuda)
