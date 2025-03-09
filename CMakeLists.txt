cmake_minimum_required(VERSION 3.18)

project(rcve-hmi LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
link_libraries(Spinnaker)

add_executable(rcve-hmi main.cpp)

add_subdirectory(spinnaker_stream)
add_subdirectory(camera)
add_subdirectory(rgb2yuyv)
add_subdirectory(sensor)
add_subdirectory(demo)

target_link_libraries(rcve-hmi ${OpenCV_LIBS} spinnaker_stream sensor rgb2yuyv camera Spinnaker)
