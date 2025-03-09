cmake_minimum_required(VERSION 3.18)

project(RCVE-Stream LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

link_libraries(Spinnaker)

add_executable(rcve_stream main.cpp)

add_subdirectory(spinnaker_stream)
add_subdirectory(rgb2yuyv)

target_link_libraries(rcve_stream spinnaker_stream rgb2yuyv Spinnaker)
