cmake_minimum_required(VERSION 3.18)

project(RCVE-Stream LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 72)

link_libraries(Spinnaker)

add_subdirectory(spinnaker_stream)
add_subdirectory(cuda)

add_executable(rcve_stream main.cpp)

target_link_libraries(rcve_stream spinnaker_stream cuda)
