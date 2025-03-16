cmake_minimum_required(VERSION 3.18)

project(RCVE-Stream LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 72)

find_package(OpenCV REQUIRED)
link_libraries(Spinnaker)

add_subdirectory(spinnaker_stream)
add_subdirectory(camera)
add_subdirectory(cuda)
add_subdirectory(sensor)
add_subdirectory(demo)

add_executable(rcve_stream main.cpp)

target_include_directories(rcve_stream 
        PUBLIC 
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(rcve_stream ${OpenCV_LIBS} spinnaker_stream sensor cuda camera Spinnaker)
