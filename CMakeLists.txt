cmake_minimum_required(VERSION 3.17)

project(rcve-hmi LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)

set(SPINNAKER_LIBRARY_DIR "/opt/spinnaker/lib")

add_executable(rcve-hmi main.cpp)

add_subdirectory(spinnaker_stream)
add_subdirectory(camera)
add_subdirectory(rgb2yuyv)
add_subdirectory(sensor)
add_subdirectory(demo)

target_link_libraries(rcve-hmi ${OpenCV_LIBS} ${SPINNAKER_LIBRARY_DIR} camera spinnaker_stream sensor rgb2yuyv)
