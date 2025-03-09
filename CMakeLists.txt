cmake_minimum_required(VERSION 3.17)

project(rcve-hmi)

add_executable(rcve-hmi main.cpp)

add_subdirectory(spinnaker_stream)
add_subdirectory(camera)
add_subdirectory(rgb2yuyv)
add_subdirectory(sensor)
#add_subdirectory(demo)

target_link_libraries(rcve-hmi camera spinnaker_stream sensor rgb2yuyv)
