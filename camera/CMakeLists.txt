cmake_minimum_required(VERSION 3.18)

project(camera)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(
        include
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/motion/include
)

add_library(camera
        STATIC
        src/ring_buffer.cpp
        src/fisheye.cpp
        src/homography.cpp
        src/util.cpp
        src/component.cpp
        src/gamma.cpp
)

target_link_libraries(camera
        PUBLIC
        ${OpenCV_LIBS}
        motion
)

set_target_properties(camera PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)
