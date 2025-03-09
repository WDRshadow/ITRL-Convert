cmake_minimum_required(VERSION 3.18)

project(camera)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_library(camera
        STATIC
        src/fisheye.cpp
        src/homography.cpp
        src/util.cpp
)

target_include_directories(camera
        PUBLIC
        include
        ${OpenCV_INCLUDE_DIRS}
)

set_target_properties(camera PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(camera ${OpenCV_LIBS})
