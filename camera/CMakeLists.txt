cmake_minimum_required(VERSION 3.18)

project(camera)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(
        include
        ${OpenCV_INCLUDE_DIRS}
)

link_libraries(${OpenCV_LIBS})

add_library(camera
        STATIC
        src/fisheye.cpp
        src/homography.cpp
        src/util.cpp
        src/component.cpp
)

set_target_properties(camera PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

add_executable(fisheye_criteria
        src/fisheye.cpp
        fisheye_criteria.cpp
)

add_executable(fisheye_undistort
        src/fisheye.cpp
        fisheye_undistort.cpp
)

add_executable(homography_criteria
        src/homography.cpp
        homography_criteria.cpp
)
