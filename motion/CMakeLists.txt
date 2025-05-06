cmake_minimum_required(VERSION 3.18)

project(motion)

enable_testing()

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(
        include
        ${OpenCV_INCLUDE_DIRS}
)

link_libraries(${OpenCV_LIBS})

add_library(motion
        STATIC
        src/cyra.cpp
        src/kbm.cpp
)

add_subdirectory(tests)

set_target_properties(sensor PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)
