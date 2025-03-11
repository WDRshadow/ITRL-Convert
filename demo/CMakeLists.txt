cmake_minimum_required(VERSION 3.18)

project(demo)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(demo
        demo.cpp
        demo_sensor.cpp
)

target_include_directories(demo
        PUBLIC
        ${CMAKE_SOURCE_DIR}/camera/include
)

set_target_properties(demo PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(demo ${OpenCV_LIBS} camera)
