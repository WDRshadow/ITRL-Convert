cmake_minimum_required(VERSION 3.18)

project(udp_server)

set(CMAKE_CXX_STANDARD 17)

add_library(udp_server_lib
        STATIC
        udp_server.cpp
)

add_executable(udp_server
        start.cpp
        udp_server.cpp
)
