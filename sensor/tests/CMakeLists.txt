include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.16.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

add_subdirectory(server)

add_executable(socket_bridge_test
        socket_bridge_test.cpp
)

target_include_directories(socket_bridge_test
        PRIVATE
        server
)

target_link_libraries(socket_bridge_test gtest_main sensor udp_server_lib)

add_test(NAME socket_bridge_test COMMAND socket_bridge_test)
