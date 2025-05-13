include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.16.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

add_executable(motion_test
        motion_test.cpp
)

target_include_directories(motion_test
        PRIVATE
        ..
)

target_link_libraries(motion_test gtest_main motion)

add_test(NAME motion_test COMMAND motion_test)
