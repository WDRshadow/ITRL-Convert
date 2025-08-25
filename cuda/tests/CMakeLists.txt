include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.16.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

add_executable(cuda_test
        cuda_test.cpp
)

target_include_directories(cuda_test
        PRIVATE
        ../include
)

target_link_libraries(cuda_test gtest_main cuda)

add_test(NAME cuda_test COMMAND cuda_test)
