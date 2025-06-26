add_library(cuda
        STATIC
        formatting.cu
)

target_include_directories(cuda
        PUBLIC
        .
)
