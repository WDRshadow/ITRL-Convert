#include <cuda_runtime.h>

#include "cuda_buffer.h"

unsigned char *get_cuda_buffer(size_t size)
{
    unsigned char *h_pinned = nullptr;
    cudaMallocHost((void **)&h_pinned, size);
    return h_pinned;
}

void free_cuda_buffer(unsigned char *h_pinned)
{
    if (h_pinned)
    {
        cudaFreeHost(h_pinned);
    }
}