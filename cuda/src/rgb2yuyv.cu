#include <cuda_runtime.h>
#include <stdio.h>

#include "rgb2yuyv.h"

#define Y_R 16896 // 66 * 256
#define Y_G 33024 // 129 * 256
#define Y_B 6400  // 25 * 256

#define U_R -9728  // -38 * 256
#define U_G -18944 // -74 * 256
#define U_B 28672  // 112 * 256

#define V_R 28672  // 112 * 256
#define V_G -24064 // -94 * 256
#define V_B -4608  // -18 * 256

#define CLAMP(x) ((x) < 0 ? 0 : ((x) > 255 ? 255 : x))

unsigned char *d_rgb24 = nullptr;
unsigned char *d_yuyv422 = nullptr;

__global__ void convert_rgb24_to_yuyv_cuda_kernel(const unsigned char *rgb24, unsigned char *yuyv422, unsigned int width, unsigned int height)
{
    int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index_rgb = (y * width + x) * 3;
        int index_yuyv = (y * width + x) * 2;

        unsigned char r0 = rgb24[index_rgb];
        unsigned char g0 = rgb24[index_rgb + 1];
        unsigned char b0 = rgb24[index_rgb + 2];
        unsigned char r1 = rgb24[index_rgb + 3];
        unsigned char g1 = rgb24[index_rgb + 4];
        unsigned char b1 = rgb24[index_rgb + 5];

        unsigned char y0 = CLAMP(((Y_R * r0 + Y_G * g0 + Y_B * b0) >> 16) + 16);
        unsigned char y1 = CLAMP(((Y_R * r1 + Y_G * g1 + Y_B * b1) >> 16) + 16);
        unsigned char u = CLAMP(((U_R * r0 + U_G * g0 + U_B * b0) >> 16) + 128);
        unsigned char v = CLAMP(((V_R * r0 + V_G * g0 + V_B * b0) >> 16) + 128);

        yuyv422[index_yuyv] = y0;
        yuyv422[index_yuyv + 1] = u;
        yuyv422[index_yuyv + 2] = y1;
        yuyv422[index_yuyv + 3] = v;
    }
}

void convert_rgb24_to_yuyv_cuda(const unsigned char *rgb24, unsigned char *yuyv422, unsigned int width, unsigned int height)
{
    static size_t size_rgb24 = width * height * 3 * sizeof(unsigned char);
    static size_t size_yuyv422 = width * height * 2 * sizeof(unsigned char);
    static dim3 blockSize(32, 16);
    static dim3 gridSize((width / 2 + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    if (d_rgb24 == nullptr)
    {
        cudaMalloc((void **)&d_rgb24, size_rgb24);
        cudaMalloc((void **)&d_yuyv422, size_yuyv422);
    }
    cudaMemcpy(d_rgb24, rgb24, width * height * 3, cudaMemcpyHostToDevice);
    convert_rgb24_to_yuyv_cuda_kernel<<<gridSize, blockSize>>>(d_rgb24, d_yuyv422, width, height);
    cudaMemcpy(yuyv422, d_yuyv422, size_yuyv422, cudaMemcpyDeviceToHost);
}

void cleanup_cuda_buffers()
{
    if (d_rgb24)
    {
        cudaFree(d_rgb24);
        cudaFree(d_yuyv422);
        d_rgb24 = nullptr;
        d_yuyv422 = nullptr;
    }
}

unsigned char *get_cuda_buffer(size_t size) 
{
    unsigned char *h_pinned = nullptr;
    cudaMallocHost((void **)&h_pinned, size);
    return h_pinned;
}

void free_cuda_buffer(unsigned char *h_pinned) 
{
    if (h_pinned) {
        cudaFreeHost(h_pinned);
    }
}
