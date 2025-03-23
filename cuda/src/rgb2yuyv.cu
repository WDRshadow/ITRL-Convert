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

bool is_cuda_initialized = false;
int stream_num_;
cudaStream_t *streams = nullptr;
unsigned int width_;
unsigned int height_;
unsigned int block_height;
size_t size_rgb24_block;
size_t size_yuyv422_block;
unsigned char *d_rgb24 = nullptr;
unsigned char *d_yuyv422 = nullptr;

const dim3 blockSize(32, 16);
dim3 gridSize;

__global__ void rgb2yuyv_kernel(const unsigned char *rgb24, unsigned char *yuyv422, unsigned int width, unsigned int height)
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
        unsigned char u = CLAMP(((((U_R * r0 + U_G * g0 + U_B * b0) + (U_R * r1 + U_G * g1 + U_B * b1)) / 2) >> 16) + 128);
        unsigned char v = CLAMP(((((V_R * r0 + V_G * g0 + V_B * b0) + (V_R * r1 + V_G * g1 + V_B * b1)) / 2) >> 16) + 128);

        yuyv422[index_yuyv] = y0;
        yuyv422[index_yuyv + 1] = u;
        yuyv422[index_yuyv + 2] = y1;
        yuyv422[index_yuyv + 3] = v;
    }
}

void init_rgb2yuyv_cuda(unsigned int width, unsigned int height, int stream_num)
{
    if (is_cuda_initialized)
    {
        cleanup_rgb2yuyv_cuda();
    }
    width_ = width;
    height_ = height;
    stream_num_ = stream_num;
    block_height = height / stream_num;
    size_rgb24_block = width * block_height * 3;
    size_yuyv422_block = width * block_height * 2;
    gridSize = dim3((width / 2 + blockSize.x - 1) / blockSize.x, (block_height + blockSize.y - 1) / blockSize.y);
    streams = (cudaStream_t *)malloc(stream_num * sizeof(cudaStream_t));
    for (int i = 0; i < stream_num; i++)
    {
        cudaStreamCreate(&streams[i]);
    }
    cudaMalloc((void **)&d_rgb24, width * height * 3);
    cudaMalloc((void **)&d_yuyv422, width * height * 2);
    is_cuda_initialized = true;
}

void rgb2yuyv_cuda(const unsigned char *rgb24, unsigned char *yuyv422)
{
    if (!is_cuda_initialized)
    {
        return;
    }
    for (int i = 0; i < stream_num_; i++)
    {
        cudaMemcpyAsync(
            d_rgb24 + i * size_rgb24_block,
            rgb24 + i * size_rgb24_block,
            size_rgb24_block,
            cudaMemcpyHostToDevice,
            streams[i]);

        rgb2yuyv_kernel<<<gridSize, blockSize, 0, streams[i]>>>(
            d_rgb24 + i * size_rgb24_block,
            d_yuyv422 + i * size_yuyv422_block,
            width_,
            block_height);

        cudaMemcpyAsync(
            yuyv422 + i * size_yuyv422_block,
            d_yuyv422 + i * size_yuyv422_block,
            size_yuyv422_block,
            cudaMemcpyDeviceToHost,
            streams[i]);
    }
    cudaDeviceSynchronize();
}

void cleanup_rgb2yuyv_cuda()
{
    if (is_cuda_initialized)
    {
        for (int i = 0; i < stream_num_; i++)
        {
            cudaStreamDestroy(streams[i]);
        }
        free(streams);
        streams = nullptr;
        cudaFree(d_rgb24);
        d_rgb24 = nullptr;
        cudaFree(d_yuyv422);
        d_yuyv422 = nullptr;
        is_cuda_initialized = false;
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
    if (h_pinned)
    {
        cudaFreeHost(h_pinned);
    }
}
