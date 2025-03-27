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

bool is_rgb2yuyv_cuda_initialized = false;
int rgb2yuyv_stream_num_;
cudaStream_t *rgb2yuyv_streams = nullptr;
unsigned int rgb2yuyv_width_;
unsigned int rgb2yuyv_height_;
unsigned int rgb2yuyv_block_height;
size_t rgb2yuyv_size_rgb_block;
size_t rgb2yuyv_size_yuyv_block;
unsigned char *rgb2yuyv_d_rgb = nullptr;
unsigned char *rgb2yuyv_d_yuyv = nullptr;

const dim3 rgb2yuyv_blockSize(32, 16);
dim3 rgb2yuyv_gridSize;

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
    if (is_rgb2yuyv_cuda_initialized)
    {
        cleanup_rgb2yuyv_cuda();
    }
    rgb2yuyv_width_ = width;
    rgb2yuyv_height_ = height;
    rgb2yuyv_stream_num_ = stream_num;
    rgb2yuyv_block_height = height / stream_num;
    rgb2yuyv_size_rgb_block = width * rgb2yuyv_block_height * 3;
    rgb2yuyv_size_yuyv_block = width * rgb2yuyv_block_height * 2;
    rgb2yuyv_gridSize = dim3((width / 2 + rgb2yuyv_blockSize.x - 1) / rgb2yuyv_blockSize.x, (rgb2yuyv_block_height + rgb2yuyv_blockSize.y - 1) / rgb2yuyv_blockSize.y);
    rgb2yuyv_streams = (cudaStream_t *)malloc(stream_num * sizeof(cudaStream_t));
    for (int i = 0; i < stream_num; i++)
    {
        cudaStreamCreate(&rgb2yuyv_streams[i]);
    }
    cudaMalloc((void **)&rgb2yuyv_d_rgb, width * height * 3);
    cudaMalloc((void **)&rgb2yuyv_d_yuyv, width * height * 2);
    is_rgb2yuyv_cuda_initialized = true;
}

void rgb2yuyv_cuda(const unsigned char *rgb24, unsigned char *yuyv422)
{
    if (!is_rgb2yuyv_cuda_initialized)
    {
        return;
    }
    for (int i = 0; i < rgb2yuyv_stream_num_; i++)
    {
        cudaMemcpyAsync(
            rgb2yuyv_d_rgb + i * rgb2yuyv_size_rgb_block,
            rgb24 + i * rgb2yuyv_size_rgb_block,
            rgb2yuyv_size_rgb_block,
            cudaMemcpyHostToDevice,
            rgb2yuyv_streams[i]);

        rgb2yuyv_kernel<<<rgb2yuyv_gridSize, rgb2yuyv_blockSize, 0, rgb2yuyv_streams[i]>>>(
            rgb2yuyv_d_rgb + i * rgb2yuyv_size_rgb_block,
            rgb2yuyv_d_yuyv + i * rgb2yuyv_size_yuyv_block,
            rgb2yuyv_width_,
            rgb2yuyv_block_height);

        cudaMemcpyAsync(
            yuyv422 + i * rgb2yuyv_size_yuyv_block,
            rgb2yuyv_d_yuyv + i * rgb2yuyv_size_yuyv_block,
            rgb2yuyv_size_yuyv_block,
            cudaMemcpyDeviceToHost,
            rgb2yuyv_streams[i]);
    }
    cudaDeviceSynchronize();
}

void cleanup_rgb2yuyv_cuda()
{
    if (!is_rgb2yuyv_cuda_initialized)
    {
        return;
    }
    for (int i = 0; i < rgb2yuyv_stream_num_; i++)
    {
        cudaStreamDestroy(rgb2yuyv_streams[i]);
    }
    free(rgb2yuyv_streams);
    rgb2yuyv_streams = nullptr;
    cudaFree(rgb2yuyv_d_rgb);
    rgb2yuyv_d_rgb = nullptr;
    cudaFree(rgb2yuyv_d_yuyv);
    rgb2yuyv_d_yuyv = nullptr;
    is_rgb2yuyv_cuda_initialized = false;
}
