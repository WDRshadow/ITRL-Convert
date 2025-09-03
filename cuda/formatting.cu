#include <cuda_runtime.h>
#include <stdio.h>

#include "formatting.h"

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

#define BLOCK_SIZE 32, 16

__global__ void rgb2yuyv_kernel(const unsigned char *rgb, unsigned char *yuyv, unsigned int width, unsigned int height)
{
    int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x + 1 < width && y < height)
    {
        int index_rgb = (y * width + x) * 3;
        int index_yuyv = (y * width + x) * 2;

        unsigned char r0 = rgb[index_rgb];
        unsigned char g0 = rgb[index_rgb + 1];
        unsigned char b0 = rgb[index_rgb + 2];
        unsigned char r1 = rgb[index_rgb + 3];
        unsigned char g1 = rgb[index_rgb + 4];
        unsigned char b1 = rgb[index_rgb + 5];

        unsigned char y0 = CLAMP(((Y_R * r0 + Y_G * g0 + Y_B * b0) >> 16) + 16);
        unsigned char y1 = CLAMP(((Y_R * r1 + Y_G * g1 + Y_B * b1) >> 16) + 16);
        unsigned char u = CLAMP(((((U_R * r0 + U_G * g0 + U_B * b0) + (U_R * r1 + U_G * g1 + U_B * b1)) / 2) >> 16) + 128);
        unsigned char v = CLAMP(((((V_R * r0 + V_G * g0 + V_B * b0) + (V_R * r1 + V_G * g1 + V_B * b1)) / 2) >> 16) + 128);

        yuyv[index_yuyv] = y0;
        yuyv[index_yuyv + 1] = u;
        yuyv[index_yuyv + 2] = y1;
        yuyv[index_yuyv + 3] = v;
    }
}

__global__ void bgra2rgb_kernel(const unsigned char *bgra, unsigned char *rgb, unsigned int width, unsigned int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index_bgra = (y * width + x) * 4;
        int index_rgb = (y * width + x) * 3;

        rgb[index_rgb] = bgra[index_bgra + 2];     // R (from BGRA position 2)
        rgb[index_rgb + 1] = bgra[index_bgra + 1]; // G (from BGRA position 1)
        rgb[index_rgb + 2] = bgra[index_bgra];     // B (from BGRA position 0)
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

CudaImageConverter::CudaImageConverter(unsigned int width, unsigned int height, int stream_num, int mode)
    : width(width), height(height), stream_num(stream_num),
      block_height(height / stream_num),
      size_bgra_block(width * block_height * 4),
      size_yuyv_block(width * block_height * 2),
      size_rgb_block(width * block_height * 3),
      mode(mode)
{
    blockSize = new dim3(BLOCK_SIZE);
    streams = new cudaStream_t[stream_num];
    for (int i = 0; i < stream_num; i++)
    {
        cudaStreamCreate(&streams[i]);
    }
    if (mode == D_BGRA2RGB)
    {
        gridSize = new dim3((width + blockSize->x - 1) / blockSize->x, (block_height + blockSize->y - 1) / blockSize->y);
    }
    else if (mode == RGB2YUYV)
    {
        gridSize = new dim3((width / 2 + blockSize->x - 1) / blockSize->x, (block_height + blockSize->y - 1) / blockSize->y);
        cudaMalloc((void **)&d_yuyv, width * height * 2);
    }
    cudaMalloc((void **)&d_rgb, width * height * 3);
}

CudaImageConverter::~CudaImageConverter()
{
    for (int i = 0; i < stream_num; i++)
    {
        cudaStreamDestroy(streams[i]);
    }
    free(streams);
    delete blockSize;
    delete gridSize;
    if (mode == RGB2YUYV)
        cudaFree(d_yuyv);
    cudaFree(d_rgb);
}

void CudaImageConverter::convert(const unsigned char *src, unsigned char *dst)
{
    for (int i = 0; i < stream_num; i++)
    {
        if (mode == D_BGRA2RGB)
        {
            bgra2rgb_kernel<<<*gridSize, *blockSize, 0, streams[i]>>>(
                src + i * size_bgra_block,
                d_rgb + i * size_rgb_block,
                width,
                block_height);

            cudaMemcpyAsync(
                dst + i * size_rgb_block,
                d_rgb + i * size_rgb_block,
                size_rgb_block,
                cudaMemcpyDeviceToHost,
                streams[i]);
        }
        else if (mode == RGB2YUYV)
        {
            cudaMemcpyAsync(
                d_rgb + i * size_rgb_block,
                src + i * size_rgb_block,
                size_rgb_block,
                cudaMemcpyHostToDevice,
                streams[i]);

            rgb2yuyv_kernel<<<*gridSize, *blockSize, 0, streams[i]>>>(
                d_rgb + i * size_rgb_block,
                d_yuyv + i * size_yuyv_block,
                width,
                block_height);

            cudaMemcpyAsync(
                dst + i * size_yuyv_block,
                d_yuyv + i * size_yuyv_block,
                size_yuyv_block,
                cudaMemcpyDeviceToHost,
                streams[i]);
        }
    }
    cudaDeviceSynchronize();
}
