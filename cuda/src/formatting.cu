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

__device__ inline float getBayerVal(const unsigned char *bayer, int x, int y, int width, int height)
{
    if (x < 0 || x >= width || y < 0 || y >= height)
        return 0.0f;
    return static_cast<float>(bayer[y * width + x]);
}

__device__ inline unsigned char clampToByte(float val)
{
    if (val < 0.0f)
        val = 0.0f;
    if (val > 255.0f)
        val = 255.0f;
    return static_cast<unsigned char>(val + 0.5f);
}

__global__ void bayer2rgb_kernel(const unsigned char *bayer, unsigned char *rgb, unsigned int width, unsigned int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int out_idx = (y * width + x) * 3;

    bool rowEven = (y % 2 == 0);
    bool colEven = (x % 2 == 0);

    float R = 0.0f, G = 0.0f, B = 0.0f;

    if (rowEven && colEven)
    {
        // R
        R = getBayerVal(bayer, x, y, width, height);
        // G
        float g_left = getBayerVal(bayer, x - 1, y, width, height);
        float g_right = getBayerVal(bayer, x + 1, y, width, height);
        float g_up = getBayerVal(bayer, x, y - 1, width, height);
        float g_down = getBayerVal(bayer, x, y + 1, width, height);
        G = (g_left + g_right + g_up + g_down) * 0.25f;
        // B
        float b_ul = getBayerVal(bayer, x - 1, y - 1, width, height);
        float b_ur = getBayerVal(bayer, x + 1, y - 1, width, height);
        float b_dl = getBayerVal(bayer, x - 1, y + 1, width, height);
        float b_dr = getBayerVal(bayer, x + 1, y + 1, width, height);
        B = (b_ul + b_ur + b_dl + b_dr) * 0.25f;
    }
    else if (rowEven && !colEven)
    {
        // G
        G = getBayerVal(bayer, x, y, width, height);
        // R
        float r_left = getBayerVal(bayer, x - 1, y, width, height);
        float r_right = getBayerVal(bayer, x + 1, y, width, height);
        R = 0.5f * (r_left + r_right);
        // B
        float b_up = getBayerVal(bayer, x, y - 1, width, height);
        float b_down = getBayerVal(bayer, x, y + 1, width, height);
        B = 0.5f * (b_up + b_down);
    }
    else if (!rowEven && colEven)
    {
        // G
        G = getBayerVal(bayer, x, y, width, height);
        // B
        float b_left = getBayerVal(bayer, x - 1, y, width, height);
        float b_right = getBayerVal(bayer, x + 1, y, width, height);
        B = 0.5f * (b_left + b_right);
        // R
        float r_up = getBayerVal(bayer, x, y - 1, width, height);
        float r_down = getBayerVal(bayer, x, y + 1, width, height);
        R = 0.5f * (r_up + r_down);
    }
    else
    {
        // B
        B = getBayerVal(bayer, x, y, width, height);
        // G
        float g_left = getBayerVal(bayer, x - 1, y, width, height);
        float g_right = getBayerVal(bayer, x + 1, y, width, height);
        float g_up = getBayerVal(bayer, x, y - 1, width, height);
        float g_down = getBayerVal(bayer, x, y + 1, width, height);
        G = (g_left + g_right + g_up + g_down) * 0.25f;
        // R
        float r_ul = getBayerVal(bayer, x - 1, y - 1, width, height);
        float r_ur = getBayerVal(bayer, x + 1, y - 1, width, height);
        float r_dl = getBayerVal(bayer, x - 1, y + 1, width, height);
        float r_dr = getBayerVal(bayer, x + 1, y + 1, width, height);
        R = (r_ul + r_ur + r_dl + r_dr) * 0.25f;
    }

    rgb[out_idx + 0] = clampToByte(R);
    rgb[out_idx + 1] = clampToByte(G);
    rgb[out_idx + 2] = clampToByte(B);
}

__global__ void rgb2yuyv_kernel(const unsigned char *rgb, unsigned char *yuyv, unsigned int width, unsigned int height)
{
    int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
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
    : width(width), height(height), stream_num(stream_num), mode(mode),
      block_height(height / stream_num),
      size_bayer_block(width * block_height),
      size_rgb_block(width * block_height * 3),
      size_yuyv_block(width * block_height * 2)
{
    blockSize = new dim3(BLOCK_SIZE);
    streams = (cudaStream_t *)malloc(stream_num * sizeof(cudaStream_t));
    streams = new cudaStream_t[stream_num];
    for (int i = 0; i < stream_num; i++)
    {
        cudaStreamCreate(&streams[i]);
    }
    if (mode == BAYER2RGB || mode == BAYER2YUYV)
    {
        gridSize_1 = new dim3((width + blockSize->x - 1) / blockSize->x, (block_height + blockSize->y - 1) / blockSize->y);
        cudaMalloc((void **)&d_bayer, width * height);
    }
    if (mode == RGB2YUYV || mode == BAYER2YUYV)
    {
        gridSize_2 = new dim3((width / 2 + blockSize->x - 1) / blockSize->x, (block_height + blockSize->y - 1) / blockSize->y);
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
    if (mode == BAYER2RGB || mode == BAYER2YUYV)
    {
        delete gridSize_1;
        cudaFree(d_bayer);
    }
    if (mode == RGB2YUYV || mode == BAYER2YUYV)
    {
        delete gridSize_2;
        cudaFree(d_yuyv);
    }
    cudaFree(d_rgb);
}

void CudaImageConverter::convert(const unsigned char *src, unsigned char *dst)
{
    for (int i = 0; i < stream_num; i++)
    {
        if (mode == BAYER2RGB || mode == BAYER2YUYV)
        {
            cudaMemcpyAsync(
                d_bayer + i * size_bayer_block,
                src + i * size_bayer_block,
                size_bayer_block,
                cudaMemcpyHostToDevice,
                streams[i]);

            bayer2rgb_kernel<<<*gridSize_1, *blockSize, 0, streams[i]>>>(
                d_bayer + i * size_bayer_block,
                d_rgb + i * size_rgb_block,
                width,
                block_height);

            if (mode == BAYER2RGB) {
                cudaMemcpyAsync(
                    dst + i * size_rgb_block,
                    d_rgb + i * size_rgb_block,
                    size_rgb_block,
                    cudaMemcpyDeviceToHost,
                    streams[i]);
            }
        }
        else if (mode == RGB2YUYV)
        {
            cudaMemcpyAsync(
                d_rgb + i * size_rgb_block,
                src + i * size_rgb_block,
                size_rgb_block,
                cudaMemcpyHostToDevice,
                streams[i]);
        }
        if (mode == RGB2YUYV || mode == BAYER2YUYV) 
        {
            rgb2yuyv_kernel<<<*gridSize_2, *blockSize, 0, streams[i]>>>(
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