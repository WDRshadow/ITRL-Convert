#include <stdio.h>
#include <cuda_runtime.h>

#include "bayerRG2rgb.h"

bool is_bayerRG2rgb_cuda_initialized = false;
int bayerRG2rgb_stream_num_;
cudaStream_t *bayerRG2rgb_streams = nullptr;
unsigned int bayerRG2rgb_width_;
unsigned int bayerRG2rgb_height_;
unsigned int bayerRG2rgb_block_height;
size_t bayerRG2rgb_size_bayer_block;
size_t bayerRG2rgb_size_rgb_block;
unsigned char *bayerRG2rgb_d_bayer = nullptr;
unsigned char *bayerRG2rgb_d_rgb = nullptr;

const dim3 bayerRG2rgb_blockSize(32, 16);
dim3 bayerRG2rgb_gridSize;

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

void init_bayerRG2rgb_cuda(unsigned int width, unsigned int height, int stream_num)
{
    if (is_bayerRG2rgb_cuda_initialized)
    {
        cleanup_bayerRG2rgb_cuda();
    }
    bayerRG2rgb_width_ = width;
    bayerRG2rgb_height_ = height;
    bayerRG2rgb_stream_num_ = stream_num;
    bayerRG2rgb_block_height = height / stream_num;
    bayerRG2rgb_size_bayer_block = width * bayerRG2rgb_block_height;
    bayerRG2rgb_size_rgb_block = width * bayerRG2rgb_block_height * 3;
    bayerRG2rgb_gridSize = dim3((width + bayerRG2rgb_blockSize.x - 1) / bayerRG2rgb_blockSize.x, (bayerRG2rgb_block_height + bayerRG2rgb_blockSize.y - 1) / bayerRG2rgb_blockSize.y);
    bayerRG2rgb_streams = (cudaStream_t *)malloc(stream_num * sizeof(cudaStream_t));
    for (int i = 0; i < stream_num; i++)
    {
        cudaStreamCreate(&bayerRG2rgb_streams[i]);
    }
    cudaMalloc((void **)&bayerRG2rgb_d_bayer, width * height);
    cudaMalloc((void **)&bayerRG2rgb_d_rgb, width * height * 3);
    is_bayerRG2rgb_cuda_initialized = true;
}

__global__ void bayerRG2rgb_kernel(const unsigned char *bayer, unsigned char *rgb, int width, int height)
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

void bayerRG2rgb_cuda(const unsigned char *bayerHost, unsigned char *rgbHost)
{
    if (!is_bayerRG2rgb_cuda_initialized)
    {
        return;
    }
    for (int i = 0; i < bayerRG2rgb_stream_num_; i++)
    {
        cudaMemcpyAsync(
            bayerRG2rgb_d_bayer + i * bayerRG2rgb_size_bayer_block,
            bayerHost + i * bayerRG2rgb_size_bayer_block,
            bayerRG2rgb_size_bayer_block,
            cudaMemcpyHostToDevice,
            bayerRG2rgb_streams[i]);

        bayerRG2rgb_kernel<<<bayerRG2rgb_gridSize, bayerRG2rgb_blockSize, 0, bayerRG2rgb_streams[i]>>>(
            bayerRG2rgb_d_bayer + i * bayerRG2rgb_size_bayer_block,
            bayerRG2rgb_d_rgb + i * bayerRG2rgb_size_rgb_block,
            bayerRG2rgb_width_,
            bayerRG2rgb_block_height);

        cudaMemcpyAsync(
            rgbHost + i * bayerRG2rgb_size_rgb_block,
            bayerRG2rgb_d_rgb + i * bayerRG2rgb_size_rgb_block,
            bayerRG2rgb_size_rgb_block,
            cudaMemcpyDeviceToHost,
            bayerRG2rgb_streams[i]);
    }
    cudaDeviceSynchronize();
}

void cleanup_bayerRG2rgb_cuda()
{
    if (!is_bayerRG2rgb_cuda_initialized)
    {
        return;
    }
    for (int i = 0; i < bayerRG2rgb_stream_num_; i++)
    {
        cudaStreamDestroy(bayerRG2rgb_streams[i]);
    }
    free(bayerRG2rgb_streams);
    bayerRG2rgb_streams = nullptr;
    cudaFree(bayerRG2rgb_d_bayer);
    bayerRG2rgb_d_bayer = nullptr;
    cudaFree(bayerRG2rgb_d_rgb);
    bayerRG2rgb_d_rgb = nullptr;
    is_bayerRG2rgb_cuda_initialized = false;
}
