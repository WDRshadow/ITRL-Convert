#include <stdio.h>
#include <cuda_runtime.h>

#include "bayer2rgb.h"

bool is_bayer2rgb_cuda_initialized = false;
int bayer2rgb_stream_num_;
cudaStream_t *bayer2rgb_streams = nullptr;
unsigned int bayer2rgb_width_;
unsigned int bayer2rgb_height_;
unsigned int bayer2rgb_block_height;
size_t bayer2rgb_size_bayer_block;
size_t bayer2rgb_size_rgb_block;
unsigned char *bayer2rgb_d_bayer = nullptr;
unsigned char *bayer2rgb_d_rgb = nullptr;

const dim3 bayer2rgb_blockSize(32, 16);
dim3 bayer2rgb_gridSize;

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

void init_bayer2rgb_cuda(unsigned int width, unsigned int height, int stream_num)
{
    if (is_bayer2rgb_cuda_initialized)
    {
        cleanup_bayer2rgb_cuda();
    }
    bayer2rgb_width_ = width;
    bayer2rgb_height_ = height;
    bayer2rgb_stream_num_ = stream_num;
    bayer2rgb_block_height = height / stream_num;
    bayer2rgb_size_bayer_block = width * bayer2rgb_block_height;
    bayer2rgb_size_rgb_block = width * bayer2rgb_block_height * 3;
    bayer2rgb_gridSize = dim3((width + bayer2rgb_blockSize.x - 1) / bayer2rgb_blockSize.x, (bayer2rgb_block_height + bayer2rgb_blockSize.y - 1) / bayer2rgb_blockSize.y);
    bayer2rgb_streams = (cudaStream_t *)malloc(stream_num * sizeof(cudaStream_t));
    for (int i = 0; i < stream_num; i++)
    {
        cudaStreamCreate(&bayer2rgb_streams[i]);
    }
    cudaMalloc((void **)&bayer2rgb_d_bayer, width * height);
    cudaMalloc((void **)&bayer2rgb_d_rgb, width * height * 3);
    is_bayer2rgb_cuda_initialized = true;
}

__global__ void bayer2rgb_kernel(const unsigned char *bayer, unsigned char *rgb, int width, int height)
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

void bayer2rgb_cuda(const unsigned char *bayerHost, unsigned char *rgbHost)
{
    if (!is_bayer2rgb_cuda_initialized)
    {
        return;
    }
    for (int i = 0; i < bayer2rgb_stream_num_; i++)
    {
        cudaMemcpyAsync(
            bayer2rgb_d_bayer + i * bayer2rgb_size_bayer_block,
            bayerHost + i * bayer2rgb_size_bayer_block,
            bayer2rgb_size_bayer_block,
            cudaMemcpyHostToDevice,
            bayer2rgb_streams[i]);

        bayer2rgb_kernel<<<bayer2rgb_gridSize, bayer2rgb_blockSize, 0, bayer2rgb_streams[i]>>>(
            bayer2rgb_d_bayer + i * bayer2rgb_size_bayer_block,
            bayer2rgb_d_rgb + i * bayer2rgb_size_rgb_block,
            bayer2rgb_width_,
            bayer2rgb_block_height);

        cudaMemcpyAsync(
            rgbHost + i * bayer2rgb_size_rgb_block,
            bayer2rgb_d_rgb + i * bayer2rgb_size_rgb_block,
            bayer2rgb_size_rgb_block,
            cudaMemcpyDeviceToHost,
            bayer2rgb_streams[i]);
    }
    cudaDeviceSynchronize();
}

void cleanup_bayer2rgb_cuda()
{
    if (!is_bayer2rgb_cuda_initialized)
    {
        return;
    }
    for (int i = 0; i < bayer2rgb_stream_num_; i++)
    {
        cudaStreamDestroy(bayer2rgb_streams[i]);
    }
    free(bayer2rgb_streams);
    bayer2rgb_streams = nullptr;
    cudaFree(bayer2rgb_d_bayer);
    bayer2rgb_d_bayer = nullptr;
    cudaFree(bayer2rgb_d_rgb);
    bayer2rgb_d_rgb = nullptr;
    is_bayer2rgb_cuda_initialized = false;
}
