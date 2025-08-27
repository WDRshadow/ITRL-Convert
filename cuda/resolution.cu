#include "resolution.h"

#define BLOCK_SIZE_ 32, 16

__global__ void rgb_resolution_kernel(const unsigned char *src, unsigned char *dst, unsigned int new_width, unsigned int new_height, unsigned int orig_width, int scale)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < new_width && y < new_height)
    {
        int sum_r = 0, sum_g = 0, sum_b = 0;

        for (int dy = 0; dy < scale; ++dy)
        {
            for (int dx = 0; dx < scale; ++dx)
            {
                unsigned int src_x = x * scale + dx;
                unsigned int src_y = y * scale + dy;
                unsigned int src_idx = (src_y * orig_width + src_x) * 3;

                sum_r += src[src_idx + 0];
                sum_g += src[src_idx + 1];
                sum_b += src[src_idx + 2];
            }
        }

        int total_pixels = scale * scale;
        unsigned int dst_idx = (y * new_width + x) * 3;
        dst[dst_idx + 0] = (unsigned char)(sum_r / total_pixels);
        dst[dst_idx + 1] = (unsigned char)(sum_g / total_pixels);
        dst[dst_idx + 2] = (unsigned char)(sum_b / total_pixels);
    }
}

CudaResolution::CudaResolution(unsigned int width, unsigned int height, int stream_num, int scale)
    : new_width(width / scale), new_height(height / scale), stream_num(stream_num), scale(scale),
      width(width), height(height),
      block_new_height(new_height / stream_num),
      size_src_block(width * height * 3 / stream_num),
      size_dst_block(new_width * block_new_height * 3)
{
    blockSize = new dim3(BLOCK_SIZE_);
    streams = new cudaStream_t[stream_num];
    for (int i = 0; i < stream_num; i++)
    {
        cudaStreamCreate(&streams[i]);
    }
    gridSize = new dim3((new_width + blockSize->x - 1) / blockSize->x, (block_new_height + blockSize->y - 1) / blockSize->y);
    cudaMalloc((void **)&d_src, width * height * 3);
    cudaMalloc((void **)&d_dst, width * height * 3 / (scale * scale));
}

CudaResolution::~CudaResolution()
{
    if (streams)
    {
        for (int i = 0; i < stream_num; i++)
        {
            cudaStreamDestroy(streams[i]);
        }
        delete[] streams;
        streams = nullptr;
    }
    if (blockSize)
    {
        delete blockSize;
        blockSize = nullptr;
    }
    if (gridSize)
    {
        delete gridSize;
        gridSize = nullptr;
    }
    if (d_src)
    {
        cudaFree(d_src);
        d_src = nullptr;
    }
    if (d_dst)
    {
        cudaFree(d_dst);
        d_dst = nullptr;
    }
}

void CudaResolution::convert(const unsigned char *src, unsigned char *dst)
{
    for (int i = 0; i < stream_num; i++)
    {
        cudaMemcpyAsync(
            d_src + i * size_src_block,
            src + i * size_src_block,
            size_src_block,
            cudaMemcpyHostToDevice,
            streams[i]);

        rgb_resolution_kernel<<<*gridSize, *blockSize, 0, streams[i]>>>(
            d_src + i * size_src_block,
            d_dst + i * size_dst_block,
            new_width,
            block_new_height,
            width,
            scale);

        cudaMemcpyAsync(
            dst + i * size_dst_block,
            d_dst + i * size_dst_block,
            size_dst_block,
            cudaMemcpyDeviceToHost,
            streams[i]);
    }
    cudaDeviceSynchronize();
}
