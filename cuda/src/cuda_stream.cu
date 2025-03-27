#include <cuda_runtime.h>

#include "cuda_stream.h"
#include "bayerRG2rgb.h"
#include "rgb2yuyv.h"

bool is_bayer2yuyv_cuda_initialized = false;
int bayer2yuyv_stream_num_;
cudaStream_t *bayer2yuyv_streams = nullptr;
unsigned int bayer2yuyv_width_;
unsigned int bayer2yuyv_height_;
unsigned int bayer2yuyv_block_height;
size_t bayer2yuyv_size_bayer_block;
size_t bayer2yuyv_size_rgb_block;
size_t bayer2yuyv_size_yuyv_block;
unsigned char *bayer2yuyv_d_bayer = nullptr;
unsigned char *bayer2yuyv_d_rgb = nullptr;
unsigned char *bayer2yuyv_d_yuyv = nullptr;

const dim3 bayer2yuyv_blockSize(32, 16);
dim3 bayer2yuyv_gridSize_1;
dim3 bayer2yuyv_gridSize_2;

void init_bayer2yuyv_cuda(unsigned int width, unsigned int height, int stream_num)
{
    if (is_bayer2yuyv_cuda_initialized)
    {
        cleanup_bayer2yuyv_cuda();
    }
    bayer2yuyv_width_ = width;
    bayer2yuyv_height_ = height;
    bayer2yuyv_stream_num_ = stream_num;
    bayer2yuyv_block_height = height / stream_num;
    bayer2yuyv_size_bayer_block = width * bayer2yuyv_block_height;
    bayer2yuyv_size_rgb_block = width * bayer2yuyv_block_height * 3;
    bayer2yuyv_size_yuyv_block = width * bayer2yuyv_block_height * 2;
    bayer2yuyv_gridSize_1 = dim3((width + bayer2yuyv_blockSize.x - 1) / bayer2yuyv_blockSize.x, (bayer2yuyv_block_height + bayer2yuyv_blockSize.y - 1) / bayer2yuyv_blockSize.y);
    bayer2yuyv_gridSize_2 = dim3((width / 2 + bayer2yuyv_blockSize.x - 1) / bayer2yuyv_blockSize.x, (bayer2yuyv_block_height + bayer2yuyv_blockSize.y - 1) / bayer2yuyv_blockSize.y);
    bayer2yuyv_streams = (cudaStream_t *)malloc(stream_num * sizeof(cudaStream_t));
    for (int i = 0; i < stream_num; i++)
    {
        cudaStreamCreate(&bayer2yuyv_streams[i]);
    }
    cudaMalloc((void **)&bayer2yuyv_d_bayer, width * height);
    cudaMalloc((void **)&bayer2yuyv_d_rgb, width * height * 3);
    cudaMalloc((void **)&bayer2yuyv_d_yuyv, width * height * 2);
    is_bayer2yuyv_cuda_initialized = true;
}

void bayer2yuyv_cuda(const unsigned char *bayer, unsigned char *yuyv)
{
    if (!is_bayer2yuyv_cuda_initialized)
    {
        return;
    }
    for (int i = 0; i < bayer2yuyv_stream_num_; i++)
    {
        cudaMemcpyAsync(
            bayer2yuyv_d_bayer + i * bayer2yuyv_size_bayer_block,
            bayer + i * bayer2yuyv_size_bayer_block,
            bayer2yuyv_size_bayer_block,
            cudaMemcpyHostToDevice,
            bayer2yuyv_streams[i]);

        bayerRG2rgb_kernel<<<bayer2yuyv_gridSize_1, bayer2yuyv_blockSize, 0, bayer2yuyv_streams[i]>>>(
            bayer2yuyv_d_bayer + i * bayer2yuyv_size_bayer_block,
            bayer2yuyv_d_rgb + i * bayer2yuyv_size_rgb_block,
            bayer2yuyv_width_,
            bayer2yuyv_block_height);

        rgb2yuyv_kernel<<<bayer2yuyv_gridSize_2, bayer2yuyv_blockSize, 0, bayer2yuyv_streams[i]>>>(
            bayer2yuyv_d_rgb + i * bayer2yuyv_size_rgb_block,
            bayer2yuyv_d_yuyv + i * bayer2yuyv_size_yuyv_block,
            bayer2yuyv_width_,
            bayer2yuyv_block_height);

        cudaMemcpyAsync(
            yuyv + i * bayer2yuyv_size_yuyv_block,
            bayer2yuyv_d_yuyv + i * bayer2yuyv_size_yuyv_block,
            bayer2yuyv_size_yuyv_block,
            cudaMemcpyDeviceToHost,
            bayer2yuyv_streams[i]);
    }
    cudaDeviceSynchronize();
}

void cleanup_bayer2yuyv_cuda()
{
    if (!is_bayer2yuyv_cuda_initialized)
    {
        return;
    }
    for (int i = 0; i < bayer2yuyv_stream_num_; i++)
    {
        cudaStreamDestroy(bayer2yuyv_streams[i]);
    }
    free(bayer2yuyv_streams);
    bayer2yuyv_streams = nullptr;
    cudaFree(bayer2yuyv_d_bayer);
    bayer2yuyv_d_bayer = nullptr;
    cudaFree(bayer2yuyv_d_rgb);
    bayer2yuyv_d_rgb = nullptr;
    cudaFree(bayer2yuyv_d_yuyv);
    bayer2yuyv_d_yuyv = nullptr;
    is_bayer2yuyv_cuda_initialized = false;
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