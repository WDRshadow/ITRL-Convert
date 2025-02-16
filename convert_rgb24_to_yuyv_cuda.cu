#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <device_launch_parameters.h>

#include "spinnaker_stream.h"

// 颜色转换常量（整数运算）
#define Y_R 77  // 0.299 * 256 ≈ 77
#define Y_G 150 // 0.587 * 256 ≈ 150
#define Y_B 29  // 0.114 * 256 ≈ 29

#define U_R -38 // -0.14713 * 256 ≈ -38
#define U_G -74 // -0.28886 * 256 ≈ -74
#define U_B 112 // 0.436 * 256 ≈ 112

#define V_R 157  // 0.615 * 256 ≈ 157
#define V_G -132 // -0.51499 * 256 ≈ -132
#define V_B -25  // -0.10001 * 256 ≈ -25

// 计算 YUYV 格式的 CUDA 核心
__global__ void convert_rgb24_to_yuyv_cuda(const uint8_t *rgb24, uint8_t *yuyv422, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 线程索引（像素索引）
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel_index = (idy * width + idx) * 3; // RGB24 是 3 通道

    if (idx >= width || idy >= height)
        return; // 边界检查

    // 获取 RGB 值
    uint8_t r = rgb24[pixel_index];
    uint8_t g = rgb24[pixel_index + 1];
    uint8_t b = rgb24[pixel_index + 2];

    // 计算 Y
    uint8_t y = (Y_R * r + Y_G * g + Y_B * b) >> 8;

    // 计算 U 和 V（每 2 像素计算 1 组 U/V）
    uint8_t u = (U_R * r + U_G * g + U_B * b) >> 8;
    uint8_t v = (V_R * r + V_G * g + V_B * b) >> 8;

    // 存储到 YUYV422 格式
    int yuyv_index = (idy * width + idx) * 2; // YUYV 是 2 通道
    yuyv422[yuyv_index] = y;
    if (idx % 2 == 0)
    {
        yuyv422[yuyv_index + 1] = u + 128; // 只给偶数索引像素赋值 U
    }
    else
    {
        yuyv422[yuyv_index + 1] = v + 128; // 只给奇数索引像素赋值 V
    }
}

// 主函数调用 CUDA 核心
void convert_rgb24_to_yuyv_gpu(const uint8_t *rgb24, uint8_t *yuyv422, int width, int height)
{
    uint8_t *d_rgb24, *d_yuyv422;
    int rgb_size = width * height * 3;
    int yuyv_size = width * height * 2;

    // 分配 GPU 内存
    cudaMalloc((void **)&d_rgb24, rgb_size);
    cudaMalloc((void **)&d_yuyv422, yuyv_size);

    // 复制数据到 GPU
    cudaMemcpy(d_rgb24, rgb24, rgb_size, cudaMemcpyHostToDevice);

    // 定义 CUDA 线程块大小（16x16）
    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);

    // 启动 CUDA 核心
    convert_rgb24_to_yuyv_cuda(d_rgb24, d_yuyv422, width, height);
    cudaDeviceSynchronize(); // 同步 GPU 计算

    // 复制结果回 CPU
    cudaMemcpy(yuyv422, d_yuyv422, yuyv_size, cudaMemcpyDeviceToHost);

    // 释放 GPU 内存
    cudaFree(d_rgb24);
    cudaFree(d_yuyv422);
}
