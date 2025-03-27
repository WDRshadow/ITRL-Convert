#include <stdio.h>
#include <cuda_runtime.h>

__global__ void bayer2nv12_kernel(const unsigned char *bayer, unsigned char *y_plane,
                                  unsigned char *uv_plane, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    unsigned char bayer_val = bayer[idx];
    unsigned char y_val;

    bool is_green = (y % 2 == 0 && x % 2 == 1) || (y % 2 == 1 && x % 2 == 0);
    bool is_red = (y % 2 == 0 && x % 2 == 0);
    bool is_blue = (y % 2 == 1 && x % 2 == 1);

    // Y
    if (is_green)
    {
        y_val = bayer_val;
    }
    else if (is_red || is_blue)
    {
        unsigned char left = (x > 0) ? bayer[y * width + x - 1] : bayer_val;
        unsigned char right = (x < width - 1) ? bayer[y * width + x + 1] : bayer_val;
        unsigned char up = (y > 0) ? bayer[(y - 1) * width + x] : bayer_val;
        unsigned char down = (y < height - 1) ? bayer[(y + 1) * width + x] : bayer_val;

        y_val = (left + right + up + down) / 4;
    }

    // Y -> Y Plane
    y_plane[y * width + x] = y_val;

    // U, V
    if (x % 2 == 0 && y % 2 == 0 && x < width - 1 && y < height - 1)
    {
        int uv_index = (y / 2) * width + x;

        unsigned char R = bayer[y * width + x];
        unsigned char G = bayer[y * width + x + 1];
        unsigned char B = bayer[(y + 1) * width + x + 1];

        // YUV formula（ITU-R BT.601）
        int u = -38 * R - 74 * G + 112 * B + 128 * 256;
        int v = 112 * R - 94 * G - 18 * B + 128 * 256;

        u = min(max(u >> 8, 0), 255);
        v = min(max(v >> 8, 0), 255);

        // U, V -> UV Plane
        uv_plane[uv_index] = u;
        uv_plane[uv_index + 1] = v;
    }
}
