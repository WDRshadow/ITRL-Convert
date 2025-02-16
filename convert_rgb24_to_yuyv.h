#define Y_R 19595  // 0.299 * 65536
#define Y_G 38470  // 0.587 * 65536
#define Y_B 7471   // 0.114 * 65536

#define U_R -11058 // -0.14713 * 65536
#define U_G -21709 // -0.28886 * 65536
#define U_B 32767  // 0.436 * 65536

#define V_R 32767  // 0.615 * 65536
#define V_G -27439 // -0.51499 * 65536
#define V_B -5328  // -0.10001 * 65536

#define CLAMP(x) (x < 0 ? 0 : (x > 255 ? 255 : x))

/**
 * Convert RGB24 to YUYV422. Using the following formulas:
 * 
 * Y = 0.299R + 0.587G + 0.114B
 * U = -0.14713R - 0.28886G + 0.436B
 * V = 0.615R - 0.51499G - 0.10001B
 * 
 * Fixed-point arithmetic optimization is used to speed up the calculation.
 * 
 * @param rgb24 RGB24 data
 * @param yuyv422 YUYV422 data (output)
 * @param width Image width
 * @param height Image height
 */
void convert_rgb24_to_yuyv(const unsigned char *rgb24, unsigned char *yuyv422, int width, int height)
{
    int index_rgb = 0;
    int index_yuyv = 0;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x += 2)
        {
            unsigned char r0 = rgb24[index_rgb];
            unsigned char g0 = rgb24[index_rgb + 1];
            unsigned char b0 = rgb24[index_rgb + 2];

            unsigned char r1 = rgb24[index_rgb + 3];
            unsigned char g1 = rgb24[index_rgb + 4];
            unsigned char b1 = rgb24[index_rgb + 5];

            unsigned char y0 = CLAMP((Y_R * r0 + Y_G * g0 + Y_B * b0) >> 16);
            unsigned char y1 = CLAMP((Y_R * r1 + Y_G * g1 + Y_B * b1) >> 16);
            unsigned char u = CLAMP((U_R * r0 + U_G * g0 + U_B * b0) >> 16) + 128;
            unsigned char v = CLAMP((V_R * r0 + V_G * g0 + V_B * b0) >> 16) + 128;

            yuyv422[index_yuyv] = y0;
            yuyv422[index_yuyv + 1] = u;
            yuyv422[index_yuyv + 2] = y1;
            yuyv422[index_yuyv + 3] = v;

            index_rgb += 6;
            index_yuyv += 4;
        }
    }
}
