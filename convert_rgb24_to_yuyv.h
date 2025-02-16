// Function to clamp the pixel values to the range [0, 255]
inline unsigned char clamp(int value)
{
    return (value < 0) ? 0 : (value > 255 ? 255 : value);
}

// Function to convert RGB24 to YUYV422
void convert_rgb24_to_yuyv(const unsigned char *rgb24, unsigned char *yuyv422, int width, int height)
{
    int index_rgb = 0;
    int index_yuyv = 0;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x += 2)
        {
            // Get RGB values for two pixels
            unsigned char r0 = rgb24[index_rgb];
            unsigned char g0 = rgb24[index_rgb + 1];
            unsigned char b0 = rgb24[index_rgb + 2];

            unsigned char r1 = rgb24[index_rgb + 3];
            unsigned char g1 = rgb24[index_rgb + 4];
            unsigned char b1 = rgb24[index_rgb + 5];

            // Convert RGB to YUV
            unsigned char y0 = clamp((0.299 * r0 + 0.587 * g0 + 0.114 * b0));
            unsigned char y1 = clamp((0.299 * r1 + 0.587 * g1 + 0.114 * b1));
            unsigned char u = clamp((-0.14713 * r0 - 0.28886 * g0 + 0.436 * b0) + 128);
            unsigned char v = clamp((0.615 * r0 - 0.51499 * g0 - 0.10001 * b0) + 128);

            // Pack YUV values into YUYV422 format
            yuyv422[index_yuyv] = y0;
            yuyv422[index_yuyv + 1] = u;
            yuyv422[index_yuyv + 2] = y1;
            yuyv422[index_yuyv + 3] = v;

            index_rgb += 6;
            index_yuyv += 4;
        }
    }
}