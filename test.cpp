#include <iostream>
#include <chrono>

#include "convert_rgb24_to_yuyv.h"
#include "convert_rgb24_to_yuyv_neon.h"

extern "C"
{
    int main()
    {
        unsigned int width = 1920;
        unsigned int height = 1080;
        auto *imageData = new unsigned char[width * height * 3];
        auto *yuyv422 = new unsigned char[width * height * 2];
        auto *yuyv4222 = new unsigned char[width * height * 2];
        ConvertContext g_convertCtx{};
        for (int i = 0; i < 100; i++)
        {
            // Randomly set pixels in imageData
            for (int j = 0; j < width * height * 3; j++)
            {
                imageData[j] = rand() % 256;
            }

            // Convert RGB24 to YUYV422
            auto start1 = std::chrono::high_resolution_clock::now();
            convert_rgb24_to_yuyv(imageData, yuyv422, width, height);
            auto end1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed1 = end1 - start1;
            std::cout << "Normal Conversion time: " << elapsed1.count() * 1000 << " ms" << std::endl;

            // Convert RGB24 to YUYV422 using NEON
            auto start2 = std::chrono::high_resolution_clock::now();
            convert_rgb24_to_yuyv_neon(imageData, yuyv4222, width, height, g_convertCtx);
            auto end2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed2 = end2 - start2;
            std::cout << "NEON Conversion time: " << elapsed2.count() * 1000 << " ms" << std::endl;
        }
    }
}