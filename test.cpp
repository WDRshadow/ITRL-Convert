#include <iostream>
#include <chrono>

#include "convert_rgb24_to_yuyv.h"
#include "convert_rgb24_to_yuyv_cuda.h"

extern "C"
{
    int main()
    {
        unsigned int width = 1920;
        unsigned int height = 1080;
        unsigned char *imageData = new unsigned char[width * height * 3];
        unsigned char *yuyv422 = new unsigned char[width * height * 2];
        for (int i = 0; i < 100; i++)
        {
            // Randomly set pixels in imageData
            for (int j = 0; j < width * height * 3; j++)
            {
                imageData[j] = rand() % 256;
            }

            // with cuda
            auto start1 = std::chrono::high_resolution_clock::now();
            convert_rgb24_to_yuyv_cuda(imageData, yuyv422, width, height);
            auto end1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed1 = end1 - start1;
            std::cout << "CUDA Conversion time: " << elapsed1.count() * 1000 << " ms" << std::endl;

            // without cuda
            auto start2 = std::chrono::high_resolution_clock::now();
            convert_rgb24_to_yuyv(imageData, yuyv422, width, height);
            auto end2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed2 = end2 - start2;
            std::cout << "Conversion time: " << elapsed2.count() * 1000 << " ms" << std::endl;
        }
    }
}