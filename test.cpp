#include <iostream>
#include <chrono>

#include "convert_rgb24_to_yuyv.h"
#include "ThreadPool.h"

extern "C"
{
    int main()
    {
        unsigned int width = 3072;
        unsigned int height = 2048;
        auto *imageData = new unsigned char[width * height * 3];
        auto *yuyv422 = new unsigned char[width * height * 2];

        // Apply for thread pool
        ThreadPool thread_pool{4, width, height};

        for (int i = 0; i < 100; i++)
        {
            // Randomly set pixels in imageData
            for (int j = 0; j < width * height * 3; j++)
            {
                imageData[j] = rand() % 256;
            }

            // normal
            auto start2 = std::chrono::high_resolution_clock::now();
            convert_rgb24_to_yuyv(imageData, yuyv422, width, height);
            auto end2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed2 = end2 - start2;
            std::cout << "Normal Conversion time: " << elapsed2.count() * 1000 << " ms" << std::endl;

            // thread pool
            auto start = std::chrono::high_resolution_clock::now();
            thread_pool.convert_task(imageData, yuyv422);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "Parallel Conversion time: " << elapsed.count() * 1000 << " ms" << std::endl;
        }
    }
}