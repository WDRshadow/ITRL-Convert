#include <iostream>
#include <chrono>
#include <algorithm>

#include "convert_rgb24_to_yuyv_parallel.h"

extern "C"
{
    int main()
    {
        unsigned int width = 3072;
        unsigned int height = 2048;
        auto *imageData = new unsigned char[width * height * 3];
        auto *yuyv422 = new unsigned char[width * height * 2];

        // Store the time of each conversion
        std::vector<double> cpu_times1;

        // Apply for thread pool
        ThreadPool thread_pool{8, width, height / 2};

        for (int i = 0; i < 100; i++)
        {
            // Randomly set pixels in imageData
            for (int j = 0; j < width * height * 3; j++)
            {
                imageData[j] = rand() % 256;
            }

            // with cuda and parallel cpu
            auto start1 = std::chrono::high_resolution_clock::now();
            thread_pool.convert_task(imageData, yuyv422);
            // merge imageData_cuda and imageData_parallel into imageData
            auto end1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed1 = end1 - start1;
            cpu_times1.push_back(elapsed1.count() * 1000);

        }

        // print the average time
        double sum1 = 0;
        for (int i = 0; i < 100; i++)
        {
            sum1 += cpu_times1[i];
        }
        std::cout << "Average time: " << sum1 / 100 << " ms" << std::endl;
    }
}