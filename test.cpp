#include <iostream>
#include <chrono>

#include "convert_rgb24_to_yuyv.h"
#include "convert_rgb24_to_yuyv_cuda.h"
#include "convert_rgb24_to_yuyv_parallel.h"

extern "C"
{
    int main()
    {
        unsigned int width = 3072;
        unsigned int height = 2048;
        auto *imageData = new unsigned char[width * height * 3];
        auto *yuyv422 = new unsigned char[width * height * 2];
        auto *yuyv4221 = new unsigned char[width * height * 2];
        auto *yuyv4222 = new unsigned char[width * height * 2];

        // Store the time of each conversion
        std::vector<double> cpu_times;
        std::vector<double> cpu_times1;
        std::vector<double> cpu_times2;

        // Apply for thread pool
        ThreadPool thread_pool{8, width, height};

        for (int i = 0; i < 100; i++)
        {
            // Randomly set pixels in imageData
            for (int j = 0; j < width * height * 3; j++)
            {
                imageData[j] = rand() % 256;
            }

            // sequential
            auto start = std::chrono::high_resolution_clock::now();
            convert_rgb24_to_yuyv(imageData, yuyv422, width, height);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            cpu_times.push_back(elapsed.count() * 1000);

            // palallel
            auto start1 = std::chrono::high_resolution_clock::now();
            thread_pool.convert_task(imageData, yuyv4221);
            auto end1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed1 = end1 - start1;
            cpu_times1.push_back(elapsed1.count() * 1000);

            // cuda
            auto start2 = std::chrono::high_resolution_clock::now();
            convert_rgb24_to_yuyv_cuda(imageData, yuyv4222, width, height);
            auto end2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed2 = end2 - start2;
            cpu_times2.push_back(elapsed2.count() * 1000);
        }

        // print the average time
        double sum = 0;
        double sum1 = 0;
        double sum2 = 0;
        for (int i = 0; i < 100; i++)
        {
            sum += cpu_times[i];
            sum1 += cpu_times1[i];
            sum2 += cpu_times2[i];
        }
        std::cout << "Sequential Average time: " << sum / 100 << " ms" << std::endl;
        std::cout << "Palallel Average time: " << sum1 / 100 << " ms" << std::endl;
        std::cout << "Cuda Average time: " << sum2 / 100 << " ms" << std::endl;
    }
}