#include <iostream>
#include <chrono>

#include "convert_rgb24_to_yuyv.h"
#include "convert_rgb24_to_yuyv_cuda.h"
#include "convert_rgb24_to_yuyv_neon.h"
#include <vector>

extern "C"
{
    int main()
    {
        unsigned int width = 3072;
        unsigned int height = 2048;
        std::vector<double> cpu_times1;
        std::vector<double> cpu_times2;
        std::vector<double> cpu_times3;
        auto *imageData = new unsigned char[width * height * 3];
        auto *yuyv422 = new unsigned char[width * height * 2];
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
            cpu_times1.push_back(elapsed1.count() * 1000);

            // normal
            auto start2 = std::chrono::high_resolution_clock::now();
            convert_rgb24_to_yuyv(imageData, yuyv422, width, height);
            auto end2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed2 = end2 - start2;
            cpu_times2.push_back(elapsed2.count() * 1000);

            // without neon
            auto start3 = std::chrono::high_resolution_clock::now();
            convert_rgb24_to_yuyv(imageData, yuyv422, width, height);
            auto end3 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed3 = end3 - start3;
            cpu_times3.push_back(elapsed3.count() * 1000);
        }
        // print cpu mean time
        double cpu_time1_mean = 0.0;
        double cpu_time2_mean = 0.0;
        double cpu_time3_mean = 0.0;
        for (size_t i = 0; i < cpu_times1.size(); i++)
        {
            cpu_time1_mean += cpu_times1[i];
            cpu_time2_mean += cpu_times2[i];
            cpu_time3_mean += cpu_times3[i];
        }
        cpu_time1_mean /= cpu_times1.size();
        cpu_time2_mean /= cpu_times2.size();
        cpu_time3_mean /= cpu_times3.size();
        std::cout << "CPU time to convert RGB24 to YUYV422 with CUDA (mean): " << cpu_time1_mean << " ms" << std::endl;
        std::cout << "CPU time to convert RGB24 to YUYV422 Normally (mean): " << cpu_time2_mean << " ms" << std::endl;
        std::cout << "CPU time to convert RGB24 to YUYV422 without NEON (mean): " << cpu_time3_mean << " ms" << std::endl;
    }
}