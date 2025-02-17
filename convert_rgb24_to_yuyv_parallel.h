//
// Created by Yunhao Xu on 25-2-17.
//

#ifndef CONVERT_RGB24_TO_YUYV_PARALLEL_H
#define CONVERT_RGB24_TO_YUYV_PARALLEL_H
#include <vector>
#include <thread>

#include "convert_rgb24_to_yuyv.h"

class ThreadPool
{
    std::vector<std::thread> workers;
    std::vector<unsigned int> starting_node;
    const unsigned int width;
    const unsigned int height;
    const unsigned int pixel_per_height;
    const unsigned int pixel_per_worker;
    const unsigned int thread_num;

public:
    ThreadPool(unsigned int thread_num, unsigned int width, unsigned int height);
    void convert_task(const unsigned char* rgb24, unsigned char* yuyv422);
};

inline ThreadPool::ThreadPool(const unsigned int thread_num, const unsigned int width, const unsigned int height):
    width(width),
    height(height), pixel_per_height(height / thread_num), thread_num(thread_num),
    pixel_per_worker(width * pixel_per_height)
{
    for (unsigned int i = 0; i < thread_num; i++)
    {
        starting_node.push_back(i * pixel_per_worker);
    }
}

inline void ThreadPool::convert_task(const unsigned char* rgb24, unsigned char* yuyv422)
{
    for (unsigned int i = 0; i < thread_num; i++)
    {
        workers.emplace_back(convert_rgb24_to_yuyv_core, rgb24, starting_node[i], yuyv422, width, pixel_per_height);
    }
    
    for (std::thread &worker : workers)
    {
        worker.join();
    }
    workers.clear();
}

#endif //CONVERT_RGB24_TO_YUYV_PARALLEL_H
