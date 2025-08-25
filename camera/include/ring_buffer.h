#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <cstring>

class RingBuffer
{
    const int width;
    const int height;
    const int frame_size;
    std::unique_ptr<unsigned char[]> buffer;
    int buffer_size = 0;
    std::atomic<int> current{0};
    
    std::thread worker_thread;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::atomic<bool> stop_thread{false};
    std::queue<std::pair<std::unique_ptr<unsigned char[]>, int>> copy_queue;
    
    std::mutex copy_mutex;
    std::condition_variable copy_cv;
    std::atomic<bool> copy_pending{false};

    void worker_function();

public:
    RingBuffer(int delay_ms, int frequency_hz, int width, int height, int bytes_per_pixel = 1)
        : width(width), height(height), frame_size(width * height * bytes_per_pixel)
    {
        buffer_size = frequency_hz * delay_ms / 1000 + 1;
        buffer = std::make_unique<unsigned char[]>(buffer_size * frame_size);
        
        std::memset(buffer.get(), 0, buffer_size * frame_size);
        
        worker_thread = std::thread(&RingBuffer::worker_function, this);
    }

    ~RingBuffer()
    {
        stop_thread = true;
        cv.notify_all();
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
    }

    [[nodiscard]] unsigned char *get_newest()
    {
        return buffer.get() + (current * frame_size);
    }

    [[nodiscard]] unsigned char *get_oldest() const
    {
        return buffer.get() + (((current + 1) % buffer_size) * frame_size);
    }

    void update(const unsigned char *new_data);
    void join();
};

#endif // RING_BUFFER_H