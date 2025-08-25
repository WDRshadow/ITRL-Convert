#include "ring_buffer.h"

void RingBuffer::worker_function()
{
    while (!stop_thread) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        
        cv.wait(lock, [this] { return !copy_queue.empty() || stop_thread; });
        
        if (stop_thread) {
            break;
        }
        
        while (!copy_queue.empty()) {
            auto [data, target_index] = std::move(copy_queue.front());
            copy_queue.pop();
            
            lock.unlock();
            
            unsigned char* target_ptr = buffer.get() + (target_index * frame_size);
            std::memcpy(target_ptr, data.get(), frame_size);
            
            {
                std::lock_guard<std::mutex> copy_lock(copy_mutex);
                copy_pending = false;
            }
            copy_cv.notify_all();
            
            lock.lock();
        }
    }
}

void RingBuffer::update(const unsigned char *new_data)
{
    if (!new_data) {
        return;
    }
    
    int next_index = (current + 1) % buffer_size;
    
    auto data_copy = std::make_unique<unsigned char[]>(frame_size);
    std::memcpy(data_copy.get(), new_data, frame_size);
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        copy_pending = true;
        copy_queue.push({std::move(data_copy), next_index});
    }
    
    current = next_index;
    
    cv.notify_one();
}

void RingBuffer::join()
{
    std::unique_lock<std::mutex> lock(copy_mutex);
    copy_cv.wait(lock, [this] { return !copy_pending; });
}
