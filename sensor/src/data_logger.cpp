#include <vector>
#include <shared_mutex>
#include <cstring>
#include <fstream>

#include "data_logger.h"

DataLogger::DataLogger(const int *ids, const int value_count, char *buffer, int buffer_size, std::shared_mutex &bufferMutex, const char *file_name)
    : ids(ids), value_count(value_count), buffer_size(buffer_size), buffer(buffer), bufferMutex(bufferMutex), file_name(file_name)
{
    local_buffer = new char[buffer_size];
    // write the header to the file (csv format)
    std::ofstream file(file_name);
    if (file.is_open())
    {
        file << "frame";
        for (int i = 0; i < value_count; ++i)
        {
            file << "," << ids[i];
        }
        file << std::endl;
        file.close();
    }
}

void DataLogger::logger()
{
    std::shared_lock lock(bufferMutex);
    std::memcpy(local_buffer, buffer, buffer_size * sizeof(char));
    lock.unlock();
    // write the data to the file (csv format)
    std::ofstream file(file_name, std::ios::app);
    if (file.is_open())
    {
        for (int i = 0; i < value_count; ++i)
        {
            float value = getFloatAt(local_buffer, ids[i]);

            if (i == 0)
            {
                file << frame_counter++;
            }
            file << "," << value;
        }
    }
    file << std::endl;
    file.close();
}
