#include <vector>
#include <shared_mutex>
#include <cstring>
#include <fstream>

#include "data_logger.h"

DataLogger::DataLogger(const int *ids, const int *type, const int value_count, char *buffer, int buffer_size, std::shared_mutex &bufferMutex, const char *file_name)
    : ids(ids), type(type), value_count(value_count), buffer_size(buffer_size), buffer(buffer), bufferMutex(bufferMutex), file_name(file_name)
{
    std::ofstream file(this->file_name);
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
    std::ofstream file(file_name, std::ios::app);
    if (file.is_open())
    {
        for (int i = 0; i < value_count; ++i)
        {
            if (i == 0)
            {
                file << frame_counter++;
            }
            if (type[i] == _TYPE_INT)
            {
                std::shared_lock lock(bufferMutex);
                int value;
                std::memcpy(&value, buffer + ids[i] * sizeof(int), sizeof(int));
                file << "," << value;
            }
            else if (type[i] == _TYPE_FLOAT)
            {
                std::shared_lock lock(bufferMutex);
                float value;
                std::memcpy(&value, buffer + ids[i] * sizeof(float), sizeof(float));
                file << "," << value;
            }
        }
    }
    file << std::endl;
    file.close();
}
