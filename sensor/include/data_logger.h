#ifndef DATA_LOGGER_H
#define DATA_LOGGER_H

#include <shared_mutex>

#include "sensor.h"

class DataLogger
{
    const int *ids;
    const int value_count;
    const int buffer_size;
    char *buffer;
    char *local_buffer;
    std::shared_mutex &bufferMutex;
    const char *file_name;
    unsigned int frame_counter = 0;

public:
    explicit DataLogger(const int *ids, int value_count, char *buffer, int buffer_size, std::shared_mutex &bufferMutex, const char *file_name);
    void logger();
};

#endif // DATA_LOGGER_H