#ifndef DATA_LOGGER_H
#define DATA_LOGGER_H

#include <shared_mutex>
#include <string>

#include "sensor.h"

#define _TYPE_INT 0
#define _TYPE_FLOAT 1

class DataLogger
{
    const int *ids;
    const int *type;
    const int value_count;
    const int buffer_size;
    char *buffer;
    std::shared_mutex &bufferMutex;
    std::string file_name;
    unsigned int frame_counter = 0;

public:
    explicit DataLogger(const int *ids, const int *type, int value_count, char *buffer, int buffer_size, std::shared_mutex &bufferMutex, const char *file_name);
    void logger();
};

#endif // DATA_LOGGER_H