#include <vector>
#include <shared_mutex>
#include <cstring>

#include "sensor.h"

using namespace std;

SensorAPI::SensorAPI(const int id, char *buffer, const int buffer_size, std::shared_mutex &bufferMutex) : id(id), buffer(buffer), buffer_size(buffer_size), bufferMutex(bufferMutex)
{
}

float SensorAPI::get_float_value() const
{
    std::shared_lock lock(bufferMutex);
    float value;
    std::memcpy(&value, buffer + id * sizeof(float), sizeof(float));
    return value;
}

int SensorAPI::get_int_value() const
{
    std::shared_lock lock(bufferMutex);
    int value;
    std::memcpy(&value, buffer + id * sizeof(int), sizeof(int));
    return value;
}
