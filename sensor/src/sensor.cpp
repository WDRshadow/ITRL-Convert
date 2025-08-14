#include <vector>
#include <shared_mutex>
#include <cstring>

#include "sensor.h"

using namespace std;

SensorAPI::SensorAPI(const int id, char* buffer, const int buffer_size, std::shared_mutex& bufferMutex):
    id(id), buffer(buffer), buffer_size(buffer_size), bufferMutex(bufferMutex)
{
    local_buffer = new char[buffer_size];
}

float SensorAPI::get_float_value() const
{
    std::shared_lock lock(bufferMutex);
    std::memcpy(local_buffer, buffer, buffer_size * sizeof(char));
    lock.unlock();
    return getFloatAt(local_buffer, id);
}

int SensorAPI::get_int_value() const
{
    std::shared_lock lock(bufferMutex);
    std::memcpy(local_buffer, buffer, buffer_size * sizeof(char));
    lock.unlock();
    return getIntAt(local_buffer, id);
}

int getIntAt(const char* buffer, const int id)
{
    if (id < 0 || id >= 25)
        throw std::out_of_range("[sensor] Invalid index.");

    int value;
    std::memcpy(&value, buffer + id * sizeof(int), sizeof(int));
    return value;
}

float getFloatAt(const char* buffer, const int id)
{
    if (id < 0 || id >= 25)
        throw std::out_of_range("[sensor] Invalid index.");

    float value;
    std::memcpy(&value, buffer + id * sizeof(float), sizeof(float));
    return value;
}
