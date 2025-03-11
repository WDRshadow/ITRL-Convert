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

float SensorAPI::get_value() const
{
    std::shared_lock lock(bufferMutex);
    std::memcpy(local_buffer, buffer, buffer_size * sizeof(char));
    lock.unlock();
    if (id == IncPkgNr)
    {
        return static_cast<float>(BigEndianToUint32(local_buffer + id * sizeof(float)));
    }
    return BigEndianToFloat(local_buffer + id * sizeof(float));
}

uint32_t BigEndianToUint32(const char* bytes)
{
    return static_cast<uint32_t>(static_cast<unsigned char>(bytes[0])) << 24 |
        static_cast<uint32_t>(static_cast<unsigned char>(bytes[1])) << 16 |
        static_cast<uint32_t>(static_cast<unsigned char>(bytes[2])) << 8 |
        static_cast<uint32_t>(static_cast<unsigned char>(bytes[3])) << 0;
}

float BigEndianToFloat(const char* bytes)
{
    uint32_t tmp = BigEndianToUint32(bytes);
    float f;
    std::memcpy(&f, &tmp, sizeof(f));
    return f;
}
