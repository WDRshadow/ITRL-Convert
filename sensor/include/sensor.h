#ifndef SENSOR_H
#define SENSOR_H

#define SteeringAngle 0
#define Velocity 1
#define Ax 2

#define Latency 0

class SensorAPI
{
    const int id;
    const int buffer_size;
    char* buffer;
    std::shared_mutex& bufferMutex;

public:
    explicit SensorAPI(int id, char* buffer, int buffer_size, std::shared_mutex& bufferMutex);
    [[nodiscard]] float get_float_value() const;
    [[nodiscard]] int get_int_value() const;
};

#endif //SENSOR_H
