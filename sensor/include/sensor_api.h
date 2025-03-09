#ifndef SENSOR_API_H
#define SENSOR_API_H
#include <sensor.h>

class Sensor_API final : public SensorBuffer
{
public:
    Sensor_API();
    [[nodiscard]] float get_value() const override;
};

#endif //SENSOR_API_H
