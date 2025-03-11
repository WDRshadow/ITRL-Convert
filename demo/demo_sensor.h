#ifndef DEMO_SENSOR_H
#define DEMO_SENSOR_H
#include <vector>

class demo_sensor
{
    std::vector<float> val;

public:
    explicit demo_sensor(const std::string& filename);
    [[nodiscard]] float get_value(int index) const;
    [[nodiscard]] int size() const;
};


#endif //DEMO_SENSOR_H
