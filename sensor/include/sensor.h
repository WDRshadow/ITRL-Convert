#ifndef SENSOR_H
#define SENSOR_H

#include <vector>

using namespace std;

class SensorBuffer
{
protected:
    vector<float> val;

public:
    virtual ~SensorBuffer() = default;
    SensorBuffer() = default;
    explicit SensorBuffer(const string& filename);
    [[nodiscard]] float get_value(int index) const;
    [[nodiscard]] virtual float get_value() const;
};

#endif //SENSOR_H
