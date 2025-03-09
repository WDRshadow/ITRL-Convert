#ifndef SENSOR_H
#define SENSOR_H

using namespace std;

class SensorBuffer
{
    vector<float> val;

public:
    explicit SensorBuffer(const string& filename);
    [[nodiscard]] float get_value(int index) const;
    [[nodiscard]] float get_value() const;
};

#endif //SENSOR_H
