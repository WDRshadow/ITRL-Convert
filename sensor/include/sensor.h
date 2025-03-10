#ifndef SENSOR_H
#define SENSOR_H

#define STR_WHE_PHI 0
#define VEL 1

#include <vector>

class SensorBase
{
protected:
    std::vector<float> val;

public:
    virtual ~SensorBase() = default;
    [[nodiscard]] virtual float get_value() const = 0;
};

class SensorBuffer final : public SensorBase
{
public:
    explicit SensorBuffer(const std::string& filename);
    [[nodiscard]] float get_value(int index) const;
    [[nodiscard]] float get_value() const override;
};

class SensorAPI final : public SensorBase
{
public:
    explicit SensorAPI(int id);
    [[nodiscard]] float get_value() const override;
};


#endif //SENSOR_H
