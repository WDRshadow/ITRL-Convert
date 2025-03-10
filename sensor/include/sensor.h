#ifndef SENSOR_H
#define SENSOR_H

#define STR_WHE_PHI "str_whe_phi"
#define VEL "vel"

#include <vector>

class SensorBase
{
public:
    virtual ~SensorBase() = default;
    [[nodiscard]] virtual float get_value() const = 0;
};

class SensorBuffer final : public SensorBase
{
    std::vector<float> val;
public:
    explicit SensorBuffer(const std::string& filename);
    [[nodiscard]] float get_value(int index) const;
    [[nodiscard]] float get_value() const override;
    [[nodiscard]] int size() const;
};

class SensorAPI final : public SensorBase
{
    const std::string id;
public:
    explicit SensorAPI(std::string id);
    [[nodiscard]] float get_value() const override;
};


#endif //SENSOR_H
