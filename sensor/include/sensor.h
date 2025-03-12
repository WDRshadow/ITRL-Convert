#ifndef SENSOR_H
#define SENSOR_H

#define RemoteSteeringAngle 0
#define IncPkgNr 1
#define ThrottlePerc 2
#define BrkPerc 3
#define RollRate 4
#define PitchRate 5
#define YawRate 6
#define AZ 7
#define AY 8
#define AX 9
#define Velocity 10
#define SteeringFeedbackTorque 11
#define Motor1 12
#define Motor2 13
#define Motor3 14
#define Motor4 15
#define Motor5 16
#define Motor6 17
#define RPM 18
#define RoadDetail 19
#define XXXXX_1 20
#define XXXXX_2 21
#define XXXXX_3 22
#define XXXXX_4 23
#define XXXXX_5 24

class SensorAPI
{
    const int id;
    const int buffer_size;
    char* buffer;
    char* local_buffer;
    std::shared_mutex& bufferMutex;

public:
    explicit SensorAPI(int id, char* buffer, int buffer_size, std::shared_mutex& bufferMutex);
    [[nodiscard]] float get_value() const;
};

uint32_t BigEndianToUint32(const char* bytes);
float BigEndianToFloat(const char* bytes);


#endif //SENSOR_H
