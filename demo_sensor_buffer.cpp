//
// Created by Yunhao Xu on 25-3-8.
//
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "sensor.h"

#ifdef DEMO
#include "demo_global_index.h"
#endif

using namespace std;

class SensorBuffer
{
    vector<float> angles_100hz;

public:
    SensorBuffer()
    {
        ifstream file("test/test.csv");
        if (!file.is_open())
        {
            cerr << "Error: cannot open the file" << endl;
            return;
        }
        string line;
        while (getline(file, line))
        {
            stringstream ss(line);
            string value;
            if (getline(ss, value, ','))
            {
                value.erase(remove_if(value.begin(), value.end(), [](char c)
                {
                    return !isdigit(c) && c != '.' && c != '-';
                }), value.end());
                angles_100hz.push_back(stof(value));
            }
        }

        file.close();
    }

    float get_angle(int index)
    {
        return angles_100hz[index];
    }

    float get_angle()
    {
        static int index = 0;
        return get_angle(index++);
    }
};

float get_steering_wheel_angle()
{
    static SensorBuffer sensor_buffer;
#ifdef DEMO
    return sensor_buffer.get_angle(GLOBAL_INDEX);
#else
    return sensor_buffer.get_angle();
#endif
}
