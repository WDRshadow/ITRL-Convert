#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>

#include "sensor.h"

#ifdef DEMO
#include "global_index.h"
#endif

using namespace std;

SensorBuffer::SensorBuffer(const string& filename)
{
    ifstream file(filename);
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
            val.push_back(stof(value));
        }
    }

    file.close();
}

float SensorBuffer::get_value(int index) const
{
    return val[index];
}

float SensorBuffer::get_value() const
{
    static int index = 0;
    return get_value(index++);
}
