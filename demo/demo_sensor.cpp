#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>

#include "demo_sensor.h"

using namespace std;

demo_sensor::demo_sensor(const std::string& filename)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "[demo] cannot open the file" << endl;
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

float demo_sensor::get_value(int index) const
{
    return val[index];
}

int demo_sensor::size() const
{
    return static_cast<int>(val.size());
}


