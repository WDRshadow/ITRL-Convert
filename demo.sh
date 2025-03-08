rm demo

g++ -std=c++17 -DDEMO -o demo demo.cpp demo_sensor_buffer.cpp `pkg-config --cflags --libs opencv4`