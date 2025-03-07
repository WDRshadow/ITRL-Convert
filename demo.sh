rm demo

g++ -I. -o demo demo.cpp -lncurses `pkg-config --cflags --libs opencv4`