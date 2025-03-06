rm test

g++ -I. -o test test.cpp `pkg-config --cflags --libs opencv4`