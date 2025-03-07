rm test

g++ -I. -o test test.cpp -lncurses `pkg-config --cflags --libs opencv4`