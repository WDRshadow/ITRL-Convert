rm libspinnaker_stream.a
rm spinnaker_stream.o
rm spinnaker_stream
rm convert_rgb24_to_yuyv_cuda.o
rm demo_sensor_buffer.o

nvcc -Xcompiler -fPIC -c convert_rgb24_to_yuyv_cuda.cu -o convert_rgb24_to_yuyv_cuda.o
g++ -std=c++17 -I/usr/include/spinnaker -c -o spinnaker_stream.o spinnaker_stream.cpp `pkg-config --cflags --libs opencv4`
g++ -std=c++17 -c -o demo_sensor_buffer.o demo_sensor_buffer.cpp

ar rcs libspinnaker_stream.a spinnaker_stream.o convert_rgb24_to_yuyv_cuda.o demo_sensor_buffer.o

go build spinnaker_stream.go
