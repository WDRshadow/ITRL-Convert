rm libspinnaker_stream.a
rm spinnaker_stream.o
rm spinnaker_stream
rm convert_rgb24_to_yuyv_cuda.o

nvcc -Xcompiler -fPIC -c convert_rgb24_to_yuyv_cuda.cu -o convert_rgb24_to_yuyv_cuda.o
g++ -I. -I/usr/include/spinnaker -c -o spinnaker_stream.o spinnaker_stream.cpp `pkg-config --cflags --libs opencv4`

ar rcs libspinnaker_stream.a spinnaker_stream.o convert_rgb24_to_yuyv_cuda.o

go build spinnaker_stream.go
