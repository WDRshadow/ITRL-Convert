rm libspinnaker_stream.a
rm spinnaker_stream.o
rm spinnaker_stream

# nvcc -Xcompiler -fPIC -shared -o libconvert_rgb24_to_yuyv_cuda.so convert_rgb24_to_yuyv_cuda.cu
# g++ -I. -I/usr/include/spinnaker -L/usr/lib -lSpinnaker -lstdc++ -L. -lconvert_rgb24_to_yuyv_cuda -c -o spinnaker_stream.o spinnaker_stream.cpp -Wl,-rpath=.

g++ -I. -I/usr/include/spinnaker -L/usr/lib -lSpinnaker -lstdc++ -c -o spinnaker_stream.o spinnaker_stream.cpp

ar rcs libspinnaker_stream.a spinnaker_stream.o

go build spinnaker_stream.go
