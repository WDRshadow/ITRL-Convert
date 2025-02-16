rm test
rm convert_rgb24_to_yuyv_cuda.o

nvcc -Xcompiler -fPIC -c convert_rgb24_to_yuyv_cuda.cu -o convert_rgb24_to_yuyv_cuda.o
g++ -I. -lstdc++ -L/usr/local/cuda/lib64 -lcudart -o test test.cpp convert_rgb24_to_yuyv_cuda.o -Wl,-rpath,/usr/local/cuda/lib64 -lcudart