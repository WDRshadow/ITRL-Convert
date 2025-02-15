rm libspinnaker_stream.a
rm spinnaker_stream.o
rm spinnaker_stream

g++ -I/usr/include/spinnaker -L/usr/lib -lSpinnaker -lstdc++ -c -o spinnaker_stream.o spinnaker_stream.cpp

ar rcs libspinnaker_stream.a spinnaker_stream.o

go build spinnaker_stream.go
