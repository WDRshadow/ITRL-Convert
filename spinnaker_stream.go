package main

/*
#cgo CXXFLAGS: -I/usr/include/spinnaker
#cgo LDFLAGS: -L. -lspinnaker_stream -lSpinnaker -lstdc++ -L/usr/local/cuda/lib64 -lcudart
extern void capture_frames(const char* video_device);
*/
import "C"
import "fmt"

func main() {
    // Call the C++ function to capture frames and send them to the virtual video device
    videoDevice := C.CString("/dev/video16")
    fmt.Println("Starting to capture frames from the FLIR camera...")

    C.capture_frames(videoDevice)

    fmt.Println("Capture process finished")
}

