#include <iostream>

#include "spinnaker_stream.h"

int main() {
    const char* videoDevice = "/dev/video16";
    std::cout << "Starting to capture frames from the FLIR camera..." << std::endl;

    capture_frames(videoDevice);

    std::cout << "Capture process finished" << std::endl;
    return 0;
}