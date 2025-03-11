#include <iostream>

#include "spinnaker_stream.h"

int main() {
    const char* videoDevice = "/dev/video16";
    std::cout << "[main] Starting to capture frames from the FLIR camera..." << std::endl;

    capture_frames(videoDevice, "192.168.1.121", 10000);

    std::cout << "[main] Capture process finished" << std::endl;
    return 0;
}