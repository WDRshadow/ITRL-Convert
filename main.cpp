#include <iostream>
#include <unordered_map>
#include <string>
#include <csignal>

#include "spinnaker_stream.h"

#define DEFAULT_VIDEO_DEVICE "/dev/video16"

bool capture_signal = false;

void run_spinnaker_stream(const char* videoDevice) {
    std::cout << "[main] Starting to capture frames from the FLIR camera..." << std::endl;
    capture_frames(videoDevice);
    std::cout << "[main] Capture process finished" << std::endl;
}

void signalHandler(int signum)
{
    std::cout << "[main] Received signal " << signum << ", cleaning up resources..." << std::endl;
    capture_signal = true;
}

std::unordered_map<std::string, std::string> parseArguments(int argc, char* argv[]) {
    std::unordered_map<std::string, std::string> args;

    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key.rfind('-', 0) == 0) {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                args[key] = argv[i + 1];
                ++i;
            } else {
                args[key] = "";
            }
        }
    }

    return args;
}

int main(int argc, char* argv[]) {
    auto args = parseArguments(argc, argv);

    if (args.find("-h") != args.end()) {
        std::cout << "[main] Usage: " << argv[0] << " [-d <video_device>]" << std::endl;
        std::cout << "[main] Options:" << std::endl;
        std::cout << "[main]   -d <video_device>    Specify the video device to capture frames from (default: /dev/video16)" << std::endl;
        return 0;
    }

    const char* videoDevice;
    if (args.find("-d") != args.end()) {
        videoDevice = args["-d"].c_str();
    } else {
        videoDevice = DEFAULT_VIDEO_DEVICE;
    }
    std::cout << "[main] Using video device: " << videoDevice << std::endl;

    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    run_spinnaker_stream(videoDevice);
}
