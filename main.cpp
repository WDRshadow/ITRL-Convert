#include <iostream>
#include <unordered_map>
#include <string>
#include <opencv2/opencv.hpp>

#include "spinnaker_stream.h"
#include "fisheye.h"
#include "homography.h"

#define DEFAULT_VIDEO_DEVICE "/dev/video16"
#define DEFAULT_IP "0.0.0.0"
#define DEFAULT_PORT 10086

void run_spinnaker_stream(const char* videoDevice, const char* ip, int port) {
    std::cout << "[main] Starting to capture frames from the FLIR camera..." << std::endl;
    capture_frames(videoDevice, ip, port);
    std::cout << "[main] Capture process finished" << std::endl;
}

void run_fc()
{
    const Size boardSize(10, 7);
    constexpr float squareSize = 0.025f;
    const String filename = "data/*.jpg";
    const Fisheye fisheye(boardSize, squareSize, &filename);
    fisheye.save("fisheye_calibration.yaml");
}

void run_fu(const std::string& filename)
{
    const Fisheye camera("fisheye_calibration.yaml");
    Mat image = imread(filename);
    camera.undistort(image, image);
    imwrite("undistorted.jpg", image);
}

void run_hc()
{
    const vector<Point2f> src{{1510, 2047}, {1560, 2047}, {1510, 797}, {1560, 797}};
    vector<Point2f> dst;
    FileStorage fs("homography_points.yaml", FileStorage::READ);
    fs["points"] >> dst;
    const Homography homography(&src, &dst);
    homography.save("homography_calibration.yaml");
}

std::unordered_map<std::string, std::string> parseArguments(int argc, char* argv[]) {
    std::unordered_map<std::string, std::string> args;

    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key.rfind("-", 0) == 0) {
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

    if (args.find("-fc") != args.end()) {
        run_fc();
        return 0;
    }

    if (args.find("-fu") != args.end()) {
        const std::string filename = args["-fu"];
        if (filename.empty()) {
            std::cerr << "[main] Usage: " << argv[0] << " <image>" << std::endl;
            return -1;
        }
        run_fu(args["-fu"]);
        return 0;
    }

    if (args.find("-hc") != args.end()) {
        run_hc();
        return 0;
    }

    const char* videoDevice;
    if (args.find("-dev") != args.end()) {
        videoDevice = args["-dev"].c_str();
    } else {
        videoDevice = DEFAULT_VIDEO_DEVICE;
    }
    std::cout << "[main] Using video device: " << videoDevice << std::endl;

    const char* ip;
    int port;
    if (args.find("-s") != args.end()) {
        if (args.find("-ip") != args.end()) {
            ip = args["-ip"].c_str();
        } else {
            ip = DEFAULT_IP;
        }
        if (args.find("-p") != args.end()) {
            port = std::stoi(args["-p"]);
        } else {
            port = DEFAULT_PORT;
        }
        std::cout << "[main] Using IP: " << ip << " and port: " << port << std::endl;
    } else {
        ip = "0.0.0.0";
        port = -1;
    }

    run_spinnaker_stream(videoDevice, ip, port);
    return 0;
}
