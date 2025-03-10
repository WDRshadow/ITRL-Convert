#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

#include "component.h"
#include "sensor.h"

using namespace std;

int main()
{
    VideoCapture cap("test/demo.mp4");
    if (!cap.isOpened())
    {
        cerr << "Error: cannot open the video file" << endl;
        return -1;
    }

    const SensorBuffer str_whe_phi("test/str_whe_phi.csv");
    const SensorBuffer vel("test/vel.csv");

    const auto interpolation_51_100 = [](int x) -> int {
        return (x * 100 + 50) / 51;
    };

    StreamImage stream_image(3072, 2048);
    const auto driver_line = make_shared<DriverLine>("fisheye_calibration.yaml", "homography_calibration.yaml");
    const auto velocity = make_shared<Velocity>(1536, 1462);
    stream_image.add_component("driver_line", driver_line);
    stream_image.add_component("velocity", velocity);

    cap.set(CAP_PROP_POS_FRAMES, 8000);
    constexpr int start = 15293;
    int count = 0;
    while (true)
    {
        Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            break;
        }
        const int index = interpolation_51_100(count);
        driver_line->update({{"str_whe_phi", to_string(str_whe_phi.get_value(start + index))}});
        velocity->update({{"vel", to_string(vel.get_value(start + index))}});
        stream_image.update(frame);
        imshow("frame", frame);
        waitKey(20);
        count++;
    }
    destroyWindow("frame");
    return 1;
}