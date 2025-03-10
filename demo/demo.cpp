#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

#include "component.h"
#include "sensor.h"

using namespace std;

int main()
{
    VideoCapture cap("data/demo.mp4");
    if (!cap.isOpened())
    {
        cerr << "Error: cannot open the video file" << endl;
        return -1;
    }
    Mat frame;

    const SensorBuffer str_whe_phi("data/str_whe_phi.csv");
    const SensorBuffer vel("data/vel.csv");

    const auto interpolation_51_100 = [](int x) -> int
    {
        return (x * 100 + 50) / 51;
    };

    StreamImage stream_image(3072, 2048);
    const auto driver_line = make_shared<DriverLine>("data/fisheye_calibration.yaml",
                                                     "data/homography_calibration.yaml");
    const auto velocity = make_shared<TextComponent>(1536, 1462, 200, 200);
    stream_image.add_component("driver_line", driver_line);
    stream_image.add_component("velocity", velocity);

    cap.set(CAP_PROP_POS_FRAMES, 8000);
    constexpr int start = 15293;
    while (true)
    {
        static int count = 0;
        int index = interpolation_51_100(count++);
        if (start + index >= str_whe_phi.size())
        {
            break;
        }
        cap >> frame;
        if (frame.empty())
        {
            break;
        }
        // component update -----------------------------------
        driver_line->update(str_whe_phi.get_value(start + index));
        velocity->update(to_string(static_cast<int>(vel.get_value(start + index))));
        stream_image.update(frame);
        // ----------------------------------------------------
        imshow("frame", frame);
        waitKey(20);
    }
    destroyWindow("frame");
    return 1;
}
