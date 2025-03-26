#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

#include "component.h"
#include "demo_sensor.h"

class RingBuffer
{
    Mat* buffer = new Mat[25];
    int current = 0;

public:
    ~RingBuffer()
    {
        delete[] buffer;
    }

    [[nodiscard]] Mat* get_newest()
    {
        return buffer + current;
    }

    [[nodiscard]] Mat* get_oldest() const
    {
        return buffer + (current + 2) % 25;
    }

    void operator<<(VideoCapture& cap)
    {
        cap >> buffer[(current + 1) % 25];
        current = (current + 1) % 25;
    }
};

int main()
{
    VideoCapture cap("data/demo.mp4");
    if (!cap.isOpened())
    {
        cerr << "[demo] cannot open the video file" << endl;
        return -1;
    }

    RingBuffer buffer;
    Mat* previous = nullptr;

    const demo_sensor str_whe_phi("data/str_whe_phi.csv");
    const demo_sensor vel("data/vel.csv");
    const demo_sensor ax("data/ax.csv");

    const auto interpolation_51_100 = [](const int x) -> int
    {
        return (x * 100 + 50) / 51;
    };

    StreamImage stream_image(3072, 2048);
    const auto prediction_line = make_shared<PredictionLine>("data/fisheye_calibration.yaml",
                                                             "data/homography_calibration.yaml", 3072, 2048);
    const auto velocity = make_shared<TextComponent>(1536, 1462, 200, 200);
    const auto demo_label = make_shared<TextComponent>(1536, 100, 2500, 200);
    const auto latency_label = make_shared<TextComponent>(2800, 100, 500, 200);
    stream_image.add_component("prediction_line", std::static_pointer_cast<Component>(prediction_line));
    stream_image.add_component("demo_label", std::static_pointer_cast<Component>(demo_label));
    stream_image.add_component("latency_label", std::static_pointer_cast<Component>(latency_label));
    stream_image.add_component("velocity", std::static_pointer_cast<Component>(velocity));

    constexpr float latency = 0.5;
    demo_label->update("Demo");
    latency_label->update("Latency: " + to_string(static_cast<int>(latency * 1000)) + " ms");
    cap.set(CAP_PROP_POS_FRAMES, 8000);
    constexpr int start = 15293;
    int count = 0;
    while (true)
    {
        int index = interpolation_51_100(count++);
        if (start + index >= str_whe_phi.size())
        {
            break;
        }
        // component update -----------------------------------
        prediction_line->update(vel.get_value(start + index - static_cast<int>(latency * 100)) * 3.6f,
                                ax.get_value(start + index - static_cast<int>(latency * 100)),
                                str_whe_phi.get_value(start + index - static_cast<int>(latency * 100)),
                                str_whe_phi.get_value(start + index), latency);
        velocity->update(to_string(static_cast<int>(vel.get_value(start + index) * 3.6f)));
        // ----------------------------------------------------
        buffer << cap;
        previous = buffer.get_oldest();
        if (buffer.get_newest()->empty())
        {
            break;
        }
        if (!previous->empty())
        {
            stream_image >> *previous;
            imshow("frame", *previous);
        }
        // imshow("frame2", *buffer.get_newest());
        waitKey(20);
    }
    destroyWindow("frame");
    destroyWindow("frame2");
    return 1;
}
