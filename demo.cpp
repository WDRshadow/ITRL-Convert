#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

#include "fisheye.h"
#include "homography.h"
#include "util.h"

using namespace std;
using namespace cv;

int main()
{
    VideoCapture cap("test/test.mp4");
    if (!cap.isOpened())
    {
        cerr << "Error: cannot open the video file" << endl;
        return -1;
    }

    ifstream file("test/test.csv");
    if (!file.is_open()) {
        cerr << "Error: cannot open the file" << endl;
        return -1;
    }

    vector<float> angles_100hz;
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        if (getline(ss, value, ',')) {
            value.erase(remove_if(value.begin(), value.end(), [](char c) { return !isdigit(c) && c != '.' && c != '-'; }), value.end());
            angles_100hz.push_back(stof(value));
        }
    }

    file.close();

    const Fisheye camera("fisheye_calibration.yaml");
    const Homography homography("homography_calibration.yaml");
    vector<Point2f> lines_;
    vector<Point2f> _lines_(600);
    const auto interpolation_51_100 = [](int x) -> int {
        return (x * 100 + 50) / 51;
    };

    cap.set(CAP_PROP_POS_FRAMES, 8000);
    const int start = 15293;
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
        if (index >= angles_100hz.size())
        {
            break;
        }
        float angle = angles_100hz[start + index] / 4.1f;
        const vector<Point2f> line_left = create_curve(Point2f(1511, 2047), Point2f(1511, 1663), Point2f(1511 + 125 * tan(angle), 1280), 300);
        const vector<Point2f> line_right = create_curve(Point2f(1561, 2047), Point2f(1561, 1663), Point2f(1561 + 125 * tan(angle), 1280), 300);
        vector<Point2f> lines = line_left;
        lines.insert(lines.end(), line_right.begin(), line_right.end());
        homography.projectPoints(lines, lines_);
        camera.distortPoints(lines_, _lines_);
        frame += _lines_;
        imshow("frame", frame);
        waitKey(20);
        count++;
    }
    destroyWindow("frame");
    return 1;
}