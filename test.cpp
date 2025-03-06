#include <opencv2/opencv.hpp>

#include "fisheye.h"
#include "homography.h"
#include "util.h"

using namespace std;
using namespace cv;

int main()
{
    const Fisheye camera("fisheye_calibration.yaml");
    const Homography homography("homography_calibration.yaml");
    const Mat img = imread("fisheye_test/test.png");
    float j = 0;
    bool flag = true;
    vector<Point2f> lines_;
    vector<Point2f> _lines_(600);
    while (true)
    {
        if (j > 50)
            flag = false;
        if (j < -50)
            flag = true;
        if (flag)
            j += 2;
        else
            j -= 2;
        const vector<Point2f> line_left = create_curve(Point2f(1511, 2047), Point2f(1511, 1663), Point2f(1511 + j, 1280), 300);
        const vector<Point2f> line_right = create_curve(Point2f(1561, 2047), Point2f(1561, 1663), Point2f(1561 + j, 1280), 300);
        vector<Point2f> lines = line_left;
        lines.insert(lines.end(), line_right.begin(), line_right.end());
        homography.projectPoints(lines, lines_);
        camera.distortPoints(lines_, _lines_);
        Mat img_ = img.clone();
        img_ += _lines_;
        resize(img_, img_, Size(1024, 768));
        imshow("img", img_);
        waitKey(100);
    }
}