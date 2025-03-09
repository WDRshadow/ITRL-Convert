#include <opencv2/opencv.hpp>

#include "util.h"
#include "fisheye.h"
#include "homography.h"

using namespace std;
using namespace cv;

void operator+=(Mat& lhs, const vector<Point2f>& rhs)
{
    for (const auto& p : rhs)
    {
        circle(lhs, p, 3, Scalar(255, 255, 255), FILLED);
    }
}

vector<Point2f> create_line(Point2f start, Point2f end, int num)
{
    vector<Point2f> line;
    line.reserve(num);
    for (int i = 0; i < num; i++)
    {
        line.push_back(start + (end - start) * i / num);
    }
    return line;
}

vector<Point2f> create_curve(Point2f start, Point2f control, Point2f end, int num)
{
    vector<Point2f> curve;
    curve.reserve(num);
    for (int i = 0; i <= num; i++)
    {
        float t = static_cast<float>(i) / static_cast<float>(num);
        float u = 1 - t;
        Point2f point = u * u * start + 2 * u * t * control + t * t * end;
        curve.push_back(point);
    }
    return curve;
}

void draw_points(const unsigned char *src, unsigned char *sol, int width, int height, const vector<Point2f>& points, const Fisheye& fisheye, const Homography& homography)
{
    static vector<Point2f> points_;
    static vector<Point2f> _points_(points.size());
    auto img = Mat(Size(width, height), CV_8UC3, (void*)src);
    homography.projectPoints(points, points_);
    fisheye.distortPoints(points_, _points_);
    img += _points_;
    memcpy(sol, img.data, width * height * 3);
}
