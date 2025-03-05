//
// Created by Yunhao Xu on 25-3-5.
//

#ifndef UTIL_H
#define UTIL_H

#include <opencv2/opencv.hpp>

#include "fisheye.h"
#include "homography.h"

using namespace std;
using namespace cv;

/**
 * Add the white points to the image.
 * @param lhs image
 * @param rhs points
 */
inline void operator+=(Mat& lhs, const vector<Point2f>& rhs)
{
    for (const auto& p : rhs)
    {
        circle(lhs, p, 3, Scalar(255, 255, 255), FILLED);
    }
}

/**
 * Create a line from start to end.
 * @param start start point
 * @param end end point
 * @param num number of points
 * @return the line
 */
inline vector<Point2f> create_line(Point2f start, Point2f end, int num)
{
    vector<Point2f> line;
    line.reserve(num);
    for (int i = 0; i < num; i++)
    {
        line.push_back(start + (end - start) * i / num);
    }
    return line;
}

inline void draw_points(const unsigned char *src, unsigned char *sol, int width, int height, const vector<Point2f>& points, const Fisheye& fisheye, const Homography& homography)
{
    static vector<Point2f> points_;
    static vector<Point2f> _points_(points.size());
    homography.projectPoints(points, points_);
    fisheye.distortPoints(points_, _points_);
    static auto img = Mat(Size(width, height), CV_8UC3, (void*)src);
    img += _points_;
    memcpy(sol, img.data, width * height * 3);
}

#endif //UTIL_H
