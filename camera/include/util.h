#ifndef UTIL_H
#define UTIL_H

#include <opencv2/opencv.hpp>

#include "fisheye.h"
#include "homography.h"

using namespace std;
using namespace cv;

void operator+=(Mat& lhs, const vector<Point2f>& rhs);

vector<Point2f> create_line(Point2f start, Point2f end, int num);

vector<Point2f> create_curve(Point2f start, Point2f control, Point2f end, int num);

void draw_points(const unsigned char *src, unsigned char *sol, int width, int height, const vector<Point2f>& points, const Fisheye& fisheye, const Homography& homography);

#endif //UTIL_H