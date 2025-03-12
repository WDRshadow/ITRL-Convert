#include <opencv2/opencv.hpp>

#include "homography.h"

int main()
{
    const vector<Point2f> src{{1510, 2047}, {1560, 2047}, {1510, 797}, {1560, 797}};
    vector<Point2f> dst;
    FileStorage fs("homography_points.yaml", FileStorage::READ);
    fs["points"] >> dst;
    const Homography homography(&src, &dst);
    homography.save("homography_calibration.yaml");
}