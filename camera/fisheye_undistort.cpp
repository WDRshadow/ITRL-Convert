#include <opencv2/opencv.hpp>

#include "fisheye.h"

int main(int argc, char** arg )
{
    if (argc != 2)
    {
        cout << "Usage: fisheye_undistort <image>" << endl;
        return -1;
    }
    const Fisheye camera("fisheye_calibration.yaml");
    const String filename = arg[1];
    Mat image = imread(filename);
    camera.undistort(image, image);
    imwrite("undistorted.jpg", image);
}