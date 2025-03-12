#include <opencv2/opencv.hpp>

#include "fisheye.h"

int main()
{
    const Size boardSize(10, 7);
    constexpr float squareSize = 0.025f;
    const String filename = "data/*.jpg";
    const Fisheye fisheye(boardSize, squareSize, &filename);
    fisheye.save("fisheye_calibration.yaml");
}