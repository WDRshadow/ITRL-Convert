//
// Created by Yunhao Xu on 25-5-6.
//
#ifndef KBM_H
#define KBM_H
#include <opencv2/opencv.hpp>

#define VEHICLE_L 2.6
#define KBM_S 25
#define KBM_N 300

std::vector<cv::Point2f> kbm(
    float delta
);

std::vector<cv::Point2f> kbm2(
    float delta
);

#endif //KBM_H
