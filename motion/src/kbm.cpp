//
// Created by Yunhao Xu on 25-5-6.
//
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "kbm.h"

std::vector<cv::Point2f> kbm(
    float delta
)
{
    float kappa = std::tan(delta) / VEHICLE_L;
    float ds = KBM_S / static_cast<float>(KBM_N);

    std::vector<cv::Point2f> pts;
    pts.reserve(KBM_N + 1);

    float x = 0.0f;
    float y = 0.0f;
    float psi = 0.0f;

    pts.emplace_back(x, y);

    // Euler integration
    for (int i = 0; i < KBM_N; ++i)
    {
        y += std::cos(psi) * ds;
        x += std::sin(psi) * ds;
        psi += kappa * ds;

        pts.emplace_back(x, y);
    }

    return pts;
}

std::vector<cv::Point2f> kbm2(
    float delta
)
{
    float kappa = std::tan(delta) / VEHICLE_L;
    float ds = KBM_S / static_cast<float>(KBM_N);

    std::vector<cv::Point2f> pts;
    pts.reserve(KBM_N + 1);

    float x = 0.0f;
    float y = 0.0f;
    float psi = 0.0f;
    constexpr double length = 1;

    pts.emplace_back(x - length, y);
    pts.emplace_back(x + length, y);

    // Euler integration
    for (int i = 0; i < KBM_N; ++i)
    {
        y += std::cos(psi) * ds;
        x += std::sin(psi) * ds;
        psi += kappa * ds;

        double x_l = x - length * cos(psi);
        double y_l = y + length * sin(psi);
        double x_r = x + length * cos(psi);
        double y_r = y - length * sin(psi);

        pts.emplace_back(x_l, y_l);
        pts.emplace_back(x_r, y_r);
    }

    return pts;
}
