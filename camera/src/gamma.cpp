#include "gamma.h"

#include <algorithm>
#include <cstdint>
#include <cassert>

PIDGammaController::PIDGammaController(
    double Kp,
    double Ki,
    double Kd,
    double gamma_min,
    double gamma_max,
    double dt)
    : Kp_(Kp), Ki_(Ki), Kd_(Kd),
      gamma_min_(gamma_min), gamma_max_(gamma_max),
      dt_(dt), integral_(0.0), prev_error_(0.0) {}

double PIDGammaController::update(double Y_current, double Y_target, double gamma_current)
{
    double error = Y_target - Y_current;
    integral_ += error * dt_;
    double derivative = (error - prev_error_) / dt_;

    double delta = Kp_ * error + Ki_ * integral_ + Kd_ * derivative;
    double gamma_next = gamma_current + delta;

    gamma_next = std::clamp(gamma_next, gamma_min_, gamma_max_);

    prev_error_ = error;

    return gamma_next;
}

void PIDGammaController::reset()
{
    integral_ = 0.0;
    prev_error_ = 0.0;
}

double computeROImeanY(const unsigned char *yuyv422_frame,
                       const int height,
                       const int width,
                       const int roi_height,
                       const int roi_width)
{
    constexpr int bytes_per_pixel = 2;
    assert(width > 0 && "Width must be positive");

    if (roi_height <= 0 || roi_width <= 0 ||
        roi_height > height || roi_width > width)
    {
        return -1.0;
    }

    const int start_row = (height - roi_height) / 2;
    const int start_col = (width - roi_width) / 2;

    uint64_t sumY = 0;
    const uint8_t *bytes = reinterpret_cast<const uint8_t *>(yuyv422_frame);

    for (int r = 0; r < roi_height; ++r)
    {
        int row = start_row + r;
        int base_index = row * width;
        for (int c = 0; c < roi_width; ++c)
        {
            int col = start_col + c;
            int pixel_index = base_index + col;
            int byte_offset = pixel_index * bytes_per_pixel;
            uint8_t Y = bytes[byte_offset];
            sumY += Y;
        }
    }

    int64_t count = int64_t(roi_height) * roi_width;
    double meanY = static_cast<double>(sumY) / static_cast<double>(count);

    return meanY;
}
