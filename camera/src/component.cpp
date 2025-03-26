#include "component.h"
#include "util.h"
#include "cyra.h"

StreamImage::StreamImage(const int width, const int height): components({}), width(width), height(height)
{
}

void StreamImage::add_component(const string& name, const shared_ptr<Component>& component)
{
    components.emplace(name, component);
}

void StreamImage::operator>>(const unsigned char* imageData) const
{
    Mat img(height, width, CV_8UC3, const_cast<unsigned char*>(imageData));
    operator>>(img);
}

void StreamImage::operator>>(Mat& imageData) const
{
    for (const auto& [str, component] : components)
    {
        *component >> imageData;
    }
}

ImageComponent::ImageComponent(const int cx, const int cy, const int width, const int height): cx(cx), cy(cy),
    width(width),
    height(height)
{
    img = Mat::zeros(height, width, CV_8UC3);
}

void ImageComponent::reset_img()
{
    if (!img.empty() && img.size() == Size(width, height) && img.type() == CV_8UC3)
    {
        img.setTo(Scalar(0, 0, 0));
    }
    else
    {
        img.create(height, width, CV_8UC3);
        img.setTo(Scalar(0, 0, 0));
    }
}

void ImageComponent::operator>>(Mat& imageData) const
{
    overlay_image(imageData, Point(cx, cy), img);
}

LineComponent::LineComponent(const string& fisheye_config, const string& homography_config, const int width,
                             const int height): width(width), height(height), lines_({}),
                                                fisheye_camera(Fisheye(fisheye_config)),
                                                homography_line(Homography(homography_config))
{
}

void LineComponent::operator>>(Mat& imageData) const
{
    imageData += lines_;
}

void LineComponent::project(const vector<Point2f>& lines)
{
    homography_line.projectPoints(lines, lines_);
    fisheye_camera.distortPoints(lines_, lines_);
}

DriverLine::DriverLine(const string& fisheye_config, const string& homography_config, const int width,
                       const int height):
    LineComponent(fisheye_config, homography_config, width, height)
{
}

void DriverLine::update(const float str_whe_phi)
{
    const float angle = str_whe_phi / 4.1f;
    const vector<Point2f> line_left = create_curve(Point2f(1511, 2047), Point2f(1511, 1663),
                                                   Point2f(1511 + 125 * tan(angle), 1280), 300);
    const vector<Point2f> line_right = create_curve(Point2f(1561, 2047), Point2f(1561, 1663),
                                                    Point2f(1561 + 125 * tan(angle), 1280), 300);
    vector<Point2f> lines = line_left;
    lines.insert(lines.end(), line_right.begin(), line_right.end());
    project(lines);
}

PredictionLine::PredictionLine(const string& fisheye_config, const string& homography_config, const int width,
                               const int height):
    LineComponent(fisheye_config, homography_config, width, height)
{
}

void PredictionLine::update(const float v, const float a, float str_whe_phi_remote, const float str_whe_phi_local, const float latency)
{
    const double omega = bycicleModel(v, str_whe_phi_remote / 20.0f, RCVE_WHBASE, RCVE_RATIO);
    const auto [x, y, theta] = predictCYRA(v, a, omega, THETA0, latency);
    vector<Point2f> lines = create_line({
                                            static_cast<float>(1536 + y * 25),
                                            static_cast<float>(2047 - x * 25)
                                        },
                                        theta, 25, 50);
    // --------------------------------------------------------------------------------------------
    const auto cos_theta = static_cast<float>(cos(theta));
    const auto sin_theta = static_cast<float>(sin(theta));
    const float x_l = 1536 + static_cast<float>(y) * 25 - 25 * cos_theta;
    const float y_l = 2047 - static_cast<float>(x) * 25 - 25 * sin_theta;
    const float x_r = 1536 + static_cast<float>(y) * 25 + 25 * cos_theta;
    const float y_r = 2047 - static_cast<float>(x) * 25 + 25 * sin_theta;
    const float angle = str_whe_phi_local / 4.1f;
    const vector<Point2f> line_left = create_curve(
        {x_l, y_l},
        {x_l + 384 * sin_theta, y_l - 384 * cos_theta},
        {
            x_l + 768 * sin_theta + 125 * tan(angle) * cos_theta,
            y_l - 768 * cos_theta + 125 * tan(angle) * sin_theta
        },
        300
    );
    const vector<Point2f> line_right = create_curve(
        {x_r, y_r},
        {x_r + 384 * sin_theta, y_r - 384 * cos_theta},
        {
            x_r + 768 * sin_theta + 125 * tan(angle) * cos_theta,
            y_r - 768 * cos_theta + 125 * tan(angle) * sin_theta
        },
        300
    );
    lines.insert(lines.end(), line_left.begin(), line_left.end());
    lines.insert(lines.end(), line_right.begin(), line_right.end());
    // --------------------------------------------------------------------------------------------
    project(lines);
}

TextComponent::TextComponent(const int x, const int y, const int width, const int height): ImageComponent(
    x, y, width, height)
{
}

void TextComponent::update(const string& text)
{
    reset_img();
    draw_text(text, width, height).copyTo(img);
}
