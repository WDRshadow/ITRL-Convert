#include <opencv2/opencv.hpp>
#include <memory>

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

void StreamImage::update(const unsigned char* imageData)
{
    Mat img(height, width, CV_8UC3, const_cast<unsigned char*>(imageData));
    for (const auto& [str, component] : components)
    {
        overlay_image(img, Point(component->get_center_x(), component->get_center_y()), *component->get_img());
    }
}

void StreamImage::update(Mat& imageData)
{
    for (const auto& [str, component] : components)
    {
        overlay_image(imageData, Point(component->get_center_x(), component->get_center_y()), *component->get_img());
    }
}

Component::Component(const int cx, const int cy, const int width, const int height): cx(cx), cy(cy), width(width),
    height(height)
{
    img = make_shared<Mat>(Mat::zeros(height, width, CV_8UC3));
}

shared_ptr<Mat> Component::get_img()
{
    return img;
}

void Component::reset_img() const
{
    if (!img->empty() && img->size() == Size(width, height) && img->type() == CV_8UC3)
    {
        img->setTo(Scalar(0, 0, 0));
    }
    else
    {
        img->create(height, width, CV_8UC3);
        img->setTo(Scalar(0, 0, 0));
    }
}


int Component::get_center_x() const
{
    return cx;
}

int Component::get_center_y() const
{
    return cy;
}

LineComponent::LineComponent(const string& fisheye_config, const string& homography_config, const int width,
                             const int height): width(width), height(height), lines_({}),
                                                fisheye_camera(Fisheye(fisheye_config)),
                                                homography_line(Homography(homography_config))
{
}

void LineComponent::operator>>(unsigned char* imageData) const
{
    Mat img(height, width, CV_8UC3, imageData);
    img += lines_;
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

void DriverLine::update(const unordered_map<string, string>& arg)
{
    update(std::stof(arg.find("str_whe_phi")->second));
}

PredictionLine::PredictionLine(const string& fisheye_config, const string& homography_config, const int width,
                               const int height):
    LineComponent(fisheye_config, homography_config, width, height)
{
}

void PredictionLine::update(const float v, const float a, const float str_whe_phi, const float latency)
{
    const double omega = bycicleModel(v, str_whe_phi, RCVE_WHBASE, RCVE_RATIO);
    State predicted = predictCYRA(v, a, omega, THETA0, latency);
    const vector<Point2f> line = create_line({static_cast<float>(1536 + predicted.y * 25), static_cast<float>(2047 - predicted.x * 25)},
                                             predicted.theta, 25, 50);
    project(line);
}

void PredictionLine::update(const unordered_map<string, string>& arg)
{
    update(
        std::stof(arg.find("v")->second),
        std::stof(arg.find("a")->second),
        std::stof(arg.find("str_whe_phi")->second),
        std::stof(arg.find("latency")->second)
    );
}


TextComponent::TextComponent(const int x, const int y, const int width, const int height): Component(
    x, y, width, height)
{
}

void TextComponent::update(const string& text) const
{
    reset_img();
    draw_text(text, width, height).copyTo(*img);
}


void TextComponent::update(const unordered_map<string, string>& arg)
{
    update(arg.find("text")->second);
}
