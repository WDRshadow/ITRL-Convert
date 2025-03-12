#ifndef COMPONENT_H
#define COMPONENT_H
#include <unordered_map>
#include <memory>
#include <opencv2/opencv.hpp>

#include "fisheye.h"
#include "homography.h"

using namespace std;
using namespace cv;

class Component;

class StreamImage
{
    unordered_map<string, shared_ptr<Component>> components;
    const int width;
    const int height;

public:
    StreamImage(int width, int height);
    void add_component(const string& name, const shared_ptr<Component>& component);
    void update(const unsigned char* imageData);
    void update(Mat& imageData);
};

class Component
{
    friend class StreamImage;
    [[nodiscard]] shared_ptr<Mat> get_img();

protected:
    const int cx;
    const int cy;
    const int width;
    const int height;
    shared_ptr<Mat> img;
    void reset_img() const;

public:
    Component(int cx, int cy, int width, int height);
    virtual ~Component() = default;
    virtual void update(const unordered_map<string, string>& arg) = 0;
    [[nodiscard]] int get_center_x() const;
    [[nodiscard]] int get_center_y() const;
};

class DriverLine
{
    vector<Point2f> lines_;
    vector<Point2f> _lines_;
    const Fisheye fisheye_camera;
    const Homography homography_line;

public:
    DriverLine(const string& fisheye_config, const string& homography_config);
    void update(float str_whe_phi);
    void operator>>(unsigned char* imageData);
    void operator>>(Mat& imageData);
};

class TextComponent final : public Component
{
public:
    TextComponent(int x, int y, int width, int height);
    void update(const string& text) const;
    void update(const unordered_map<string, string>& arg) override;
};

#endif //COMPONENT_H
