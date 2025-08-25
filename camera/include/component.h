#ifndef COMPONENT_H
#define COMPONENT_H
#include <unordered_map>
#include <memory>
#include <opencv2/opencv.hpp>

#include "fisheye.h"
#include "homography.h"

#define ORIGIN_X 1536
#define ORIGIN_Y 2047
#define PIXELS_PER_METER 25
#define STR_WHE_RATIO 60.0f

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
    void operator>>(const unsigned char* imageData) const;
    void operator>>(Mat& imageData) const;
};

class Component
{
public:
    virtual ~Component() = default;
    virtual void operator>>(Mat& imageData) const = 0;
};

class ImageComponent : public Component
{
protected:
    const int cx;
    const int cy;
    const int width;
    const int height;
    Mat img;
    void reset_img();

public:
    ImageComponent(int cx, int cy, int width, int height);
    void operator>>(Mat& imageData) const override;
};

class LineComponent : public Component
{
protected:
    const int width;
    const int height;
    vector<Point2f> lines_;
    const Fisheye fisheye_camera;
    const Homography homography_line;
    void project(const vector<Point2f>& lines);

public:
    LineComponent(const string& fisheye_config, const string& homography_config, int width, int height);
    void operator>>(Mat& imageData) const override;
};

class DriverLine final : public LineComponent
{
public:
    DriverLine(const string& fisheye_config, const string& homography_config, int width, int height);
    void update(float str_whe_phi);
};

class PredictionLine final : public LineComponent
{
public:
    PredictionLine(const string& fisheye_config, const string& homography_config, int width, int height);
    void update(float v, float a, float str_whe_phi_remote, float str_whe_phi_local, float latency);
};

class TextComponent final : public ImageComponent
{
public:
    TextComponent(int x, int y, int width, int height);
    void update(const string& text);
};

class ImageComponent_2 final : public ImageComponent
{
    const int origin_width;
    const int origin_height;
public:
    ImageComponent_2(int x, int y, int width, int height, int origin_width, int origin_height);
    void update(const unsigned char* imageData);
};

#endif //COMPONENT_H
