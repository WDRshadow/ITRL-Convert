#ifndef COMPONENT_H
#define COMPONENT_H
#include <fisheye.h>
#include <homography.h>
#include <unordered_map>
#include <memory>

using namespace std;

namespace cv
{
    class Mat;
    template <typename T>
    class Point_;
    typedef Point_<float> Point2f;
}

class Component;

class StreamImage
{
    unordered_map<string, shared_ptr<Component>> components;
    const int width;
    const int height;

public:
    StreamImage(int width, int height);
    void add_component(const string& name, const shared_ptr<Component>& component);
    void update(const unsigned char* imageData, unsigned char* output);
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

class DriverLine final : public Component
{
    unique_ptr<vector<Point2f>> lines_;
    unique_ptr<vector<Point2f>> _lines_;
    const Fisheye fisheye_camera;
    const Homography homography_line;

public:
    DriverLine(const string& fisheye_config, const string& homography_config);
    void update(const unordered_map<string, string>& arg) override;
};

class Velocity final : public Component
{
public:
    Velocity(int x, int y);
    void update(const unordered_map<string, string>& arg) override;
};

#endif //COMPONENT_H
