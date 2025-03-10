#ifndef FISHEYE_H
#define FISHEYE_H

using namespace std;
using namespace cv;

class Fisheye
{
    Mat K;
    Mat D;

public:
    Fisheye(Size boardSize, float squareSize, const String* filename);
    explicit Fisheye(const String& filename);
    void save(const String& filename) const;
    void undistort(const Mat& src, Mat& sol) const;
    void distortPoints(const vector<Point2f>& src, vector<Point2f>& sol) const;
};

#endif //FISHEYE_H
