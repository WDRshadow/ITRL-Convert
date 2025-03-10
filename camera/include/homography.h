#ifndef HOMOGRAPHY_H
#define HOMOGRAPHY_H

using namespace std;
using namespace cv;

class Homography
{
    Mat H;

public:
    Homography(const vector<Point2f> *srcPoints, const vector<Point2f> *dstPoints);
    explicit Homography(const String &filename);
    void save(const String &filename) const;
    void projectPoints(const vector<Point2f> &srcPoints, vector<Point2f> &dstPoints) const;
};

#endif // HOMOGRAPHY_H
