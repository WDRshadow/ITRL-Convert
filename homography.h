//
// Created by Yunhao Xu on 25-2-27.
//

#ifndef HOMOGRAPHY_H
#define HOMOGRAPHY_H

#include <opencv2/opencv.hpp>

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

inline Homography::Homography(const vector<Point2f> *srcPoints, const vector<Point2f> *dstPoints)
{
    if (srcPoints->size() != dstPoints->size() || srcPoints->size() < 4)
    {
        cerr << "Error: There must be at least 4 points and the number of source and destination points must be the same." << endl;
        return;
    }
    H = findHomography(*srcPoints, *dstPoints, RANSAC);
}

inline Homography::Homography(const String &filename)
{
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened())
    {
        cerr << "Cannot open file " << filename << endl;
        return;
    }
    fs["H"] >> H;
    fs.release();
}

inline void Homography::save(const String &filename) const
{
    FileStorage fs(filename, FileStorage::WRITE);
    fs << "H" << H;
    fs.release();
}

inline void Homography::projectPoints(const vector<Point2f> &srcPoints, vector<Point2f> &dstPoints) const
{
    if (H.empty())
    {
        cerr << "Error: Homography matrix is empty." << endl;
        return;
    }
    perspectiveTransform(srcPoints, dstPoints, H);
}

#endif // HOMOGRAPHY_H
