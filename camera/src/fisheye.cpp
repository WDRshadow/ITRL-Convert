#include <opencv2/opencv.hpp>

#include "fisheye.h"

using namespace std;
using namespace cv;

Fisheye::Fisheye(Size boardSize, float squareSize, const String* filename)
{
    vector<string> imageFiles;
    glob(*filename, imageFiles);

    if (imageFiles.empty())
    {
        cout << "No images found." << endl;
        return;
    }

    vector<vector<Point3f>> objectPoints;
    vector<vector<Point2f>> imagePoints;

    vector<Point3f> obj;
    for (int i = 0; i < boardSize.height; i++)
    {
        for (int j = 0; j < boardSize.width; j++)
        {
            obj.emplace_back(static_cast<float>(j) * squareSize, static_cast<float>(i) * squareSize, 0);
        }
    }

    Size imageSize;
    for (const auto& imageFile : imageFiles)
    {
        Mat img = imread(imageFile);
        if (img.empty())
        {
            cout << "Cannot load image: " << imageFile << endl;
            continue;
        }
        imageSize = img.size();

        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);

        vector<Point2f> corners;
        bool found = findChessboardCorners(gray, boardSize, corners,
                                           CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK |
                                           CALIB_CB_NORMALIZE_IMAGE);
        if (found)
        {
            cornerSubPix(gray, corners, Size(3, 3), Size(-1, -1),
                         TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
            imagePoints.push_back(corners);
            objectPoints.push_back(obj);
        }
        else
        {
            cout << "No Chessboard detected: " << imageFile << endl;
        }
    }

    if (imagePoints.empty())
    {
        cout << "No enough data for criteria." << endl;
        return;
    }

    K = Mat::eye(3, 3, CV_64F);
    D = Mat::zeros(4, 1, CV_64F);
    vector<Mat> rvecs, tvecs;

    TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-6);

    cout << "Number of object points: " << objectPoints.size() << endl;
    cout << "Number of image points: " << imagePoints.size() << endl;
    cout << "Image size: " << imageSize << endl;

    double rms = calibrate(objectPoints, imagePoints, imageSize, K, D, rvecs, tvecs,
                           fisheye::CALIB_RECOMPUTE_EXTRINSIC, criteria);

    cout << "RMS: " << rms << endl;
    cout << "K: " << endl << K << endl;
    cout << "D: " << endl << D << endl;
}

Fisheye::Fisheye(const String& filename)
{
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened())
    {
        cout << "Cannot open file " << filename << endl;
        return;
    }
    fs["K"] >> K;
    fs["D"] >> D;
    fs.release();
}

void Fisheye::save(const String& filename) const
{
    cout << "K: " << endl << K << endl;
    cout << "D: " << endl << D << endl;

    // 将标定结果保存到文件中
    FileStorage fs(filename, FileStorage::WRITE);
    fs << "K" << K;
    fs << "D" << D;
    fs.release();

    cout << "Parameters save to " << filename << endl;
}

void Fisheye::undistort(const Mat& src, Mat& sol) const
{
    fisheye::undistortImage(src, sol, K, D, K);
}

void Fisheye::distortPoints(const vector<Point2f>& src, vector<Point2f>& sol) const
{
    static const double fx = K.at<double>(0, 0);
    static const double fy = K.at<double>(1, 1);
    static const double cx = K.at<double>(0, 2);
    static const double cy = K.at<double>(1, 2);

    static vector<Point2f> src_(src.size());

    transform(src.begin(), src.end(), src_.begin(),
        [](const Point2f& p) -> Point2f {
        return {static_cast<float>((p.x - cx) / fx), static_cast<float>((p.y - cy) / fy)};
    });

    fisheye::distortPoints(src_, sol, K, D);
}