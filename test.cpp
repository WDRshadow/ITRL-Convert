#include <opencv2/opencv.hpp>
#include <ncurses.h>

#include "fisheye.h"
#include "homography.h"
#include "util.h"

using namespace std;
using namespace cv;

[[noreturn]] int main()
{
    const Fisheye camera("fisheye_calibration.yaml");
    const Homography homography("homography_calibration.yaml");
    const Mat img = imread("test/test.png");
    float j = 0;
    vector<Point2f> lines_;
    vector<Point2f> _lines_(600);

    initscr();
    cbreak();
    noecho();
    nodelay(stdscr, TRUE);

    while (true)
    {
        int ch = getch();
        if (j > -50 && ch == 'a' || ch == 'A')
        {
            j -= 5;
        }
        else if (j < 50 && ch == 'd' || ch == 'D')
        {
            j += 5;
        }
        else if (ch == 'q' || ch == 'Q')
        {
            break;
        }
        const vector<Point2f> line_left = create_curve(Point2f(1511, 2047), Point2f(1511, 1663), Point2f(1511 + j, 1280), 300);
        const vector<Point2f> line_right = create_curve(Point2f(1561, 2047), Point2f(1561, 1663), Point2f(1561 + j, 1280), 300);
        vector<Point2f> lines = line_left;
        lines.insert(lines.end(), line_right.begin(), line_right.end());
        homography.projectPoints(lines, lines_);
        camera.distortPoints(lines_, _lines_);
        Mat img_ = img.clone();
        img_ += _lines_;
        resize(img_, img_, Size(1024, 768));
        imshow("img", img_);
        waitKey(100);
    }

    endwin();
}