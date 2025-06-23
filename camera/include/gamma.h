#ifndef GAMMA_H
#define GAMMA_H

class PIDGammaController
{
public:
    PIDGammaController(
        double Kp,
        double Ki,
        double Kd,
        double gamma_min = 0.5,
        double gamma_max = 2.0,
        double dt = 1.0 / 60.0);
    double update(double Y_current, double Y_target, double gamma_current);
    void reset();

private:
    double Kp_, Ki_, Kd_;
    double gamma_min_, gamma_max_;
    double dt_;
    double integral_;
    double prev_error_;
};

double computeROImeanY(const unsigned char *yuyv422_frame,
                       const int height,
                       const int width,
                       const int roi_height,
                       const int roi_width);

#endif // GAMMA_H