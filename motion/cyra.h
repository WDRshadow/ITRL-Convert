#ifndef CYRA_H
#define CYRA_H

#define RCVE_RATIO 4.1
#define RCVE_WHBASE 2.7
#define THETA0 0.0

/**
* State
* x: x-coordinate (relative to the vehicle origin)
* y: y-coordinate
* theta: heading angle (orientation of the vehicle)
*/
struct State
{
    double x;
    double y;
    double theta;
};

/**
 * CYRA Model Prediction
 * @param v0 velocity (m/s)
 * @param a0 acceleration (m/s²)
 * @param omega0 yaw rate (rad/s)
 * @param theta0 initial heading angle (rad)
 * @param t prediction time (s)
 * @return predicted state
 */
State predictCYRA(double v0, double a0, double omega0, double theta0, double t);

/**
* Bicycle Model Prediction
* @param v velocity (m/s)
* @param omega_sw steering wheel angle (rad)
* @param wheelbases distance between the front and rear axles (m)
* @param ratio ratio between the steering wheel angle and the front wheel angle
*/
double bycicleModel(double v, double omega_sw, double wheelbases, double ratio);

#endif //CYRA_H
