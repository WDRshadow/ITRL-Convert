#include <cmath>

#include "cyra.h"

State predictCYRA(const double v0, const double a0, const double omega0, const double theta0, const double t)
{
    State state{};
    if (std::fabs(omega0) < 1e-6)
    {
        double displacement = v0 * t + 0.5 * a0 * t * t;
        state.x = displacement * std::cos(theta0);
        state.y = displacement * std::sin(theta0);
    }
    else
    {
        double theta_t = theta0 + omega0 * t;
        double sinTheta0 = std::sin(theta0);
        double sinThetaT = std::sin(theta_t);
        double cosTheta0 = std::cos(theta0);
        double cosThetaT = std::cos(theta_t);

        state.x = v0 / omega0 * (sinThetaT - sinTheta0)
            + a0 / (omega0 * omega0) * (omega0 * t * sinThetaT + cosThetaT - cosTheta0);

        state.y = -(v0 / omega0) * (cosThetaT - cosTheta0)
            + a0 / (omega0 * omega0) * (-omega0 * t * cosThetaT + sinThetaT - sinTheta0);
    }

    state.theta = theta0 + omega0 * t;

    return state;
}

double bycicleModel(const double v, const double omega_sw, const double wheelbases, const double ratio)
{
    return v * std::tan(omega_sw / ratio) / wheelbases;
}

