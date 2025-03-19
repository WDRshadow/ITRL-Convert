#include <iostream>
#include <gtest/gtest.h>

#include "cyra.h"

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(MOTION, CYRA)
{
    double v0 = 10.0;
    double a0 = 1.0;
    double omega0 = -0.1;
    double latency = 2.0;

    State predicted = predictCYRA(v0, a0, omega0, THETA0, latency);

    std::cout << "t = " << latency << " status:" << std::endl;
    std::cout << "x = " << predicted.x << " m" << std::endl;
    std::cout << "y = " << predicted.y << " m" << std::endl;
    std::cout << "theta = " << predicted.theta << " rad" << std::endl;
}

TEST(MOTION, BYCICLE)
{
    double v = 10.0;
    double a0 = 1.0;
    double omega_sw = 0.1;
    double latency = 2.0;

    double omega0 = bycicleModel(v, omega_sw, RCVE_WHBASE, RCVE_RATIO);
    State predicted = predictCYRA(v, a0, omega0, THETA0, latency);

    std::cout << "t = " << latency << " status:" << std::endl;
    std::cout << "x = " << predicted.x << " m" << std::endl;
    std::cout << "y = " << predicted.y << " m" << std::endl;
    std::cout << "theta = " << predicted.theta << " rad" << std::endl;
}