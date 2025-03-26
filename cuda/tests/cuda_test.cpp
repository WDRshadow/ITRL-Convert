#include <iostream>
#include <gtest/gtest.h>

#include "bayerRG2rgb.h"

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(CUDA, BAYER_RGB)
{
    int width  = 3072;
    int height = 2048;

    unsigned char* bayerHost = new unsigned char[width * height];
    unsigned char* rgbHost   = new unsigned char[width * height * 3];

    for(int i = 0; i < width * height; ++i)
    {
        bayerHost[i] = static_cast<unsigned char>(rand() % 256);
    }
    
    init_bayerRG2rgb_cuda(width, height, 8);
    bayerRG2rgb_cuda(bayerHost, rgbHost);
    cleanup_bayerRG2rgb_cuda();

    std::cout << "Conversion done. Output RGB size = " << width * height * 3 << " bytes." << std::endl;

    delete[] bayerHost;
    delete[] rgbHost;
}