#include <iostream>
#include <gtest/gtest.h>

#include "bayerRG2rgb.h"
#include "rgb2yuyv.h"
#include "cuda_stream.h"

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(CUDA, H_MEM)
{
    int width  = 3072;
    int height = 2048;

    unsigned char* h_mem = get_cuda_buffer(width * height);

    for(int i = 0; i < width * height; ++i)
    {
        h_mem[i] = static_cast<unsigned char>(rand() % 256);
    }

    std::cout << "[cuda] Memory allocation done. Size = " << width * height << " bytes." << std::endl;

    free_cuda_buffer(h_mem);
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

    std::cout << "[cuda] Conversion done. Output RGB size = " << width * height * 3 << " bytes." << std::endl;

    delete[] bayerHost;
    delete[] rgbHost;
}

TEST(CUDA, RGB_YUYV)
{
    int width  = 3072;
    int height = 2048;

    unsigned char* rgbHost = new unsigned char[width * height * 3];
    unsigned char* yuyvHost = new unsigned char[width * height * 2];

    for(int i = 0; i < width * height * 3; ++i)
    {
        rgbHost[i] = static_cast<unsigned char>(rand() % 256);
    }
    
    init_rgb2yuyv_cuda(width, height, 8);
    rgb2yuyv_cuda(rgbHost, yuyvHost);
    cleanup_rgb2yuyv_cuda();

    std::cout << "[cuda] Conversion done. Output YUYV size = " << width * height * 2 << " bytes." << std::endl;

    delete[] rgbHost;
    delete[] yuyvHost;
}

TEST(CUDA, BAYER_YUYV)
{
    int width  = 3072;
    int height = 2048;

    unsigned char* bayerHost = new unsigned char[width * height];
    unsigned char* yuyvHost  = new unsigned char[width * height * 2];

    for(int i = 0; i < width * height; ++i)
    {
        bayerHost[i] = static_cast<unsigned char>(rand() % 256);
    }
    
    init_bayer2yuyv_cuda(width, height, 8);
    bayer2yuyv_cuda(bayerHost, yuyvHost);
    cleanup_bayer2yuyv_cuda();

    std::cout << "[cuda] Conversion done. Output YUYV size = " << width * height * 2 << " bytes." << std::endl;

    delete[] bayerHost;
    delete[] yuyvHost;
}