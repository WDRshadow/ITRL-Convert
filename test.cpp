#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>

#include "convert_rgb24_to_yuyv_cuda.h"

extern "C"
{

    int configure_video_device(int video_fd, int width, int height, __u32 pixel_format)
    {
        struct v4l2_format vfmt = {};
        vfmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
        vfmt.fmt.pix.width = width;
        vfmt.fmt.pix.height = height;
        vfmt.fmt.pix.pixelformat = pixel_format;
        vfmt.fmt.pix.field = V4L2_FIELD_NONE;
        switch (pixel_format)
        {
        case V4L2_PIX_FMT_RGB24:
            vfmt.fmt.pix.bytesperline = width * 3;
            vfmt.fmt.pix.sizeimage = width * height * 3;
            break;
        case V4L2_PIX_FMT_YUYV:
            vfmt.fmt.pix.bytesperline = width * 2;
            vfmt.fmt.pix.sizeimage = width * height * 2;
            break;
        default:
            std::cerr << "Unsupported pixel format" << std::endl;
            return -1;
        }

        if (ioctl(video_fd, VIDIOC_S_FMT, &vfmt) < 0)
        {
            std::cerr << "Failed to set video format on virtual device" << std::endl;
            return -1;
        }

        return 0;
    }

    int main()
    {
        unsigned int width = 3072;
        unsigned int height = 2048;
        auto *imageData = new unsigned char[width * height * 3];
        auto *yuyv422 = new unsigned char[width * height * 2];

        int video_fd = open("/dev/video16", O_WRONLY);
        if (video_fd == -1)
        {
            std::cerr << "Failed to open virtual device" << std::endl;
            return -1;
        }

        if (configure_video_device(video_fd, width, height, V4L2_PIX_FMT_YUYV) != 0)
        {
            std::cerr << "Failed to configure virtual device" << std::endl;
            return -1;
        }

        while (true)
        {
            // Randomly set pixels in imageData
            for (int j = 0; j < width * height * 3; j++)
            {
                imageData[j] = rand() % 256;
            }

            // cuda
            convert_rgb24_to_yuyv_cuda(imageData, yuyv422, width, height);

            // Write the YUYV422 (16 bits per pixel) data to the virtual video device as YUYV422
            if (write(video_fd, yuyv422, width * height * 2) == -1)
            {
                std::cerr << "Error writing frame to virtual device" << std::endl;
                break;
            }
        }
    }
}