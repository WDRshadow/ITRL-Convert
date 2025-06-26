#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <thread>
#include <sl/Camera.hpp>

#include "zed_stream.h"
#include "formatting.h"

#define CUDA_STREAMS 8

bool is_init = false;
int video_fd;
sl::Mat image;
unsigned char *bayer = nullptr;
unsigned char *yuyv = nullptr;

void capture_frames(const char *video_device, bool &signal)
{
    // Open the virtual V4L2 device
    video_fd = open(video_device, O_WRONLY);
    if (video_fd < 0)
    {
        std::cerr << "[zed stream] Failed to open virtual video device" << std::endl;
        return;
    }

    // Initialize zed
    sl::Camera zed;

    sl::InitParameters init_params;
    init_params.camera_resolution = sl::RESOLUTION::HD1080; // Use HD1080 video mode
    init_params.camera_fps = 60;                            // Set fps at 30

    sl::ERROR_CODE err = zed.open(init_params);
    if (err != sl::ERROR_CODE::SUCCESS)
        exit(-1);

    // Define the converter pointer
    std::unique_ptr<CudaImageConverter> converter_bayer2rgb;
    std::unique_ptr<CudaImageConverter> converter_rgb2yuyv;
    std::unique_ptr<CudaImageConverter> converter_bayer2yuyv;

    unsigned int width;
    unsigned int height;

    while (!signal)
    {
        // Grab an image
        if (zed.grab() == sl::ERROR_CODE::SUCCESS)
        {
            zed.retrieveImage(image, sl::VIEW::LEFT);
        }

        // Print the pixel format only once and initialize the virtual device and CUDA
        if (!is_init)
        {
            // Get the image size
            width = image.getWidth();
            height = image.getHeight();
            std::cout << "[zed stream] Image size: " << width << "x" << height << std::endl;

            // Set the video device format
            struct v4l2_format vfmt = {};
            vfmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
            vfmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
            vfmt.fmt.pix.field = V4L2_FIELD_NONE;
            vfmt.fmt.pix.width = width;
            vfmt.fmt.pix.height = height;
            vfmt.fmt.pix.bytesperline = width * 2;
            vfmt.fmt.pix.sizeimage = width * height * 2;
            if (ioctl(video_fd, VIDIOC_S_FMT, &vfmt) < 0)
            {
                std::cerr << "[zed stream] Failed to set video format on virtual device" << std::endl;
                break;
            }

            // Set the frame rate
            struct v4l2_streamparm streamparm = {};
            streamparm.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
            streamparm.parm.output.timeperframe.numerator = 1;
            streamparm.parm.output.timeperframe.denominator = 60;
            if (ioctl(video_fd, VIDIOC_S_PARM, &streamparm) < 0)
            {
                std::cerr << "[zed stream] Failed to set frame rate" << std::endl;
                break;
            }

            // Initialize CUDA and allocate memory for
            converter_bayer2yuyv = std::make_unique<CudaImageConverter>(width, height, CUDA_STREAMS, BAYER2YUYV);
            yuyv = get_cuda_buffer(width * height * 2);

            std::cout << "[zed stream] Converting to YUYV422 format..." << std::endl;
            is_init = true;
        }

        image.getValue(image.getWidth(), image.getHeight(), bayer);

        converter_bayer2yuyv->convert(bayer, yuyv);

        // Write the YUYV422 (16 bits per pixel) data to the virtual video device as YUYV422
        if (write(video_fd, yuyv, width * height * 2) == -1)
        {
            std::cerr << "[zed stream] Error writing frame to virtual device" << std::endl;
            break;
        }
    }

    // Cleanup
    if (is_init)
    {
        bayer = nullptr;
        free_cuda_buffer(yuyv);
        yuyv = nullptr;
        is_init = false;
    }
    zed.close();
    close(video_fd);
}
