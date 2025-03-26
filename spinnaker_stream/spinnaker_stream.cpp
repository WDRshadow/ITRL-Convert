#include <Spinnaker.h>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <thread>

#include "spinnaker_stream.h"
#include "cuda_stream.h"

#define BUFFER_SIZE 8192
#define CUDA_STREAMS 8

bool is_init = false;
int video_fd;
Spinnaker::SystemPtr system_c = nullptr;
Spinnaker::CameraPtr camera = nullptr;
Spinnaker::CameraList camList;
Spinnaker::ImagePtr pImage = nullptr;
unsigned char* imageData = nullptr;
unsigned char* yuyv422 = nullptr;

void capture_frames(const char* video_device)
{
    // Open the virtual V4L2 device
    video_fd = open(video_device, O_WRONLY);
    if (video_fd < 0)
    {
        std::cerr << "[spinnaker stream] Failed to open virtual video device" << std::endl;
        return;
    }

    // Initialize Spinnaker
    system_c = Spinnaker::System::GetInstance();
    camList = system_c->GetCameras();
    if (camList.GetSize() == 0)
    {
        std::cerr << "[spinnaker stream] No cameras detected!" << std::endl;
        return;
    }
    camera = camList.GetByIndex(0);
    camera->Init();
    camera->BeginAcquisition();

    unsigned int width;
    unsigned int height;

    while (!signal)
    {
        pImage = camera->GetNextImage();

        if (pImage->IsIncomplete())
        {
            std::cerr << "[spinnaker stream] Image incomplete: " << pImage->GetImageStatus() << std::endl;
            continue;
        }

        // Print the pixel format only once and initialize the virtual device and CUDA
        if (!is_init)
        {
            // Get the image size
            width = pImage->GetWidth();
            height = pImage->GetHeight();
            std::cout << "[spinnaker stream] Image size: " << pImage->GetWidth() << "x" << pImage->GetHeight() <<
                std::endl;

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
                std::cerr << "[spinnaker stream] Failed to set video format on virtual device" << std::endl;
                break;
            }

            // Set the frame rate
            struct v4l2_streamparm streamparm = {};
            streamparm.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
            streamparm.parm.output.timeperframe.numerator = 1;
            streamparm.parm.output.timeperframe.denominator = 60;
            if (ioctl(video_fd, VIDIOC_S_PARM, &streamparm) < 0)
            {
                std::cerr << "[spinnaker stream] Failed to set frame rate" << std::endl;
                break;
            }

            // Initialize CUDA and allocate memory for YUYV422 format
            init_bayer2yuyv_cuda(width, height, CUDA_STREAMS);
            yuyv422 = get_cuda_buffer(width * height * 2);

            std::cout << "[spinnaker stream] Converting to YUYV422 format..." << std::endl;
            is_init = true;
        }

        // Handle BayerRG8 format: Convert BayerRG8 to YUYV
        imageData = static_cast<unsigned char*>(pImage->GetData());
        bayer2yuyv_cuda(imageData, yuyv422);

        // Write the YUYV422 (16 bits per pixel) data to the virtual video device as YUYV422
        if (write(video_fd, yuyv422, width * height * 2) == -1)
        {
            std::cerr << "[spinnaker stream] Error writing frame to virtual device" << std::endl;
            break;
        }

        pImage->Release();
    }

    // Cleanup
    if (is_init)
    {
        imageData = nullptr;
        free_cuda_buffer(yuyv422);
        yuyv422 = nullptr;
        cleanup_bayer2yuyv_cuda();
        is_init = false;
    }
    pImage = nullptr;
    camera->EndAcquisition();
    camera->DeInit();
    camera = nullptr;
    camList.Clear();
    system_c->ReleaseInstance();
    system_c = nullptr;
    close(video_fd);
}
