#include <Spinnaker.h>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>

#include "spinnaker_stream.h"
#include "component.h"
#include "rgb2yuyv.h"
#include "util.h"
#include "sensor.h"

extern "C" {
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

__u32 spinnaker_to_v4l2_format(Spinnaker::PixelFormatEnums pixelFormat)
{
    switch (pixelFormat)
    {
    case Spinnaker::PixelFormatEnums::PixelFormat_RGB8:
        return V4L2_PIX_FMT_RGB24;
    case Spinnaker::PixelFormatEnums::PixelFormat_BGR8:
        return V4L2_PIX_FMT_BGR24;
    case Spinnaker::PixelFormatEnums::PixelFormat_Mono8:
        return V4L2_PIX_FMT_GREY;
    case Spinnaker::PixelFormatEnums::PixelFormat_YUV422Packed:
        return V4L2_PIX_FMT_YUYV; // YUV422 packed (YUYV)
    default:
        return 0; // Unsupported format
    }
}

const char* pixel_format_to_string(Spinnaker::PixelFormatEnums pixelFormat)
{
    switch (pixelFormat)
    {
    case Spinnaker::PixelFormatEnums::PixelFormat_Mono8:
        return "Mono8";
    case Spinnaker::PixelFormatEnums::PixelFormat_Mono16:
        return "Mono16";
    case Spinnaker::PixelFormatEnums::PixelFormat_RGB8:
        return "RGB8";
    case Spinnaker::PixelFormatEnums::PixelFormat_BGR8:
        return "BGR8";
    case Spinnaker::PixelFormatEnums::PixelFormat_YUV422Packed:
        return "YUV422Packed";
    case Spinnaker::PixelFormatEnums::PixelFormat_BayerRG8:
        return "BayerRG8";
    default:
        return "Unknown format";
    }
}

void capture_frames(const char* video_device)
{
    // Open the virtual V4L2 device
    int video_fd = open(video_device, O_WRONLY);
    if (video_fd < 0)
    {
        std::cerr << "Failed to open virtual video device" << std::endl;
        return;
    }

    // Initialize Spinnaker
    Spinnaker::SystemPtr system = Spinnaker::System::GetInstance();
    Spinnaker::CameraList camList = system->GetCameras();

    if (camList.GetSize() == 0)
    {
        std::cerr << "No cameras detected!" << std::endl;
        return;
    }

    Spinnaker::CameraPtr camera = camList.GetByIndex(0);
    camera->Init();

    bool pixel_format_printed = false;
    bool is_configured = false;

    // Start capturing images
    camera->BeginAcquisition();

    // Initialize Streaming Component
    StreamImage stream_image(3072, 2048);
    const auto driver_line = make_shared<DriverLine>("fisheye_calibration.yaml", "homography_calibration.yaml");
    const auto velocity = make_shared<Velocity>(1536, 1462);
    stream_image.add_component("driver_line", driver_line);
    stream_image.add_component("velocity", velocity);
    const SensorBuffer str_whe_phi("test/str_whe_phi.csv");
    const SensorBuffer vel("test/vel.csv");
    const auto interpolation_60_100 = [](int x) -> int {
        return x * 3 / 5;
    };

    while (true)
    {
        static Spinnaker::ImagePtr pImage = nullptr;

        pImage = camera->GetNextImage();

        static unsigned int width = pImage->GetWidth();
        static unsigned int height = pImage->GetHeight();

        if (pImage->IsIncomplete())
        {
            std::cerr << "Image incomplete: " << pImage->GetImageStatus() << std::endl;
            continue;
        }

        // Print the pixel format only once
        if (!pixel_format_printed)
        {
            std::cout << "Captured pixel format: " << pixel_format_to_string(pImage->GetPixelFormat()) << std::endl;
            std::cout << "Image size: " << pImage->GetWidth() << "x" << pImage->GetHeight() << std::endl;
            std::cout << "Converting to YUYV422 format..." << std::endl;
            pixel_format_printed = true;
        }

        // Handle BayerRG8 format: Convert BayerRG8 to RGB8
        static unsigned char* imageData = nullptr;
        imageData = static_cast<unsigned char*>(pImage->Convert(Spinnaker::PixelFormatEnums::PixelFormat_RGB8)->
                                                        GetData());

        // Add other components to the image
        // -----------------------------------------------------------------------------------------------
        static constexpr int start = 15293;
        static int count = 0;
        static int index = interpolation_60_100(count++);
        static auto *_img_ = new unsigned char[width * height * 3];
        driver_line->update({{"str_whe_phi", to_string(str_whe_phi.get_value(start + index))}});
        velocity->update({{"vel", to_string(vel.get_value(start + index))}});
        stream_image.update(imageData, _img_);
        // -----------------------------------------------------------------------------------------------

        // Convert RGB24 to YUYV422
        static auto* yuyv422 = new unsigned char[width * height * 2];
        convert_rgb24_to_yuyv_cuda(_img_, yuyv422, width, height);

        // Configure the virtual video device for YUYV422
        if (!is_configured)
        {
            if (configure_video_device(video_fd, width, height, V4L2_PIX_FMT_YUYV) != 0)
            {
                std::cerr << "Failed to configure virtual device" << std::endl;
                break;
            }
            is_configured = true;
        }

        // Write the YUYV422 (16 bits per pixel) data to the virtual video device as YUYV422
        if (write(video_fd, yuyv422, width * height * 2) == -1)
        {
            std::cerr << "Error writing frame to virtual device" << std::endl;
            break;
        }

        pImage->Release();
    }

    cleanup_cuda_buffers();
    camera->EndAcquisition();
    camera->DeInit();
    camList.Clear();
    system->ReleaseInstance();

    close(video_fd);
}
}
