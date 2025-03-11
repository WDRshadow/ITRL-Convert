#include <Spinnaker.h>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <thread>

#include "spinnaker_stream.h"
#include "component.h"
#include "rgb2yuyv.h"
#include "socket_bridge.h"
#include "sensor.h"

extern "C" {
int configure_video_device(int video_fd, int width, int height)
{
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
        std::cerr << "Failed to set video format on virtual device" << std::endl;
        return -1;
    }

    return 0;
}

void capture_frames(const char* video_device, const std::string& ip, const int port)
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
    const auto velocity = make_shared<TextComponent>(1536, 1462, 200, 200);
    stream_image.add_component("driver_line", driver_line);
    stream_image.add_component("velocity", velocity);
    std::shared_mutex bufferMutex;
    constexpr int buffer_size = 8192;
    auto buffer = new char[buffer_size];
    std::thread t(receive_data_loop, ip, port, std::ref(buffer), 8192, std::ref(bufferMutex));
    t.detach();
    const SensorAPI str_whe_phi(RemoteSteeringAngle, buffer, buffer_size, bufferMutex);
    const SensorAPI vel(Velocity, buffer, buffer_size, bufferMutex);

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
            std::cout << "Captured pixel format: BayerRG8" << std::endl;
            std::cout << "Image size: " << pImage->GetWidth() << "x" << pImage->GetHeight() << std::endl;
            std::cout << "Converting to YUYV422 format..." << std::endl;
            pixel_format_printed = true;
        }

        // Handle BayerRG8 format: Convert BayerRG8 to RGB8
        static unsigned char* imageData = nullptr;
        imageData = static_cast<unsigned char*>(pImage->Convert(Spinnaker::PixelFormatEnums::PixelFormat_RGB8)->
                                                        GetData());

        // Add components to the image
        static auto* _img_ = new unsigned char[width * height * 3];
        driver_line->update(str_whe_phi.get_value());
        velocity->update(to_string(static_cast<int>(vel.get_value())));
        stream_image.update(imageData, _img_);

        // Convert RGB24 to YUYV422
        static auto* yuyv422 = new unsigned char[width * height * 2];
        convert_rgb24_to_yuyv_cuda(_img_, yuyv422, width, height);

        // Configure the virtual video device for YUYV422
        if (!is_configured)
        {
            if (configure_video_device(video_fd, width, height) != 0)
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
