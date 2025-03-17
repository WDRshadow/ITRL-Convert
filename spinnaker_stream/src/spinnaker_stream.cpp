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
        std::cerr << "[spinnaker stream] Failed to set video format on virtual device" << std::endl;
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
        std::cerr << "[spinnaker stream] Failed to open virtual video device" << std::endl;
        return;
    }

    // Initialize Spinnaker
    Spinnaker::SystemPtr system = Spinnaker::System::GetInstance();
    Spinnaker::CameraList camList = system->GetCameras();
    if (camList.GetSize() == 0)
    {
        std::cerr << "[spinnaker stream] No cameras detected!" << std::endl;
        return;
    }
    Spinnaker::CameraPtr camera = camList.GetByIndex(0);
    camera->Init();
    camera->BeginAcquisition();

    // Initialize Streaming Component
    bool is_sensor_connected = false;
    SocketBridge* bridge = nullptr;
    if (port != -1)
    {
        bridge = new SocketBridge(ip, port);
        is_sensor_connected = bridge->isValid();
    }
    if (is_sensor_connected) {
        std::cout << "[spinnaker stream] Listening to sensor data..." << std::endl;
    } else {
        if (bridge != nullptr) {
            delete bridge;
            bridge = nullptr;
        }
        std::cout << "[spinnaker stream] Sensor data not available." << std::endl;
    }


    bool is_init = false;
    unsigned char* imageData = nullptr;
    unsigned char* yuyv422 = nullptr;
    char* buffer = nullptr;

    while (true)
    {
        static Spinnaker::ImagePtr pImage = nullptr;

        pImage = camera->GetNextImage();

        static unsigned int width = pImage->GetWidth();
        static unsigned int height = pImage->GetHeight();

        if (pImage->IsIncomplete())
        {
            std::cerr << "[spinnaker stream] Image incomplete: " << pImage->GetImageStatus() << std::endl;
            continue;
        }

        // Print the pixel format only once and initialize the virtual device
        if (!is_init)
        {
            if (configure_video_device(video_fd, width, height) != 0)
            {
                std::cerr << "[spinnaker stream] Failed to configure virtual device" << std::endl;
                break;
            }
            std::cout << "[spinnaker stream] Image size: " << pImage->GetWidth() << "x" << pImage->GetHeight() << std::endl;
            std::cout << "[spinnaker stream] Converting to YUYV422 format..." << std::endl;
            yuyv422 = get_cuda_buffer(width * height * 2);
            is_init = true;
        }

        // Handle BayerRG8 format: Convert BayerRG8 to RGB8
        imageData = static_cast<unsigned char*>(pImage->Convert(Spinnaker::PixelFormatEnums::PixelFormat_RGB8)->
                                                        GetData());

        // Add components to the image
        if (is_sensor_connected)
        {
            static StreamImage stream_image(width, height);
            static DriverLine driver_line("fisheye_calibration.yaml", "homography_calibration.yaml", width, height);
            static auto velocity = make_shared<TextComponent>(1536, 1462, 200, 200);
            static std::shared_mutex bufferMutex;
            static bool is_first = true;
            constexpr int buffer_size = 8192;
            if (is_first) {
                buffer = new char[buffer_size];
                std::thread t(receive_data_loop, bridge, buffer, buffer_size, std::ref(bufferMutex));
                t.detach();
                stream_image.add_component("velocity", velocity);
                is_first = false;
            }
            static SensorAPI str_whe_phi(RemoteSteeringAngle, buffer, buffer_size, bufferMutex);
            static SensorAPI vel(Velocity, buffer, buffer_size, bufferMutex);
            driver_line.update(str_whe_phi.get_value());
            driver_line >> imageData;
            velocity->update(to_string(static_cast<int>(vel.get_value())));
            stream_image.update(imageData);
        }

        // Convert RGB24 to YUYV422
        convert_rgb24_to_yuyv_cuda(imageData, yuyv422, width, height);

        // Write the YUYV422 (16 bits per pixel) data to the virtual video device as YUYV422
        if (write(video_fd, yuyv422, width * height * 2) == -1)
        {
            std::cerr << "[spinnaker stream] Error writing frame to virtual device" << std::endl;
            break;
        }

        pImage->Release();
    }

    cleanup_cuda_buffers();
    if (yuyv422 != nullptr)
    {
        free_cuda_buffer(yuyv422);
    }
    if (buffer != nullptr)
    {
        delete[] buffer;
    }
    if (bridge != nullptr)
    {
        delete bridge;
    }
    camera->EndAcquisition();
    camera->DeInit();
    camList.Clear();
    system->ReleaseInstance();

    close(video_fd);
}
