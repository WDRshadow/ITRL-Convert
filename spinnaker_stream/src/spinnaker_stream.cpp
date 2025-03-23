#include <Spinnaker.h>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <thread>
#include <csignal>

#include "spinnaker_stream.h"
#include "component.h"
#include "rgb2yuyv.h"
#include "socket_bridge.h"
#include "sensor.h"

#define CUDA_STREAMS 8

bool is_init = false;
int video_fd;
Spinnaker::SystemPtr system_c = nullptr;
Spinnaker::CameraPtr camera = nullptr;
Spinnaker::CameraList camList;
Spinnaker::ImagePtr pImage = nullptr;
SocketBridge *bridge = nullptr;
char *buffer = nullptr;
bool is_sensor_init = false;
bool is_thread_running = false;
unsigned char *imageData = nullptr;
unsigned char *yuyv422 = nullptr;
std::thread sensor_thread;

void cleanup_stream()
{
    if (is_sensor_init)
    {
        is_thread_running = false;
        delete[] buffer;
        buffer = nullptr;
        delete bridge;
        bridge = nullptr;
    }
    if (is_init)
    {
        imageData = nullptr;
        free_cuda_buffer(yuyv422);
        cleanup_cuda_buffers(CUDA_STREAMS);
    }
    if (pImage.IsValid())
    {
        pImage->Release();
    }
    camera->EndAcquisition();
    camera->DeInit();
    camList.Clear();
    system_c->ReleaseInstance();
    close(video_fd);
}

void signalHandler(int signum)
{
    std::cout << "[spinnaker stream] Received signal " << signum << ", cleaning up resources..." << std::endl;
    cleanup_stream();
    exit(signum);
}

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

void capture_frames(const char *video_device, const std::string &ip, const int port)
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

    // Register signal handlers for cleanup
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    // Define the smart pointer objects
    std::unique_ptr<StreamImage> stream_image;
    std::shared_ptr<DriverLine> driver_line;
    std::shared_ptr<PredictionLine> prediction_line;
    std::shared_ptr<TextComponent> velocity;
    std::shared_mutex bufferMutex;
    std::unique_ptr<SensorAPI> str_whe_phi;
    std::unique_ptr<SensorAPI> vel;
    std::unique_ptr<SensorAPI> ax;

    // Initialize Streaming Component
    bool is_sensor_connected = false;
    if (port != -1)
    {
        bridge = new SocketBridge(ip, port);
        is_sensor_connected = bridge->isValid();
    }
    if (is_sensor_connected)
    {
        std::cout << "[spinnaker stream] Listening to sensor data..." << std::endl;
    }
    else
    {
        if (bridge)
        {
            delete bridge;
            bridge = nullptr;
        }
        std::cout << "[spinnaker stream] Sensor data not available." << std::endl;
    }

    while (true)
    {
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
        imageData = static_cast<unsigned char *>(pImage->Convert(Spinnaker::PixelFormatEnums::PixelFormat_RGB8)->GetData());

        // Add components to the image
        if (is_sensor_connected)
        {
            if (!is_sensor_init)
            {
                constexpr int buffer_size = 8192;
                buffer = new char[buffer_size];
                stream_image = std::make_unique<StreamImage>(width, height);
                driver_line = std::make_shared<DriverLine>("fisheye_calibration.yaml", "homography_calibration.yaml", width, height);
                prediction_line = std::make_shared<PredictionLine>("fisheye_calibration.yaml", "homography_calibration.yaml", width, height);
                velocity = make_shared<TextComponent>(1536, 1462, 200, 200);
                str_whe_phi = std::make_unique<SensorAPI>(RemoteSteeringAngle, buffer, buffer_size, bufferMutex);
                vel = std::make_unique<SensorAPI>(Velocity, buffer, buffer_size, bufferMutex);
                ax = std::make_unique<SensorAPI>(AX, buffer, buffer_size, bufferMutex);
                is_thread_running = true;
                sensor_thread = std::thread(receive_data_loop, bridge, buffer, buffer_size, std::ref(bufferMutex), std::ref(is_thread_running));
                stream_image->add_component("velocity", velocity);
                is_sensor_init = true;
            }
            driver_line->update(str_whe_phi->get_value());
            *driver_line >> imageData;
            prediction_line->update(vel->get_value(), ax->get_value(), str_whe_phi->get_value(), 0.1);
            *prediction_line >> imageData;
            velocity->update(to_string(static_cast<int>(vel->get_value())));
            stream_image->update(imageData);
        }

        // Convert RGB24 to YUYV422
        rgb2yuyv_cuda(imageData, yuyv422, width, height, CUDA_STREAMS);

        // Write the YUYV422 (16 bits per pixel) data to the virtual video device as YUYV422
        if (write(video_fd, yuyv422, width * height * 2) == -1)
        {
            std::cerr << "[spinnaker stream] Error writing frame to virtual device" << std::endl;
            break;
        }

        pImage->Release();
    }

    cleanup_stream();
}
