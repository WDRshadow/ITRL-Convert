#include <Spinnaker.h>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <thread>

#include "spinnaker_stream.h"
#include "component.h"
#include "formatting.h"
#include "socket_bridge.h"
#include "sensor.h"
#include "gamma.h"

#define BUFFER_SIZE 8192
#define CUDA_STREAMS 8
#define Y_TARGET 128.0

bool is_init = false;
int video_fd;
Spinnaker::SystemPtr system_c = nullptr;
Spinnaker::CameraPtr camera = nullptr;
Spinnaker::CameraList camList;
Spinnaker::ImagePtr pImage = nullptr;
SocketBridge *bridge = nullptr;
char *buffer = nullptr;
bool is_sensor_init = false;
bool thread_signal = false;
bool is_thread_running = false;
unsigned char *bayer = nullptr;
unsigned char *rgb = nullptr;
unsigned char *yuyv = nullptr;
std::thread sensor_thread;

void capture_frames(const char *video_device, const std::string &ip, const int port, bool &signal)
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

    // Define the sensor data components
    std::unique_ptr<StreamImage> stream_image;
    std::shared_ptr<PredictionLine> prediction_line;
    std::shared_ptr<TextComponent> velocity;
    std::shared_ptr<TextComponent> latency_label;
    std::unique_ptr<SensorAPI> str_whe_phi;
    std::unique_ptr<SensorAPI> vel;
    std::unique_ptr<SensorAPI> ax;
    std::shared_mutex bufferMutex;

    // Initialize Streaming Component
    bool is_sensor_connected = false;
    if (port != -1)
    {
        bridge = new SocketBridge(ip, port);
        if (bridge)
        {
            is_sensor_connected = bridge->isValid();
        }
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

    // Define the converter pointer
    std::unique_ptr<CudaImageConverter> converter_bayer2rgb;
    std::unique_ptr<CudaImageConverter> converter_rgb2yuyv;
    std::unique_ptr<CudaImageConverter> converter_bayer2yuyv;
    std::unique_ptr<PIDGammaController> gamma_controller;

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
            std::cout << "[spinnaker stream] Image size: " << pImage->GetWidth() << "x" << pImage->GetHeight() << std::endl;

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

            // Initialize CUDA and allocate memory for
            if (is_sensor_connected)
            {
                converter_bayer2rgb = std::make_unique<CudaImageConverter>(width, height, CUDA_STREAMS, BAYER2RGB);
                converter_rgb2yuyv = std::make_unique<CudaImageConverter>(width, height, CUDA_STREAMS, RGB2YUYV);
                rgb = get_cuda_buffer(width * height * 3);
            }
            else
            {
                converter_bayer2yuyv = std::make_unique<CudaImageConverter>(width, height, CUDA_STREAMS, BAYER2YUYV);
            }
            yuyv = get_cuda_buffer(width * height * 2);

            // Initialize Gamma controller
            gamma_controller = std::make_unique<PIDGammaController>(0.1, 0.01, 0.01);

            std::cout << "[spinnaker stream] Converting to YUYV422 format..." << std::endl;
            is_init = true;
        }

        bayer = static_cast<unsigned char *>(pImage->GetData());

        // Add components to the image
        if (is_sensor_connected)
        {
            if (!is_sensor_init)
            {
                buffer = new char[BUFFER_SIZE];
                stream_image = std::make_unique<StreamImage>(width, height);
                prediction_line = std::make_shared<PredictionLine>("fisheye_calibration.yaml",
                                                                   "homography_calibration.yaml", width, height);
                velocity = make_shared<TextComponent>(1536, 1462, 200, 200);
                latency_label = make_shared<TextComponent>(2800, 100, 500, 200);
                str_whe_phi = std::make_unique<SensorAPI>(RemoteSteeringAngle, buffer, BUFFER_SIZE, bufferMutex);
                vel = std::make_unique<SensorAPI>(Velocity, buffer, BUFFER_SIZE, bufferMutex);
                ax = std::make_unique<SensorAPI>(AX, buffer, BUFFER_SIZE, bufferMutex);
                sensor_thread = std::thread(receive_data_loop, bridge, buffer, BUFFER_SIZE, std::ref(bufferMutex),
                                            std::ref(thread_signal), std::ref(is_thread_running));
                stream_image->add_component("prediction_line", std::static_pointer_cast<Component>(prediction_line));
                stream_image->add_component("velocity", std::static_pointer_cast<Component>(velocity));
                stream_image->add_component("latency_label", std::static_pointer_cast<Component>(latency_label));
                latency_label->update("Latency: 0 ms");
                is_sensor_init = true;
            }
            converter_bayer2rgb->convert(bayer, rgb);
            prediction_line->update(vel->get_value() * 3.6f, ax->get_value(), str_whe_phi->get_value(), str_whe_phi->get_value(), 0.0);
            velocity->update(to_string(static_cast<int>(vel->get_value() * 3.6f)));
            *stream_image >> rgb;
            converter_rgb2yuyv->convert(rgb, yuyv);
        }
        else
        {
            converter_bayer2yuyv->convert(bayer, yuyv);
        }

        // Adjust the gamma value based on the mean Y value in the center ROI
        double meanY = computeROImeanY(yuyv, height, width, height / 4, width / 4);
        if (meanY >= 0.0)
        {
            const double gamma_current = camera->Gamma.GetValue();
            double gamma = gamma_controller->update(meanY, Y_TARGET, gamma_current);
            camera->Gamma.SetValue(gamma);
        }
        else
        {
            std::cerr << "[spinnaker stream] Error computing mean Y value" << std::endl;
        }

        // Write the YUYV422 (16 bits per pixel) data to the virtual video device as YUYV422
        if (write(video_fd, yuyv, width * height * 2) == -1)
        {
            std::cerr << "[spinnaker stream] Error writing frame to virtual device" << std::endl;
            break;
        }

        pImage->Release();
    }

    // Cleanup
    if (is_sensor_init)
    {
        thread_signal = true;
        if (sensor_thread.joinable())
        {
            int count = 0;
            while (is_thread_running && count++ < 30)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if (is_thread_running)
            {
                std::cerr << "[spinnaker stream] Sensor thread did not exit gracefully" << std::endl;
                sensor_thread.detach();
            }
            else
            {
                sensor_thread.join();
            }
        }
        thread_signal = false;
        delete[] buffer;
        buffer = nullptr;
        delete bridge;
        bridge = nullptr;
        is_sensor_init = false;
    }
    if (is_init)
    {
        bayer = nullptr;
        free_cuda_buffer(rgb);
        rgb = nullptr;
        free_cuda_buffer(yuyv);
        yuyv = nullptr;
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
