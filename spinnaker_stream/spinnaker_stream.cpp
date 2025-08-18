#include <Spinnaker.h>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <thread>
#include <chrono>

#include "spinnaker_stream.h"
#include "component.h"
#include "formatting.h"
#include "socket_bridge.h"
#include "sensor.h"
#include "data_logger.h"
#include "gamma.h"
#include "ring_buffer.h"

#define BUFFER_SIZE 2048
#define CUDA_STREAMS 8
#define Y_TARGET 115.0
#define FORWARD 0

#define DATA_NUM 3
const int _data_logger_ids[] = {IncPkgNr, Velocity, AX};
const int _data_logger_type[] = {_TYPE_INT, _TYPE_FLOAT, _TYPE_FLOAT};
#define DATA_NUM_2 5
const int _data_logger_ids_2[] = {PkgNr, RefStrAngle, RefThrottle, RefBrk, Direction};
const int _data_logger_type_2[] = {_TYPE_INT, _TYPE_FLOAT, _TYPE_FLOAT, _TYPE_FLOAT, _TYPE_INT};
#define DATA_NUM_3 1
const int _data_logger_ids_3[] = {Latency};
const int _data_logger_type_3[] = {_TYPE_FLOAT};

bool is_init = false;
int video_fd;
Spinnaker::SystemPtr system_c = nullptr;
Spinnaker::CameraList camList;
Spinnaker::CameraPtr camera = nullptr;
Spinnaker::CameraPtr camera_2 = nullptr;
Spinnaker::ImagePtr pImage = nullptr;
SocketBridge *bridge = nullptr;
SocketBridge *bridge_2 = nullptr;
SocketBridge *bridge_3 = nullptr;
char *buffer = nullptr;
char *buffer_2 = nullptr;
char *buffer_3 = nullptr;
bool is_sensor_init = false;
bool thread_signal = false;
bool is_thread_running = false;
bool is_thread_running_2 = false;
bool is_thread_running_3 = false;
unsigned char *bayer = nullptr;
unsigned char *rgb = nullptr;
unsigned char *yuyv = nullptr;
std::thread sensor_thread;
std::thread sensor_thread_2;
std::thread sensor_thread_3;

void capture_frames(const char *video_device, const std::string &ip, const int port, bool &signal, const int fps, const int delay_ms, const char *logger, const bool is_hmi)
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
    camera->GammaEnable.SetValue(true);
    if (camList.GetSize() > 1)
    {
        std::cerr << "[spinnaker stream] More than one camera detected, adding the reverse camera." << std::endl;
        camera_2 = camList.GetByIndex(1);
        camera_2->Init();
        camera_2->BeginAcquisition();
    }

    // Define the sensor data components
    std::unique_ptr<StreamImage> stream_image;
    std::shared_ptr<PredictionLine> prediction_line;
    std::shared_ptr<TextComponent> velocity;
    std::shared_ptr<TextComponent> latency_label;
    std::unique_ptr<SensorAPI> latency;
    std::unique_ptr<SensorAPI> vel;
    std::unique_ptr<SensorAPI> ax;
    std::unique_ptr<SensorAPI> str_whe_phi;
    std::unique_ptr<SensorAPI> direction;
    std::unique_ptr<DataLogger> data_logger;
    std::unique_ptr<DataLogger> data_logger_2;
    std::shared_mutex bufferMutex;
    std::shared_mutex bufferMutex_2;
    std::shared_mutex bufferMutex_3;
    std::unique_ptr<RingBuffer> image_buffer = nullptr;

    // Initialize Streaming Component
    bool is_sensor_connected = false;
    if (port != -1)
    {
        bridge = new SocketBridge(ip, port);
        if (bridge)
        {
            is_sensor_connected = bridge->isValid();
        }
        bridge_2 = new SocketBridge(ip, port + 1);
        if (bridge_2)
        {
            is_sensor_connected = is_sensor_connected && bridge_2->isValid();
        }
        bridge_3 = new SocketBridge(ip, port + 2);
        if (bridge_3)
        {
            is_sensor_connected = is_sensor_connected && bridge_3->isValid();
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
        if (bridge_2)
        {
            delete bridge_2;
            bridge_2 = nullptr;
        }
        if (bridge_3)
        {
            delete bridge_3;
            bridge_3 = nullptr;
        }
        std::cout << "[spinnaker stream] Sensor data not available." << std::endl;
    }

    // Define the converter pointer
    std::unique_ptr<CudaImageConverter> converter_bayer2rgb;
    std::unique_ptr<CudaImageConverter> converter_rgb2yuyv;
    std::unique_ptr<CudaImageConverter> converter_bayer2rgb_2;
    std::unique_ptr<PIDGammaController> gamma_controller;

    unsigned int width;
    unsigned int height;

    // Calculate additional sleep time for frame rate control
    const int default_fps = 60;
    int additional_sleep_ms = 0;
    if (fps < default_fps)
    {
        additional_sleep_ms = (1000 / fps) - (1000 / default_fps);
    }

    // Vehicle direction
    int vehicle_direction = FORWARD;

    while (!signal)
    {
        pImage = vehicle_direction == FORWARD ? camera->GetNextImage() : camera_2->GetNextImage();

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
            streamparm.parm.output.timeperframe.denominator = fps;
            if (ioctl(video_fd, VIDIOC_S_PARM, &streamparm) < 0)
            {
                std::cerr << "[spinnaker stream] Failed to set frame rate" << std::endl;
                break;
            }

            // Initialize CUDA and allocate memory for
            converter_bayer2rgb = std::make_unique<CudaImageConverter>(width, height, CUDA_STREAMS, BAYER2RGB);
            converter_rgb2yuyv = std::make_unique<CudaImageConverter>(width, height, CUDA_STREAMS, RGB2YUYV);
            rgb = get_cuda_buffer(width * height * 3);
            yuyv = get_cuda_buffer(width * height * 2);

            // Initialize other components
            gamma_controller = std::make_unique<PIDGammaController>(0.0001, 0.0, 0.00001, 0.25, 4.0, 0.01);

            // Initialize ring buffer for variable delay, BayerRG format (1 bytes per pixel)
            if (delay_ms > 0)
            {
                image_buffer = std::make_unique<RingBuffer>(delay_ms, fps, width, height, 1);
                std::cout << "[spinnaker stream] Ring buffer initialized for " << delay_ms << "ms delay" << std::endl;
            }

            std::cout << "[spinnaker stream] Converting to YUYV422 format..." << std::endl;
            is_init = true;
        }

        bayer = static_cast<unsigned char *>(pImage->GetData());

        if (delay_ms > 0)
        {
            image_buffer->update(bayer);
            unsigned char *delayed_bayer = image_buffer->get_oldest();
            converter_bayer2rgb->convert(delayed_bayer, rgb);
        }
        else
        {
            converter_bayer2rgb->convert(bayer, rgb);
        }

        // Add components to the image
        if (is_sensor_connected)
        {
            if (!is_sensor_init)
            {
                buffer = new char[BUFFER_SIZE];
                buffer_2 = new char[BUFFER_SIZE];
                buffer_3 = new char[BUFFER_SIZE];
                if (is_hmi)
                {
                    vel = std::make_unique<SensorAPI>(Velocity, buffer, BUFFER_SIZE, bufferMutex);
                    ax = std::make_unique<SensorAPI>(AX, buffer, BUFFER_SIZE, bufferMutex);
                    str_whe_phi = std::make_unique<SensorAPI>(RefStrAngle, buffer_2, BUFFER_SIZE, bufferMutex_2);
                    latency = std::make_unique<SensorAPI>(Latency, buffer_3, BUFFER_SIZE, bufferMutex_3);
                    stream_image = std::make_unique<StreamImage>(width, height);
                    prediction_line = std::make_shared<PredictionLine>("fisheye_calibration.yaml",
                                                                       "homography_calibration.yaml", width, height);
                    velocity = make_shared<TextComponent>(1536, 1462, 200, 200);
                    latency_label = make_shared<TextComponent>(2800, 100, 500, 200);
                    stream_image->add_component("prediction_line", std::static_pointer_cast<Component>(prediction_line));
                    stream_image->add_component("velocity", std::static_pointer_cast<Component>(velocity));
                    stream_image->add_component("latency_label", std::static_pointer_cast<Component>(latency_label));
                }
                direction = std::make_unique<SensorAPI>(Direction, buffer_2, BUFFER_SIZE, bufferMutex_2);
                if (logger)
                {
                    data_logger = std::make_unique<DataLogger>(_data_logger_ids, _data_logger_type, DATA_NUM, buffer, BUFFER_SIZE, bufferMutex, logger);
                    std::string logger_2 = std::string(logger);
                    size_t pos = logger_2.find(".csv");
                    if (pos != std::string::npos)
                    {
                        logger_2.insert(pos, "_2");
                    }
                    data_logger_2 = std::make_unique<DataLogger>(_data_logger_ids_2, _data_logger_type_2, DATA_NUM_2, buffer_2, BUFFER_SIZE, bufferMutex_2, logger_2.c_str());
                }
                sensor_thread = std::thread(receive_data_loop, bridge, buffer, BUFFER_SIZE, std::ref(bufferMutex),
                                            std::ref(thread_signal), std::ref(is_thread_running));
                sensor_thread_2 = std::thread(receive_data_loop, bridge_2, buffer_2, BUFFER_SIZE, std::ref(bufferMutex_2),
                                              std::ref(thread_signal), std::ref(is_thread_running_2));
                sensor_thread_3 = std::thread(receive_data_loop, bridge_3, buffer_3, BUFFER_SIZE, std::ref(bufferMutex_3),
                                              std::ref(thread_signal), std::ref(is_thread_running_3));
                is_sensor_init = true;
            }
            if (vehicle_direction == FORWARD && is_hmi)
            {
                const int total_delay = delay_ms + latency->get_int_value();
                prediction_line->update(vel->get_float_value() * 3.6f, ax->get_float_value(), str_whe_phi->get_float_value(), str_whe_phi->get_float_value(), total_delay / 1000);
                velocity->update(to_string(static_cast<int>(vel->get_float_value() * 3.6f)));
                latency_label->update(std::to_string(total_delay) + " ms");
                *stream_image >> rgb;
            }
            if (logger)
            {
                data_logger->logger();
                data_logger_2->logger();
            }
            vehicle_direction = direction->get_int_value();
            vehicle_direction = (vehicle_direction == 1) ? FORWARD : vehicle_direction;
        }
        converter_rgb2yuyv->convert(rgb, yuyv);

        // Adjust the gamma value based on the mean Y value in the center ROI
        if (vehicle_direction == FORWARD && gamma_controller)
        {
            double meanY = computeROImeanY(yuyv, height, width, height / 4, width / 4);
            if (meanY > 0.0)
            {
                const double gamma_current = camera->Gamma.GetValue();
                double gamma = gamma_controller->update(meanY, Y_TARGET, gamma_current);
                camera->Gamma.SetValue(gamma);
                // std::cout << "[spinnaker stream] Mean Y value: " << meanY << ", Gamma set to: " << gamma << std::endl;
            }
            else
            {
                std::cerr << "[spinnaker stream] Error computing mean Y value" << std::endl;
            }
        }

        // Write the YUYV422 (16 bits per pixel) data to the virtual video device
        if (write(video_fd, yuyv, width * height * 2) == -1)
        {
            std::cerr << "[spinnaker stream] Error writing frame to virtual device" << std::endl;
            break;
        }

        // Sleep additional time to achieve target fps (if lower than default 60fps)
        if (additional_sleep_ms > 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(additional_sleep_ms));
        }

        if (delay_ms > 0)
        {
            image_buffer->join();
        }

        pImage->Release();
    }

    // Cleanup
    if (is_sensor_init)
    {
        thread_signal = true;
        if (sensor_thread.joinable() && sensor_thread_2.joinable() && sensor_thread_3.joinable())
        {
            int count = 0;
            while ((is_thread_running || is_thread_running_2 || is_thread_running_3) && count++ < 30)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if (is_thread_running || is_thread_running_2 || is_thread_running_3)
            {
                std::cerr << "[spinnaker stream] Sensor thread did not exit gracefully" << std::endl;
                sensor_thread.detach();
                sensor_thread_2.detach();
                sensor_thread_3.detach();
            }
            else
            {
                sensor_thread.join();
                sensor_thread_2.join();
                sensor_thread_3.join();
            }
        }
        thread_signal = false;
        delete[] buffer;
        buffer = nullptr;
        delete[] buffer_2;
        buffer_2 = nullptr;
        delete[] buffer_3;
        buffer_3 = nullptr;
        delete bridge;
        bridge = nullptr;
        delete bridge_2;
        bridge_2 = nullptr;
        delete bridge_3;
        bridge_3 = nullptr;
        is_sensor_init = false;
    }
    if (is_init)
    {
        bayer = nullptr;
        free_cuda_buffer(rgb);
        rgb = nullptr;
        free_cuda_buffer(yuyv);
        yuyv = nullptr;
        if (delay_ms > 0)
        {
            image_buffer.reset();
        }
        is_init = false;
    }
    pImage = nullptr;
    camera->EndAcquisition();
    camera->DeInit();
    camera = nullptr;
    if (camera_2)
    {
        camera_2->EndAcquisition();
        camera_2->DeInit();
        camera_2 = nullptr;
    }
    camList.Clear();
    system_c->ReleaseInstance();
    system_c = nullptr;
    close(video_fd);
}
