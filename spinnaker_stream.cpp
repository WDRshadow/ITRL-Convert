#include <Spinnaker.h>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <chrono>

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
            break;
        }

        if (ioctl(video_fd, VIDIOC_S_FMT, &vfmt) < 0)
        {
            std::cerr << "Failed to set video format on virtual device" << std::endl;
            return -1;
        }

        return 0;
    }

    // Function to convert Spinnaker PixelFormat to V4L2 format
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

    // Function to get the pixel format as a string
    const char *pixel_format_to_string(Spinnaker::PixelFormatEnums pixelFormat)
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

    // Function to capture frames from the FLIR camera and stream them in supported formats
    void capture_frames(const char *video_device)
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

        // Start capturing images
        camera->BeginAcquisition();

        // CPU time storage
        std::vector<double> cpu_times_to_rgb24;
        std::vector<double> cpu_times_to_yuyv422;

        // Capture 100 frames for testing
        int count = 0;
        while (count < 1000)
        {
            Spinnaker::ImagePtr pImage = camera->GetNextImage();
            if (pImage->IsIncomplete())
            {
                std::cerr << "Image incomplete: " << pImage->GetImageStatus() << std::endl;
                continue;
            }

            // Get pixel format directly from the ImagePtr object
            Spinnaker::PixelFormatEnums pixelFormat = pImage->GetPixelFormat();

            // Print the pixel format only once
            if (!pixel_format_printed)
            {
                const char *pixelFormatName = pixel_format_to_string(pixelFormat);
                std::cout << "Captured pixel format: " << pixelFormatName << std::endl;
                std::cout << "Image size: " << pImage->GetWidth() << "x" << pImage->GetHeight() << std::endl;
                pixel_format_printed = true;
            }

            unsigned char *imageData = nullptr;
            unsigned int width = pImage->GetWidth();
            unsigned int height = pImage->GetHeight();
            static unsigned char *yuyv422 = new unsigned char[width * height * 2];

            // Handle BayerRG8 format: Convert BayerRG8 to RGB8
            if (pixelFormat == Spinnaker::PixelFormatEnums::PixelFormat_BayerRG8)
            {
                auto start = std::chrono::high_resolution_clock::now(); // Start timer
                Spinnaker::ImagePtr convertedImage = pImage->Convert(Spinnaker::PixelFormatEnums::PixelFormat_RGB8);
                auto end = std::chrono::high_resolution_clock::now(); // End timer
                std::chrono::duration<double> elapsed = end - start;
                cpu_times_to_rgb24.push_back(elapsed.count() * 1000);
                imageData = static_cast<unsigned char *>(convertedImage->GetData());
            }
            else
            {
                imageData = static_cast<unsigned char *>(pImage->GetData());
            }

            // Convert RGB24 to YUYV422
            auto start1 = std::chrono::high_resolution_clock::now(); // Start timer
            convert_rgb24_to_yuyv_cuda(imageData, yuyv422, width, height);
            auto end1 = std::chrono::high_resolution_clock::now(); // End timer
            std::chrono::duration<double> elapsed1 = end1 - start1;
            cpu_times_to_yuyv422.push_back(elapsed1.count() * 1000);

            // Configure the virtual video device for YUYV422
            if (configure_video_device(video_fd, width, height, V4L2_PIX_FMT_YUYV) != 0)
            {
                std::cerr << "Failed to configure virtual device" << std::endl;
                break;
            }

            // Write the YUYV422 (16 bits per pixel) data to the virtual video device as YUYV422
            if (write(video_fd, yuyv422, width * height * 2) == -1)
            {
                std::cerr << "Error writing frame to virtual device" << std::endl;
                break;
            }

            pImage->Release();

            count++;
        }

        // Print CPU mean times
        double cpu_time_to_rgb24_mean = 0.0;
        double cpu_time_to_yuyv422_mean = 0.0;
        for (size_t i = 0; i < cpu_times_to_rgb24.size(); i++)
        {
            cpu_time_to_rgb24_mean += cpu_times_to_rgb24[i];
            cpu_time_to_yuyv422_mean += cpu_times_to_yuyv422[i];
        }
        cpu_time_to_rgb24_mean /= cpu_times_to_rgb24.size();
        cpu_time_to_yuyv422_mean /= cpu_times_to_yuyv422.size();
        std::cout << "CPU time to convert BayerRG8 to RGB8 (mean): " << cpu_time_to_rgb24_mean << " ms" << std::endl;
        std::cout << "CPU time to convert RGB24 to YUYV422 (mean): " << cpu_time_to_yuyv422_mean << " ms" << std::endl;

        cleanup_cuda_buffers();
        camera->EndAcquisition();
        camera->DeInit();
        camList.Clear();
        system->ReleaseInstance();

        close(video_fd);
    }
}
