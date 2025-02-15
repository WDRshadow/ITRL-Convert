#include <Spinnaker.h>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>

extern "C" {

    // Function to configure the virtual V4L2 device for RGB24 format
    int configure_video_device(int video_fd, int width, int height, __u32 pixel_format) {
        struct v4l2_format vfmt = {};
        vfmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
        vfmt.fmt.pix.width = width;
        vfmt.fmt.pix.height = height;
        vfmt.fmt.pix.pixelformat = pixel_format;  // Set pixel format to RGB24
        vfmt.fmt.pix.field = V4L2_FIELD_NONE;
        vfmt.fmt.pix.bytesperline = width * 3;  // RGB24 = 3 bytes per pixel
        vfmt.fmt.pix.sizeimage = width * height * 3;  // 3 bytes per pixel (RGB24)

        if (ioctl(video_fd, VIDIOC_S_FMT, &vfmt) < 0) {
            std::cerr << "Failed to set video format on virtual device" << std::endl;
            return -1;
        }

        return 0;
    }

    // Function to convert Spinnaker PixelFormat to V4L2 format
    __u32 spinnaker_to_v4l2_format(Spinnaker::PixelFormatEnums pixelFormat) {
        switch (pixelFormat) {
            case Spinnaker::PixelFormatEnums::PixelFormat_RGB8:
                return V4L2_PIX_FMT_RGB24;
            case Spinnaker::PixelFormatEnums::PixelFormat_BGR8:
                return V4L2_PIX_FMT_BGR24;
            case Spinnaker::PixelFormatEnums::PixelFormat_Mono8:
                return V4L2_PIX_FMT_GREY;
            case Spinnaker::PixelFormatEnums::PixelFormat_YUV422Packed:
                return V4L2_PIX_FMT_YUYV;  // YUV422 packed (YUYV)
            default:
                return 0;  // Unsupported format
        }
    }

    // Function to get the pixel format as a string
    const char* pixel_format_to_string(Spinnaker::PixelFormatEnums pixelFormat) {
        switch (pixelFormat) {
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
    void capture_frames(const char* video_device) {
        // Open the virtual V4L2 device
        int video_fd = open(video_device, O_WRONLY);
        if (video_fd < 0) {
            std::cerr << "Failed to open virtual video device" << std::endl;
            return;
        }

        // Initialize Spinnaker
        Spinnaker::SystemPtr system = Spinnaker::System::GetInstance();
        Spinnaker::CameraList camList = system->GetCameras();

        if (camList.GetSize() == 0) {
            std::cerr << "No cameras detected!" << std::endl;
            return;
        }

        Spinnaker::CameraPtr camera = camList.GetByIndex(0);
        camera->Init();

        bool pixel_format_printed = false;

        // Start capturing images
        camera->BeginAcquisition();

        while (true) {
            Spinnaker::ImagePtr pImage = camera->GetNextImage();
            if (pImage->IsIncomplete()) {
                std::cerr << "Image incomplete: " << pImage->GetImageStatus() << std::endl;
                continue;
            }

            // Get pixel format directly from the ImagePtr object
            Spinnaker::PixelFormatEnums pixelFormat = pImage->GetPixelFormat();

            // Print the pixel format only once
            if (!pixel_format_printed) {
                const char* pixelFormatName = pixel_format_to_string(pixelFormat);
                std::cout << "Captured pixel format: " << pixelFormatName << std::endl;
                std::cout << "Raw pixel format value: " << pixelFormat << std::endl;  // Print raw value
                pixel_format_printed = true;
            }

            unsigned char* imageData = nullptr;
            unsigned int width = pImage->GetWidth();
            unsigned int height = pImage->GetHeight();

            // Handle BayerRG8 format: Convert BayerRG8 to RGB8
            if (pixelFormat == Spinnaker::PixelFormatEnums::PixelFormat_BayerRG8) {
                Spinnaker::ImagePtr convertedImage = pImage->Convert(Spinnaker::PixelFormatEnums::PixelFormat_RGB8);
                imageData = static_cast<unsigned char*>(convertedImage->GetData());
                std::cout << "Converted BayerRG8 to RGB8" << std::endl;
            } else {
                imageData = static_cast<unsigned char*>(pImage->GetData());
            }

            // Configure the virtual video device for RGB24
            if (configure_video_device(video_fd, width, height, V4L2_PIX_FMT_RGB24) != 0) {
                std::cerr << "Failed to configure virtual device" << std::endl;
                break;
            }

            // Write the RGB8 (24 bits per pixel) data to the virtual video device as RGB24
            if (write(video_fd, imageData, width * height * 3) == -1) {
                std::cerr << "Error writing frame to virtual device" << std::endl;
                break;
            }

            pImage->Release();
        }

        camera->EndAcquisition();
        camera->DeInit();
        camList.Clear();
        system->ReleaseInstance();

        close(video_fd);
    }
}

