#include <Spinnaker.h>
#include <cuda.h>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>

#include "cuda_stream.h"
#include "bayer2nv12.h"

#define CUDA_STREAMS 8

bool is_init = false;
int video_fd;
Spinnaker::SystemPtr system_c = nullptr;
Spinnaker::CameraPtr camera = nullptr;
Spinnaker::CameraList camList;
Spinnaker::ImagePtr pImage = nullptr;
unsigned char *bayer = nullptr;
unsigned char *nv12 = nullptr;

int stream_num_ = CUDA_STREAMS;
cudaStream_t *streams = nullptr;
unsigned int block_height;
size_t size_image;
size_t size_bayer_block;
size_t size_nv12_y_block;
size_t size_nv12_uv_block;
unsigned char *d_bayer = nullptr;
unsigned char *d_nv12 = nullptr;

const dim3 blockSize(32, 16);
dim3 gridSize;

int main()
{
    // Open the virtual V4L2 device
    const char *video_device = "/dev/video16";
    video_fd = open(video_device, O_WRONLY);
    if (video_fd < 0)
    {
        std::cerr << "[cuda encoder] Failed to open virtual video device" << std::endl;
        return;
    }

    // Initialize Spinnaker
    system_c = Spinnaker::System::GetInstance();
    camList = system_c->GetCameras();
    if (camList.GetSize() == 0)
    {
        std::cerr << "[cuda encoder] No cameras detected!" << std::endl;
        return -1;
    }
    camera = camList.GetByIndex(0);
    camera->Init();
    camera->BeginAcquisition();

    unsigned int width;
    unsigned int height;

    int count = 0;

    while (true)
    {
        pImage = camera->GetNextImage();

        if (pImage->IsIncomplete())
        {
            std::cerr << "[cuda encoder] Image incomplete: " << pImage->GetImageStatus() << std::endl;
            continue;
        }

        // Print the pixel format only once and initialize CUDA
        if (!is_init)
        {
            // Image size
            width = pImage->GetWidth();
            height = pImage->GetHeight();
            std::cout << "[cuda encoder] Image size: " << pImage->GetWidth() << "x" << pImage->GetHeight() << std::endl;

            // Set the video device format
            struct v4l2_format vfmt = {};
            vfmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
            vfmt.fmt.pix.pixelformat = V4L2_PIX_FMT_NV12;
            vfmt.fmt.pix.field = V4L2_FIELD_NONE;
            vfmt.fmt.pix.width = width;
            vfmt.fmt.pix.height = height;
            vfmt.fmt.pix.bytesperline = width;
            vfmt.fmt.pix.sizeimage = width * height * 3 / 2;
            if (ioctl(video_fd, VIDIOC_S_FMT, &vfmt) < 0)
            {
                std::cerr << "[cuda encoder] Failed to set video format on virtual device" << std::endl;
                break;
            }

            // Set the frame rate
            struct v4l2_streamparm streamparm = {};
            streamparm.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
            streamparm.parm.output.timeperframe.numerator = 1;
            streamparm.parm.output.timeperframe.denominator = 60;
            if (ioctl(video_fd, VIDIOC_S_PARM, &streamparm) < 0)
            {
                std::cerr << "[cuda encoder] Failed to set frame rate" << std::endl;
                break;
            }

            // Initialize CUDA
            block_height = height / stream_num_;
            size_image = width * height;
            size_bayer_block = width * block_height;
            size_nv12_y_block = width * block_height;
            size_nv12_uv_block = width * block_height / 2;
            gridSize = dim3((width + blockSize.x - 1) / blockSize.x, (block_height + blockSize.y - 1) / blockSize.y);
            streams = (cudaStream_t *)malloc(stream_num_ * sizeof(cudaStream_t));
            for (int i = 0; i < stream_num_; i++)
            {
                cudaStreamCreate(&streams[i]);
            }
            cudaMalloc((void **)&d_bayer, size_image);
            cudaMalloc((void **)&d_nv12, size_image * 3 / 2);
            nv12 = get_cuda_buffer(size_image * 3 / 2);

            std::cout << "[cuda encoder] Encoding video stream..." << std::endl;
            is_init = true;
        }

        bayer = static_cast<unsigned char *>(pImage->GetData());

        for (int i = 0; i < stream_num_; i++)
        {
            cudaMemcpyAsync(
                d_bayer + i * size_bayer_block,
                bayer + i * size_bayer_block,
                size_bayer_block,
                cudaMemcpyHostToDevice,
                streams[i]);

            bayer2nv12_kernel<<<gridSize, blockSize, 0, streams[i]>>>(
                d_bayer + i * size_bayer_block,
                d_nv12 + i * size_nv12_y_block,
                d_nv12 + size_image + i * size_nv12_uv_block,
                width,
                block_height);

            // Copy NV12 Y plane to host
            cudaMemcpyAsync(
                nv12 + i * size_nv12_y_block,
                d_nv12 + i * size_nv12_y_block,
                size_nv12_y_block,
                cudaMemcpyDeviceToHost,
                streams[i]);
            // Copy NV12 UV plane to host
            cudaMemcpyAsync(
                nv12 + size_image + i * size_nv12_uv_block,
                d_nv12 + size_image + i * size_nv12_uv_block,
                size_nv12_uv_block,
                cudaMemcpyDeviceToHost,
                streams[i]);

            cudaDeviceSynchronize();
        }

        // Write the NV12 (12 bits per pixel) data to the virtual video device as NV12
        if (write(video_fd, nv12, width * height * 3 / 2) == -1)
        {
            std::cerr << "[cuda encoder] Error writing frame to virtual device" << std::endl;
            break;
        }

        pImage->Release();
    }

    // Cleanup
    if (is_init)
    {
        for (int i = 0; i < stream_num_; i++)
        {
            cudaStreamDestroy(streams[i]);
        }
        free(streams);
        streams = nullptr;
        cudaFree(d_bayer);
        d_bayer = nullptr;
        cudaFree(d_nv12);
        d_nv12 = nullptr;
        free_cuda_buffer(nv12);
        nv12 = nullptr;
    }
    bayer = nullptr;
    pImage = nullptr;
    camera->EndAcquisition();
    camera->DeInit();
    camera = nullptr;
    camList.Clear();
    system_c->ReleaseInstance();
    system_c = nullptr;
    is_init = false;
}