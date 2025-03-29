#include <Spinnaker.h>
#include <cuda.h>
#include <iostream>

#include "cuda_stream.h"
#include "bayer2nv12.h"
#include "h265_encoder.h"

#define CUDA_STREAMS 8

bool is_init = false;
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
    // Initialize Spinnaker
    system_c = Spinnaker::System::GetInstance();
    camList = system_c->GetCameras();
    if (camList.GetSize() == 0)
    {
        std::cerr << "[spinnaker stream] No cameras detected!" << std::endl;
        return -1;
    }
    camera = camList.GetByIndex(0);
    camera->Init();
    camera->BeginAcquisition();

    unsigned int width;
    unsigned int height;

    int count = 0;

    while (count++ < 10)
    {
        pImage = camera->GetNextImage();

        if (pImage->IsIncomplete())
        {
            std::cerr << "[spinnaker stream] Image incomplete: " << pImage->GetImageStatus() << std::endl;
            continue;
        }

        // Print the pixel format only once and initialize CUDA
        if (!is_init)
        {
            // Image size
            width = pImage->GetWidth();
            height = pImage->GetHeight();
            std::cout << "[cuda encoder] Image size: " << pImage->GetWidth() << "x" << pImage->GetHeight() << std::endl;

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

            // Initialize H265 encoder
            init_encoder(width, height);

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

        encode_frame(nv12, size_image * 3 / 2);

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
        cleanup_encoder();
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