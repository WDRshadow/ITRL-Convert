#include <Spinnaker.h>
#include <cuda.h>
#include <iostream>

#include "nvEncodeAPI.h"
#include "cuda_stream.h"
#include "bayer2nv12.h"

#define CUDA_STREAMS 8

bool is_init = false;
Spinnaker::SystemPtr system_c = nullptr;
Spinnaker::CameraPtr camera = nullptr;
Spinnaker::CameraList camList;
Spinnaker::ImagePtr pImage = nullptr;
unsigned char *bayer = nullptr;

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

NV_ENCODE_API_FUNCTION_LIST nvencFunc = {0};
NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS sessionExParams = {0};
NV_ENC_INITIALIZE_PARAMS initParams = {0};
NV_ENC_CONFIG encodeConfig = {0};
NV_ENC_CREATE_INPUT_BUFFER createInputBufferParams = {0};
NV_ENC_CREATE_BITSTREAM_BUFFER createBitstreamBufferParams = {0};
NV_ENC_PIC_PARAMS picParams = {0};
NV_ENC_LOCK_BITSTREAM lockBitstreamData = {0};
CUcontext cuContext = nullptr;
CUdevice cuDevice;
void *encoder = nullptr;
void *inputBuffer = nullptr;
void *bitstreamBuffer = nullptr;

int main()
{
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

    unsigned int width;
    unsigned int height;

    while (true)
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

            // Initialize CUDA
            cuInit(0);
            cuDeviceGet(&cuDevice, 0);
            cuCtxCreate(&cuContext, 0, cuDevice);

            // Initilize NVENC
            nvencFunc.version = NV_ENCODE_API_FUNCTION_LIST_VER;
            if (NvEncodeAPICreateInstance(&nvencFunc) != NV_ENC_SUCCESS)
            {
                std::cerr << "[cuda encoder] Cannot load NVENC API" << std::endl;
                return -1;
            }

            // Create encoder
            sessionExParams.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
            sessionExParams.device = cuContext;
            sessionExParams.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
            sessionExParams.apiVersion = NVENCAPI_VERSION;
            nvencFunc.nvEncOpenEncodeSessionEx(&sessionExParams, &encoder);

            initParams.version = NV_ENC_INITIALIZE_PARAMS_VER;
            initParams.encodeWidth = width;
            initParams.encodeHeight = height;
            initParams.darWidth = width / 2;
            initParams.darHeight = height / 2;
            initParams.frameRateNum = 30;
            initParams.frameRateDen = 1;
            initParams.enablePTD = 1;
            initParams.reportSliceOffsets = 0;
            initParams.enableSubFrameWrite = 0;
            initParams.maxEncodeWidth = width;
            initParams.maxEncodeHeight = height;
            initParams.encodeGUID = NV_ENC_CODEC_HEVC_GUID;
            initParams.presetGUID = NV_ENC_PRESET_P3_GUID;

            encodeConfig.version = NV_ENC_CONFIG_VER;
            encodeConfig.rcParams.averageBitRate = 5000000;
            encodeConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
            initParams.encodeConfig = &encodeConfig;

            createInputBufferParams.version = NV_ENC_CREATE_INPUT_BUFFER_VER;
            createInputBufferParams.width = width;
            createInputBufferParams.height = height;
            createInputBufferParams.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
            nvencFunc.nvEncCreateInputBuffer(encoder, &createInputBufferParams);
            inputBuffer = createInputBufferParams.inputBuffer;

            createBitstreamBufferParams.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
            createBitstreamBufferParams.size = width * height * 2;
            nvencFunc.nvEncCreateBitstreamBuffer(encoder, &createBitstreamBufferParams);
            bitstreamBuffer = createBitstreamBufferParams.bitstreamBuffer;

            picParams.version = NV_ENC_PIC_PARAMS_VER;
            picParams.inputBuffer = inputBuffer;
            picParams.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
            picParams.inputWidth = width;
            picParams.inputHeight = height;
            picParams.outputBitstream = bitstreamBuffer;
            picParams.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
            picParams.encodePicFlags = 0;

            lockBitstreamData.version = NV_ENC_LOCK_BITSTREAM_VER;
            lockBitstreamData.outputBitstream = bitstreamBuffer;

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
            cudaDeviceSynchronize();
        }

        nvencFunc.nvEncEncodePicture(encoder, &picParams);

        nvencFunc.nvEncLockBitstream(encoder, &lockBitstreamData);

        pImage->Release();
    }

    // Cleanup

    nvencFunc.nvEncDestroyEncoder(encoder);
    cuCtxDestroy(cuContext);

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