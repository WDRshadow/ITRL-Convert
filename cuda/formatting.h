#ifndef FORMATTING_H
#define FORMATTING_H

#include <cstddef>

#ifdef __CUDACC__
__global__ void bayer2rgb_kernel(const unsigned char *bayer, unsigned char *rgb, unsigned int width, unsigned int height);
__global__ void rgb2yuyv_kernel(const unsigned char *rgb, unsigned char *yuyv, unsigned int width, unsigned int height);
#endif

#define BAYER2YUYV 0
#define BAYER2RGB 1
#define RGB2YUYV 2

/**
 * Use cudaHostAlloc to allocate a pinned memory buffer.
 *
 * @param size Size of the buffer
 * @return Pointer to the buffer
 */
unsigned char *get_cuda_buffer(size_t size);

/**
 * Free the CUDA buffer.
 */
void free_cuda_buffer(unsigned char *buffer);

typedef struct CUstream_st *cudaStream_t;
typedef struct dim3 dim3;
typedef long unsigned int size_t;

class CudaImageConverter {
protected:
    const unsigned int width;
    const unsigned int height;
    const int stream_num;
    const int mode;

    const unsigned int block_height;
    const size_t size_bayer_block;
    const size_t size_rgb_block;
    const size_t size_yuyv_block;
    unsigned char *d_bayer = nullptr;
    unsigned char *d_rgb = nullptr;
    unsigned char *d_yuyv = nullptr;

    cudaStream_t *streams = nullptr;
    dim3 *blockSize;
    dim3 *gridSize_1;
    dim3 *gridSize_2;

public:
    /**
     * Constructor for CudaImageConverter.
     *
     * @param width Width of the image
     * @param height Height of the image
     * @param stream_num Number of CUDA streams
     * @param mode Conversion mode (BAYER2YUYV, BAYER2RGB, RGB2YUYV)
     */
    CudaImageConverter(unsigned int width, unsigned int height, int stream_num, int mode);
    ~CudaImageConverter();
    CudaImageConverter(const CudaImageConverter&) = delete;
    CudaImageConverter& operator=(const CudaImageConverter&) = delete;
    CudaImageConverter(CudaImageConverter&&) = delete;
    CudaImageConverter& operator=(CudaImageConverter&&) = delete;

    /**
     * Convert pixel format with CUDA. 
     *
     * @param src Input data
     * @param dst Output data
     */
    void convert(const unsigned char *src, unsigned char *dst);
};

#endif // FORMATTING_H