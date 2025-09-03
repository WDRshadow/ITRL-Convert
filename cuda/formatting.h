#ifndef FORMATTING_H
#define FORMATTING_H

#include <cstddef>

#define D_BGRA2RGB 0
#define RGB2YUYV 1

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

class CudaImageConverter
{
protected:
    const unsigned int width;
    const unsigned int height;
    const int stream_num;

    const unsigned int block_height;
    const size_t size_bgra_block;
    const size_t size_rgb_block;
    const size_t size_yuyv_block;
    unsigned char *d_yuyv = nullptr;
    unsigned char *d_rgb = nullptr;
    const int mode;

    cudaStream_t *streams = nullptr;
    dim3 *blockSize;
    dim3 *gridSize;

public:
    /**
     * Constructor for CudaImageConverter.
     *
     * @param width Width of the image
     * @param height Height of the image
     * @param stream_num Number of CUDA streams
     */
    CudaImageConverter(unsigned int width, unsigned int height, int stream_num, int mode);
    ~CudaImageConverter();
    CudaImageConverter(const CudaImageConverter &) = delete;
    CudaImageConverter &operator=(const CudaImageConverter &) = delete;
    CudaImageConverter(CudaImageConverter &&) = delete;
    CudaImageConverter &operator=(CudaImageConverter &&) = delete;

    /**
     * Convert pixel format with CUDA.
     *
     * @param src Input data
     * @param dst Output data
     */
    void convert(const unsigned char *src, unsigned char *dst);
};

#endif // FORMATTING_H