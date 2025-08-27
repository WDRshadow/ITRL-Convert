#ifndef RESOLUTION_H
#define RESOLUTION_H

#ifdef __CUDACC__
__global__ void rgb_resolution_kernel(const unsigned char *src, unsigned char *dst, unsigned int width, unsigned int height, unsigned int orig_width, int scale);
#endif

#define SCALE_1 1
#define SCALE_2 2

typedef struct CUstream_st *cudaStream_t;
typedef struct dim3 dim3;
typedef long unsigned int size_t;

class CudaResolution
{
protected:
    const unsigned int width;
    const unsigned int height;
    const unsigned int new_width;
    const unsigned int new_height;
    const int stream_num;
    const int scale;

    const unsigned int block_new_height;
    const size_t size_src_block;
    const size_t size_dst_block;
    unsigned char *d_src = nullptr;
    unsigned char *d_dst = nullptr;

    cudaStream_t *streams = nullptr;
    dim3 *blockSize;
    dim3 *gridSize;

public:
    /**
     * Constructor for CudaResolution.
     *
     * @param width Width of the image
     * @param height Height of the image
     * @param stream_num Number of CUDA streams
     * @param scale Resolution scale (SCALE_4K, SCALE_2K, SCALE_FHD, SCALE_HD)
     */
    CudaResolution(unsigned int width, unsigned int height, int stream_num, int scale);
    ~CudaResolution();
    CudaResolution(const CudaResolution &) = delete;
    CudaResolution &operator=(const CudaResolution &) = delete;
    CudaResolution(CudaResolution &&) = delete;
    CudaResolution &operator=(CudaResolution &&) = delete;

    /**
     * Convert image with CUDA.
     *
     * @param src Input data
     * @param dst Output data
     */
    void convert(const unsigned char *src, unsigned char *dst);
};

#endif // RESOLUTION_H