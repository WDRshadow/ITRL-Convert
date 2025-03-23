#ifndef RGB2YUYV_H
#define RGB2YUYV_H

/**
 * Initialize CUDA buffers.
 *
 * @param width Width of the image
 * @param height Height of the image
 * @param stream_num Number of CUDA streams
 */
void init_rgb2yuyv_cuda(unsigned int width, unsigned int height, int stream_num);

/**
 * Convert RGB24 to YUYV422 with CUDA. 
 *
 * Fixed-point arithmetic optimization is used to speed up the calculation.
 *
 * @param rgb24 RGB24 data
 * @param yuyv422 YUYV422 data (output)
 */
void rgb2yuyv_cuda(const unsigned char *rgb24, unsigned char *yuyv422);

/**
 * Cleanup CUDA buffers.
 */
void cleanup_rgb2yuyv_cuda();

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

#endif // RGB2YUYV_H
