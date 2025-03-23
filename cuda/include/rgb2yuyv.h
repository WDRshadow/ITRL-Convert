#ifndef RGB2YUYV_H
#define RGB2YUYV_H

/**
 * Convert RGB24 to YUYV422 with CUDA. 
 *
 * Fixed-point arithmetic optimization is used to speed up the calculation.
 *
 * @param rgb24 RGB24 data
 * @param yuyv422 YUYV422 data (output)
 * @param width Image width
 * @param height Image height
 * @param stream_num Stream number
 */
void rgb2yuyv_cuda(const unsigned char *rgb24, unsigned char *yuyv422, unsigned int width,
                   unsigned int height, int stream_num);

/**
 * Cleanup CUDA buffers.
 * @param stream_num Stream number
 */
void cleanup_cuda_buffers(int stream_num);

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
