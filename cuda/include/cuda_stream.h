#ifndef CUDA_BUFFER_H
#define CUDA_BUFFER_H

/**
 * Initialize CUDA buffers.
 *
 * @param width Width of the image
 * @param height Height of the image
 * @param stream_num Number of CUDA streams
 */
void init_bayer2yuyv_cuda(unsigned int width, unsigned int height, int stream_num);

/**
 * Convert Bayer RG to YUYV422 with CUDA. 
 *
 * @param bayer Bayer RG data (input)
 * @param yuyv YUYV422 data (output)
 */
void bayer2yuyv_cuda(const unsigned char *bayer, unsigned char *yuyv);

/**
 * Cleanup CUDA buffers.
 */
void cleanup_bayer2yuyv_cuda();

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

#endif // CUDA_BUFFER_H