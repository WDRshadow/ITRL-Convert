#ifndef CONVERT_RGB24_TO_YUYV_CUDA_H
#define CONVERT_RGB24_TO_YUYV_CUDA_H

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * Convert RGB24 to YUYV422 with CUDA. Using the following formulas:
     *
     * 1. Y = 0.299R + 0.587G + 0.114B
     * 2. U = -0.14713R - 0.28886G + 0.436B
     * 3. V = 0.615R - 0.51499G - 0.10001B
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

#ifdef __cplusplus
}
#endif

#endif // CONVERT_RGB24_TO_YUYV_CUDA_H
