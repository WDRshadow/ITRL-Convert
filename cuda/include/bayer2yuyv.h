#ifndef BAYER2YUYV_H
#define BAYER2YUYV_H

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

#endif // BAYER2YUYV_H