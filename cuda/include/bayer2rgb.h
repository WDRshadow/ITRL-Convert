#ifndef BAYER2RGB_H
#define BAYER2RGB_H

#ifdef __CUDACC__
__global__ void bayer2rgb_kernel(const unsigned char *bayer, unsigned char *rgb, int width, int height);
#endif

/**
 * Initialize CUDA buffers.
 *
 * @param width Width of the image
 * @param height Height of the image
 * @param stream_num Number of CUDA streams
 */
void init_bayer2rgb_cuda(unsigned int width, unsigned int height, int stream_num);

/**
 * Convert Bayer RG to RGB24 with CUDA.
 *
 * @param bayerHost Bayer RG data
 * @param rgbHost RGB24 data (output)
 */
void bayer2rgb_cuda(const unsigned char *bayerHost, unsigned char *rgbHost);

/**
 * Cleanup CUDA buffers.
 */
void cleanup_bayer2rgb_cuda();

#endif // BAYER2RGB_H