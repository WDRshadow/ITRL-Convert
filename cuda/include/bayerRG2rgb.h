#ifndef BAYER_RG2RGB_H
#define BAYER_RG2RGB_H

/**
 * Initialize CUDA buffers.
 *
 * @param width Width of the image
 * @param height Height of the image
 * @param stream_num Number of CUDA streams
 */
void init_bayerRG2rgb_cuda(unsigned int width, unsigned int height, int stream_num);

/**
 * Convert Bayer RG to RGB24 with CUDA.
 *
 * @param bayerHost Bayer RG data
 * @param rgbHost RGB24 data (output)
 */
void bayerRG2rgb_cuda(const unsigned char *bayerHost, unsigned char *rgbHost);

/**
 * Cleanup CUDA buffers.
 */
void cleanup_bayerRG2rgb_cuda();

#endif // BAYER_RG2RGB_H