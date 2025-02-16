#ifndef CONVERT_RGB24_TO_YUYV_CUDA_H
#define CONVERT_RGB24_TO_YUYV_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

void convert_rgb24_to_yuyv_cuda(const unsigned char *rgb24, unsigned char *yuyv422, int width, int height);
void cleanup_cuda_buffers();

#ifdef __cplusplus
}
#endif

#endif  // CONVERT_RGB24_TO_YUYV_CUDA_H
