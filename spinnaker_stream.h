#ifndef SPINNAKER_STREAM_H
#define SPINNAKER_STREAM_H

#ifdef __cplusplus
extern "C" {
#endif

void convert_rgb24_to_yuyv_gpu(const uint8_t *rgb24, uint8_t *yuyv422, int width, int height);

#ifdef __cplusplus
}
#endif

#endif  // SPINNAKER_STREAM_H
