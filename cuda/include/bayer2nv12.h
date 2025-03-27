#ifndef BAYER2NV12_H
#define BAYER2NV12_H

#ifdef __CUDACC__
__global__ void bayer2nv12_kernel(const unsigned char *bayer, unsigned char *y_plane,
                                  unsigned char *uv_plane, int width, int height);
#endif

#endif