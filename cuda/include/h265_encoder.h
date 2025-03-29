#ifndef H265_ENCODER_H
#define H265_ENCODER_H

void init_encoder(unsigned int width, unsigned int height);

void encode_frame(unsigned char *nv12_data, size_t size_nv12);

void cleanup_encoder();

#endif