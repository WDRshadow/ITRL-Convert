#include <NvVideoEncoder.h>
#include <NvBuffer.h>
#include <fcntl.h>
#include <unistd.h>

#include "h265_encoder.h"

NvVideoEncoder *enc = nullptr;
int h265_fd;

bool h265_is_init = false;
unsigned int h265_width_;
unsigned int h265_height_;

void preQueueOutputBuffers()
{
    int numBuffers = enc->output_plane.getNumBuffers();
    for (int i = 0; i < numBuffers; i++)
    {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[2];
        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));
        v4l2_buf.m.planes = planes;
        v4l2_buf.index = i;
        if (enc->output_plane.qBuffer(v4l2_buf, nullptr) < 0)
        {
            std::cerr << "[h265 coder] 预队列输出平面缓冲区 " << i << " 失败" << std::endl;
        }
    }
}

void init_encoder(unsigned int width, unsigned int height)
{
    h265_fd = open("output.h265", O_CREAT | O_WRONLY | O_TRUNC, 0666);
    enc = NvVideoEncoder::createVideoEncoder("enc0", O_NONBLOCK);

    enc->setCapturePlaneFormat(V4L2_PIX_FMT_H265, width, height, 2 * 1024 * 1024);
    enc->setOutputPlaneFormat(V4L2_PIX_FMT_NV12M, width, height);

    enc->setFrameRate(60, 1);
    enc->setBitrate(4000000);
    enc->setProfile(V4L2_MPEG_VIDEO_H265_PROFILE_MAIN);
    enc->setInsertSpsPpsAtIdrEnabled(true);

    // 设置缓冲区，出错则进行清理
    if (enc->output_plane.setupPlane(V4L2_MEMORY_MMAP, 10, true, false) < 0)
    {
        std::cerr << "设置输出平面失败" << std::endl;
        cleanup_encoder();
        return;
    }
    if (enc->capture_plane.setupPlane(V4L2_MEMORY_MMAP, 10, true, false) < 0)
    {
        std::cerr << "设置捕获平面失败" << std::endl;
        cleanup_encoder();
        return;
    }

    // 启动流
    if (enc->output_plane.setStreamStatus(true) < 0)
    {
        std::cerr << "启动输出流失败" << std::endl;
        cleanup_encoder();
        return;
    }
    if (enc->capture_plane.setStreamStatus(true) < 0)
    {
        std::cerr << "启动捕获流失败" << std::endl;
        cleanup_encoder();
        return;
    }

    // 预先将所有输出平面缓冲区入队
    preQueueOutputBuffers();

    h265_width_ = width;
    h265_height_ = height;
    h265_is_init = true;
}

void encode_frame(unsigned char *nv12_data, size_t size_nv12)
{
    if (!h265_is_init) {
        std::cerr << "编码器未初始化" << std::endl;
        return;
    }

    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[2];

    memset(&v4l2_buf, 0, sizeof(v4l2_buf));
    memset(planes, 0, sizeof(planes));

    v4l2_buf.m.planes = planes;

    if (enc->output_plane.dqBuffer(v4l2_buf, nullptr, nullptr, 10) < 0)
    {
        std::cerr << "[h265 coder] Error dequeueing buffer" << std::endl;
        return;
    }

    NvBuffer *buffer = enc->output_plane.getNthBuffer(v4l2_buf.index);

    // 拷贝Y平面
    memcpy(buffer->planes[0].data, nv12_data, h265_width_ * h265_height_);
    buffer->planes[0].bytesused = h265_width_ * h265_height_;

    // 拷贝UV平面
    memcpy(buffer->planes[1].data, nv12_data + h265_width_ * h265_height_, h265_width_ * h265_height_ / 2);
    buffer->planes[1].bytesused = h265_width_ * h265_height_ / 2;

    enc->output_plane.qBuffer(v4l2_buf, nullptr);

    while (enc->capture_plane.dqBuffer(v4l2_buf, nullptr, nullptr, 0) == 0)
    {
        NvBuffer::NvBufferPlane &enc_plane = enc->capture_plane.getNthBuffer(v4l2_buf.index)->planes[0];
        write(h265_fd, enc_plane.data, enc_plane.bytesused);
        enc->capture_plane.qBuffer(v4l2_buf, nullptr);
    }
}

void cleanup_encoder()
{
    if (enc)
    {
        enc->capture_plane.setStreamStatus(false);
        enc->output_plane.setStreamStatus(false);
        delete enc;
        enc = nullptr;
    }
    if (h265_fd >= 0)
    {
        close(h265_fd);
        h265_fd = -1;
    }
    h265_is_init = false;
}
