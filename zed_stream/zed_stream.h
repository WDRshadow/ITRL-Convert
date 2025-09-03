#ifndef ZED_STREAM_H
#define ZED_STREAM_H

void capture_frames(const char* video_device, const std::string& ip, int port, bool &signal, int fps, int delay_ms, const char* logger, bool is_hmi, bool is_p_hmi, int scale);

#endif // ZED_STREAM_H