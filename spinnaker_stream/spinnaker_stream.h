#ifndef SPINNAKER_STREAM_H
#define SPINNAKER_STREAM_H

// Function to capture frames from the FLIR camera and stream them in supported formats
void capture_frames(const char* video_device, const std::string& ip, int port, bool &signal, int fps, int delay_ms, const char* logger, bool is_hmi, bool is_p_hmi, int scale);

#endif //SPINNAKER_STREAM_H
