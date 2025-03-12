#ifndef SPINNAKER_STREAM_H
#define SPINNAKER_STREAM_H

int configure_video_device(int video_fd, int width, int height);

// Function to capture frames from the FLIR camera and stream them in supported formats
void capture_frames(const char* video_device, const std::string& ip, int port);

#endif //SPINNAKER_STREAM_H
