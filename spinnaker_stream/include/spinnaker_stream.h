#ifndef SPINNAKER_STREAM_H
#define SPINNAKER_STREAM_H

extern "C" {
    // Function to capture frames from the FLIR camera and stream them in supported formats
    void capture_frames(const char* video_device);
    }

#endif //SPINNAKER_STREAM_H
