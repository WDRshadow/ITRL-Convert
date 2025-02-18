# Building the binary
```
./build.sh
```
When re-building you may need to manually remove the build files and binary

# Running the example

First prepare the video devices for receiving the processed video. To do this, run the v4l2 script:
```
./init_v4l2.sh
```

Then, you can start streaming video to the first video device by running the binary built in the previous step with
```
./spinnaker_stream
```

Now the stream is available on `/dev/video16`. If you want to view it, open a new terminal and run
```
ffplay /dev/video16
```

## Optional YUYV422 test (No longer work)
The stream created above is in the wrong format for FleetMQ to be able to stream it. To stream it, it needs to be converted into the YUYV422 format. Technically, `ffmpeg` is capable of this, but it adds a huge amount of latency. To see this test and try it's compatibility with FleetMQ, run the following:
```
ffmpeg -f video4linux2 -input_format rgb24 -i /dev/video16 -pix_fmt yuyv422 -f v4l2 /dev/video17
```

New the new YUYV422 stream is available on `/dev/video17`. To view it directly, run
```
ffplay /dev/video17
```

To see how it looks with FleetMQ, run the FleetMQ streaming stack and make sure to include `/dev/video17` in the docker compose file.

## RGB24 to YUYV422 Baseline test

In baseline test, we randomly create 100 image with each 3072x2048 pixels and convert it to YUYV422 format. You can run the test by following the previous steps and running the following command:
```bash
./build_test.sh && ./test
```
The test result of time taken to process one image (average) is as follows (run three time and take the average):

    Note: The test device is NVIDIA Jetson AGX Xavier (32G).

| Processing Type | Time (ms) | CPU clock time (ms) |
|-----------------|-----------|---------------------|
| Sequential      | 59.0      | 58.8                |
| Parallel        | 9.2       | 46.7                |
| CUDA            | 10.0      | 6.3                 |

For other devices, just for reference, the test result is shown below:

    Note: The test device is AMD 5800X (8vCPU) with RTX4070.

| Processing Type | Time (ms) | CPU clock time (ms) |
|-----------------|-----------|---------------------|
| Sequential      | 77.8      | 76.2                |
| Parallel        | 17.2      | 110.2               |
| CUDA            | 4.4       | 4.2                 |

## RGB24 to YUYV422 FleetMQ streaming test

The code has been implemented. You can directly run the test by following the previous steps and running the following command:
```bash
./build.sh && sudo ./spinnaker_stream
```
The code has been changed to capture the 1000 images and convert them from Bayer to YUYV format then directly write to video device '/dev/video16'. A timmer is used to measure the time taken to process one image (all the way from Bayer to YUYB).

The conversion is implemented using sequential processing, parallel processing, and CUDA processing. The test result of time taken to process one image (average) is as follows (run three time and take the average):

    Note: The test device is NVIDIA Jetson AGX Xavier (32G).

|  Processing Type  | Time (ms) | CPU clock time (ms) |
|-------------------|-----------|---------------------|
| Bayer to RGB only | 16.8      | 17.3                |
| Sequential        | 47.6      | 53.6                |
| Parallel          | 16.8      | 50.5                |
| CUDA              | 16.8      | 19.3                |

