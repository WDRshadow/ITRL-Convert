# Building the binary
```
./build.sh
```
When re-building you may need to manually remove the build files and binary

# Running the example

First prepare the video devices for receiving the processed video. To do this, run the v4l2 script:
```
sudo ./init_v4l2.sh
```

Then, you can start streaming video to the first video device by running the binary built in the previous step with
```
sudo ./spinnaker_stream
```

Now the stream is available on `/dev/video16`. If you want to view it, open a new terminal and run
```
ffplay /dev/video16
```

## RGB24 to YUYV422 Baseline test

In baseline test, we randomly create 100 image with each 3072x2048 pixels and convert it to YUYV422 format. The test result of time taken to process one image (average) is as follows (run three time and take the average):

    Note: The test device is NVIDIA Jetson AGX Xavier (32G).

| Processing Type | Time (ms) |
|-----------------|-----------|
| Sequential      | 59.0      |
| Parallel        | 10.9      |
| CUDA            | 17.1      |

For other devices, just for reference, the test result is shown below:

    Note: The test device is AMD 5800X (8vCPU) with RTX4070.

| Processing Type | Time (ms) |
|-----------------|-----------|
| Sequential      | 77.8      |
| Parallel        | 17.2      |
| CUDA            | 4.2       |

## RGB24 to YUYV422 FleetMQ streaming

The converting code has been implemented. You can directly convert the images stream by following the previous steps and running the following command:
```bash
./build.sh && sudo ./spinnaker_stream
```
Then you can use `Cheese` or `ffplay /dev/video16` to view the video stream.

---

For testing, we capture 1000 images and convert them from Bayer to YUYV format then directly write to video device '/dev/video16'. A timmer is used to measure the time taken to process one image (all the way from Bayer to YUYB).

The conversion is implemented using sequential processing, parallel processing, and CUDA processing. The test result of time taken to process one image (average) is as follows (run three time and take the average):

    Note: The test device is NVIDIA Jetson AGX Xavier (32G).

|  Processing Type  | Time (ms) |
|-------------------|-----------|
| Bayer to RGB only | 8.7       |
| Sequential        | 47.6      |
| Parallel          | 16.0      |
| CUDA              | 16.0      |

