# RCVE Video Stream

**The most simple way** to run the video stream on RCVE dev computer is to run following script in any directory. 

```bash
rcve_start.sh
```

The streamming process will be started in the background and the video will be available on `/dev/video16`. The sensor data will be forwarded to port '10002'. 

To stop the process, run the following script in any directory.

```bash
rcve_stop.sh
```

# Build the binary
```bash
mkdir build
cd build
cmake ..
make -j
```

You can find the binary file `flir_stream` in the `build` directory.

# Run the video stream

First prepare the video devices for receiving the processed video. To do this, run the following script:
```bash
sudo rmmod v4l2loopback
sudo modprobe v4l2loopback video_nr=16 card_label="RCVECamera" exclusive_caps=1
```

Then, you can start streaming video to the first video device by running the binary built in the previous step in `build` folder with
```bash
./rcve_stream
```

Now the stream is available on `/dev/video16`. If you want to view it, open a new terminal and run
```bash
ffplay /dev/video16
```

Once the video stream is running, you can use FleetMQ to stream the video to the cloud, or use `Cheese` or `ffplay /dev/video16` to view the video stream.

# Parameters

You can also run the binary with the following parameters (default):
- `-h` to display the help message
- `-d <device>` to specify the video device (`/dev/video16`)
- `-fps <fps>` to specify the frames per second (default is `60`)
- `-s` to add the sensor data to the video stream (`false`)
- `-ip <ip>` to bind the IP address of the sensor data (`0.0.0.0`)
- `-p <port>` to bind the UDP port of the sensor data (`10086`)
- `-log <logger>` to specify the logger file (default is `""`)
- `-fc` to calibrate the fisheye camera
- `-fu <image>` to undistort the image
- `-hc` to calibrate the homography matrix

# Components with sensor data

If the parameter `-s` is added, the software will try to bind the IP address and port to receive the sensor data. If bind successfully, the driver line and velocity components will display on the video.

To display the components, you will need to save the fisheye and homography matrix in the execution directory. You can find the tutorial below.

# Benchmark

For production testing, we capture 1000 images and convert them from Bayer to YUYV format then directly write to video device '/dev/video16'. A timmer is used to measure the time taken to process one image (all the way from Bayer to YUYB).

The conversion is implemented using sequential processing, parallel processing, and CUDA processing. The test result of time taken to process one image (average) is as follows (run three time and take the average):

    Note: The test device is NVIDIA Jetson AGX Xavier (32G).

|  Processing Type       | Time (ms) |
|------------------------|-----------|
| Bayer to RGB only      | 8.7       |
| Sequential             | 47.6      |
| Parallel               | 16.0      |
| CUDA (pure stream)     | 8.0       |
| CUDA (with components) | 10.0      |

# Fisheye criteria

1. Save your images in `data/*.jpg` which is taken by fisheye camera with a chessboard that has 10x7 vertices and each square size is 2.5cm in A4 paper.
2. Run the executable file with parameter `./rcve_stream -fc` in the build directory.
3. The program will output the matrix with distortion coefficients in `fisheye_calibration.yaml`.

# Fisheye undistortion

1. Run the executable file with parameters `./rcve_stream -fu <image>` in the build directory with the image you want to undistort.
2. The program will output the undistorted image in the same directory.

# Homography criteria

1. Save your points in `homography_points.yaml` with the following format:
    ```yaml
    %YAML:1.0
    ---
    #Formet: [Left front wheel x/y, right front wheel x/y, left front 50m x/y, right front 50m x/y]
    points: [ 767., 2047., 2303., 2047., 1023., 1365., 2047., 1365. ]
    ```
2. Run the executable file with parameter `./rcve_stream -hc` in the build directory.
3. The program will output the homography matrix in `homography_calibration.yaml`.


    Note: The points should be the pixel coordinates from a undistorted image.
