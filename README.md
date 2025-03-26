# Build the binary
```bash
mkdir build
cd build
cmake ..
make -j
```

You can find the binary file `rcve_stream` in the `build` directory.

# Run the video stream

First prepare the video devices for receiving the processed video. To do this, run the v4l2 script:
```bash
./init_v4l2.sh
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

# Benchmark

For production testing, we capture 1000 images and convert them from Bayer to YUYV format then directly write to video device '/dev/video16'. A timmer is used to measure the time taken to process one image (all the way from Bayer to YUYB).

The conversion is implemented using sequential processing, parallel processing, and CUDA processing. The test result of time taken to process one image (average) is as follows (run three time and take the average):

    Note: The test device is NVIDIA Jetson AGX Xavier (32G).

| Processing Type      | Time (ms) |
|----------------------|-----------|
| Sequential           | 47.6      |
| Parallel             | 16.0      |
| CUDA                 | 8.0       |
