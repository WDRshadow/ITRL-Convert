# ZED Camera Video Stream

## Introduction

This project is designed to stream video from a ZED camera to a virtual video device using v4l2loopback. It supports high frame rates (up to 60 FPS) and can overlay additional information such as sensor data and HMI elements onto the video stream. The project is optimized for performance using CUDA for image processing tasks.

## Software architecture

This ZED camera streaming application follows a modular architecture designed for high-performance real-time video processing and sensor data integration. The system is composed of several key modules:

### Workflow Overview

```
┌───────────────┐
│   Camera Img  │◄─────────────────────────────────────────────────┐
└───────────────┘                                                  │
        │                                                          |
        ▼                                                          |
┌───────────────┐       ┌───────────────┐       ┌───────────────┐  |
│   BGRA → RGB  │──────→│  CYRA motion  │──────→│  KBM motion   │  |
└───────────────┘       └───────────────┘       └───────────────┘  |
                                                        │          |
                                                        ▼          |
┌───────────────┐       ┌───────────────┐       ┌───────────────┐  |
│  Merge Image  │◄──────│       -       │◄──────│   Homography  │  |
└───────────────┘       └───────────────┘       └───────────────┘  |
        │                                                          |
        ▼                                                          |
┌───────────────┐       ┌───────────────┐       ┌───────────────┐  |
│   RGB → YUYV  │──────→│ Write to v4l2 │──────→│  Release Img  │──┘ 
└───────────────┘       └───────────────┘       └───────────────┘
```

### Module Descriptions

#### 1. **Main Entry Point** (`main.cpp`)
- Command-line argument parsing and configuration
- Signal handling for graceful shutdown
- Orchestrates initialization of all subsystems
- Supports multiple operation modes: streaming, calibration, and testing

#### 2. **Zed Stream Module** (`zed_stream/`)
- **Purpose**: Core video capture and streaming functionality
- **Key Components**:
  - ZED camera initialization and control using ZED SDK
  - Real-time frame capture with configurable FPS (up to 30)
  - Integration with V4L2 loopback device for video output
  - Multi-threaded sensor data reception and processing

#### 3. **CUDA Module** (`cuda/`)
- **Purpose**: High-performance GPU-accelerated image processing
- **Key Components**:
  - `CudaImageConverter`: Handles BGRA to RGB and RGB to YUYV format conversion
  - `CudaResolution`: Manages image scaling and resolution adjustments
  - Kernel implementations for parallel processing on GPU

#### 4. **Camera Module** (`camera/`)
- **Purpose**: Image processing, calibration, and visual component rendering
- **Key Components**:
  - `Fisheye`: Fisheye camera calibration and undistortion
  - `Homography`: Perspective transformation for top-down view
  - `Component System`: Modular visual overlay system
    - `StreamImage`: Container for multiple visual components
    - `LineComponent`: Renders driving lines and predictions
    - `TextComponent`: Displays text information
    - `ImageComponent`: Handles image overlays
  - `RingBuffer`: Efficient circular buffer for image data
  - `PIDGammaController`: Automatic exposure and gamma correction

#### 5. **Sensor Module** (`sensor/`)
- **Purpose**: Real-time sensor data collection and network communication
- **Key Components**:
  - `SocketBridge`: UDP socket communication for sensor data
  - `SensorAPI`: Unified interface for accessing sensor values
  - `DataLogger`: CSV logging of sensor data with timestamps
  - Multi-port listening (base port, base+1, base+2) for different data streams

#### 6. **Motion Module** (`motion/`)
- **Purpose**: Vehicle motion prediction and kinematic modeling
- **Key Components**:
  - `CYRA Model`: Constant Yaw Rate and Acceleration prediction model
  - `Bicycle Model`: Vehicle dynamics for steering prediction
  - State prediction for autonomous vehicle path planning

### Data Flow

1. **Image Acquisition**: ZED camera captures raw Bayer format images via Spinnaker SDK
2. **GPU Processing**: CUDA kernels convert Bayer → RGB → YUYV with hardware acceleration
3. **Component Rendering**: Visual overlays (driving lines, text, HMI elements) are rendered onto the image
4. **Sensor Integration**: Real-time sensor data is received via UDP and integrated into visual components
5. **Output Streaming**: Processed frames are written to V4L2 loopback device for consumption by external applications

### Performance Optimization

- **Multi-threading**: Separate threads for camera capture, sensor data reception, and processing
- **CUDA Acceleration**: GPU-based image format conversion achieving ~8ms processing time
- **Memory Management**: Pinned CUDA memory for efficient CPU-GPU data transfer
- **Circular Buffers**: Lock-free ring buffers for high-throughput data handling

### Configuration & Calibration

The system supports runtime calibration for:
- **Fisheye Distortion**: Automatic calibration using chessboard patterns
- **Homography Transformation**: Perspective correction for bird's-eye view
- **Sensor Mapping**: Configurable sensor data sources and display parameters

This modular architecture enables easy extension for new sensors, different camera types, and additional visual components while maintaining high performance for real-time applications.



## Dependency

- OpenCV
- v4l2loopback
- CUDA
- ZED SDK

## Usage

### Build the binary
```bash
run/build
```

### Initialize the v4l2 package
```bash
run/init_v4l2
```

### Run the video stream

```bash
run/svea_stream
```

Now the stream is available on `/dev/video16`. If you want to view it, open a new terminal and run
```bash
ffplay /dev/video16
```

## Run the stream with FleetMQ and lower level system of RCVE

### Build the binary
```bash
run/build
```

### Run the stream with FleetMQ
```bash
run/streamming start [-delay <time_ms>] [-hmi] [-p_hmi]
```
For the parameters, please refer to `run/streamming -h`. You can check the log in `run/logs/`.

Once the stream is started, you can view it FleetMQ Web app by navigating to `https://app.fleetmq.com` and logging in with your credentials. The stream will be available under the "Streams" section.

### Restart the stream with FleetMQ
```bash
run/streamming restart [-delay <time_ms>] [-hmi] [-p_hmi]
```

### Stop the stream with FleetMQ
```bash
run/streamming stop
```
The output data will be saved in `run/output/` directory.

## Parameters

You can also run the binary with the following parameters (default):
- `-h` to display the help message
- `-d <device>` to specify the video device (`/dev/video16`)
- `-fps <fps>` to specify the frames per second (default is `60`)
- `-scale <scale>` to specify the scale of the frame size (default: `1`)
- `-delay <time_ms>` to specify the delay time in milliseconds (default: `0`)
- `-s` to add the sensor data to the video stream (`false`)
- `-hmi` to add HMI to the stream
- `-p_hmi` to add Prediction HMI to the stream
- `-ip <ip>` to bind the IP address of the sensor data (`0.0.0.0`)
- `-p <port>` to bind the UDP port of the sensor data (`10086`)
- `-log <logger>` to specify the logger file
- `-fc` to calibrate the fisheye camera
- `-fu <image>` to undistort the image
- `-hc` to calibrate the homography matrix

## Components with sensor data

If the parameter `-s` is added, the software will try to bind the IP address and port to receive the sensor data. If bind successfully, the driver line and velocity components will display on the video.

The software will listen to `<port>` for listening to the lower system data and `<port> + 1` for listening to the control tower data. And also listen to `<port> + 2` for the latency from FleetMQ SDK.

To display the components, you will need to save the fisheye and homography matrix in the execution directory. You can find the tutorial below.

## Benchmark

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

## Fisheye criteria

1. Save your images in `run/data/*.jpg` which is taken by fisheye camera with a chessboard that has 10x7 vertices and each square size is 2.5cm in A4 paper.
2. Run the executable file with parameter `./svea_stream -fc` in the `run` directory.
3. The program will output the matrix with distortion coefficients in `fisheye_calibration.yaml`.

## Fisheye undistortion

1. Run the executable file with parameters `run/svea_stream -fu <image>` in the `run` directory with the image you want to undistort.
2. The program will output the undistorted image in the same directory.

## Homography criteria

1. Save your points in `homography_points.yaml` with the following format:
    ```yaml
    %YAML:1.0
    ---
    #Formet: [Left front wheel x/y, right front wheel x/y, left front 50m x/y, right front 50m x/y]
    points: [ 479., 1079., 1440., 1079., 639., 720., 1280., 720. ]
    ```
2. Run the executable file with parameter `./svea_stream -hc` in the `run` directory.
3. The program will output the homography matrix in `homography_calibration.yaml`.


    Note: The points should be the pixel coordinates from a undistorted image.
