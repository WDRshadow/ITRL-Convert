# Build executable file

```bash
mkdir build
cd build
cmake ..
make
```

# Fisheye criteria

1. Save your images in `data/*.jpg` which is taken by fisheye camera with a chessboard that has 10x7 vertices and each square size is 2.5cm in A4 paper.
2. Run the executable file `fisheye_criteria` in the build directory.
3. The program will output the matrix with distortion coefficients in `fisheye_calibration.yaml`.

# Fisheye undistortion

1. Run the executable file `fisheye_undistort <image>` in the build directory with the image you want to undistort.
2. The program will output the undistorted image in the same directory.

# Homography criteria

1. Save your points in `homography_points.yaml` with the following format:
    ```yaml
    %YAML:1.0
    ---
    #Formet: [Left front wheel x/y, right front wheel x/y, left front 50m x/y, right front 50m x/y]
    points: [ 767., 2047., 2303., 2047., 1023., 1365., 2047., 1365. ]
    ```
2. Run the executable file `homography_criteria` in the build directory.
3. The program will output the homography matrix in `homography_calibration.yaml`.


    Note: The points should be the pixel coordinates from a undistorted image.
