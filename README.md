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

## Optional YUYV422 test
The stream created above is in the wrong format for FleetMQ to be able to stream it. To stream it, it needs to be converted into the YUYV422 format. Technically, `ffmpeg` is capable of this, but it adds a huge amount of latency. To see this test and try it's compatibility with FleetMQ, run the following:
```
ffmpeg -f video4linux2 -input_format rgb24 -i /dev/video16 -pix_fmt yuyv422 -f v4l2 /dev/video17
```

New the new YUYV422 stream is available on `/dev/video17`. To view it directly, run
```
ffplay /dev/video17
```

To see how it looks with FleetMQ, run the FleetMQ streaming stack and make sure to include `/dev/video17` in the docker compose file.