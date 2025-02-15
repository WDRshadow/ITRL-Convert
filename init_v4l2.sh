sudo rmmod v4l2loopback

sudo modprobe v4l2loopback video_nr=16,17 card_label="FleetMQRGBCamera,FleetMQYUYVCamera" exclusive_caps=1
#sudo modprobe v4l2loopback video_nr=16 card_label="FleetMQCamera" exclusive_caps=1
