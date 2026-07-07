# Measure Sim-to-Real Gap: Designing an Affordable Real-World Benchmark Platform for Reinforcement Learning in AIoT Systems

Instructions is under writing ...

This repository is from another repository https://github.com/RongpingZhou/rwrl that the author has been working on since 2025.

Some code is from the below repositories:

stable-baselines3

https://github.com/DLR-RM/stable-baselines3

rl-baselines3-zoo

https://github.com/DLR-RM/rl-baselines3-zoo

rliable

https://github.com/google-research/rliable

A3C_Keras_FlappyBird

https://github.com/shalabhsingh/A3C_Keras_FlappyBird

These code was used and modified for this project.

**Hardware**:

The hardware is only tested under Linux

Teensy 4.1 Head

https://core-electronics.com.au/teensy-4-1-headers.html

Use the program in the below repository to program Teensy to be a emulated USB keyboard

https://github.com/RongpingZhou/etherkey

which was forked from

https://github.com/Flowm/etherkey

USB to serial cable:

https://core-electronics.com.au/usb-to-ttl-serial-uart-rs232-adaptor-pl2303hx.html

Aperture, focus and zoom manually controllable camera which transmits video via UVC standard, such as the camera from the below link:

https://www.amazon.com.au/ELP-Lightburn-Varifocal-Webcamera-Conference/dp/B0F3DD6V8M/ref=sr_1_1?crid=17LLC0IYNKTG7&dib=eyJ2IjoiMSJ9.BKGsq-shR9lY8ZYDeLpR9g.dxFBvqA9Hqx-8bhT_OJP5YDbmrjXGzicfgcBY6ZwUxs&dib_tag=se&keywords=ELP+12mp+USB+Camera+Manual+Zoom+Webcam+for+Computer+1080P+120fps+Variable+Focus+Lightburn+Camera+High+Speed+3.6-10mm+3X+Zoom+USB2.0+PC+Cam+Varifocal+USB+Security+Webcamera+for+Video+Conference&qid=1749221425&s=computers&sprefix=elp+12mp+usb+camera+manual+zoom+webcam+for+computer+1080p+120fps+variable+focus+lightburn+camera+high+speed+3.6-10mm+3x+zoom+usb2.0+pc+cam+varifocal+usb+security+webcamera+for+video+conference%2Ccomputers%2C231&sr=1-1

Camera stand:

https://www.amazon.com.au/OXENDURE-Suspension-Scissor-Compatible-Logitech/dp/B07RNHZYGW/ref=asc_df_B07RNHZYGW/?tag=googleshopdsk-22&linkCode=df0&hvadid=712260569166&hvpos=&hvnetw=g&hvrand=6101658522981671598&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9071771&hvtargid=pla-941430588700&psc=1&mcid=8b421920465c3cbdbf038e092d23ec2d&gad_source=1

USB to DB9/RS232 cable

https://www.amazon.com.au/Cable-Matters-RS-232-Male-Serial/dp/B00J4N9T9C/ref=sr_1_8?crid=2HTETQ8IXOOP4&dib=eyJ2IjoiMSJ9.ncWkgmSce2_EPU1klFLw2a82f3FJlxGjO1j_ccMtA066s2TgXjBt-MBBBGKZM3zo2qaNdSLJUxB3k6tT6gorIKhy8Q2wepuORPMj7mQk3BTstHWIQSkMQDliMBRKvyV1rq9LtS3dfSPytWfC1wIKqEEhbJBf8480OHpjlxA8OuCwdg8Hw_f2i4TFe4t4_xgsaQDZCBNDfqGiqv1xe6aaiYucSWDOmMlSeLKzWkZ_3jVNXkJ6DnoIqqsuKqgOuncOw1XtOvnD7vCrQLQ307lIE5pILHSQjWzIoZ5PW8Hny2s.6M28x9Krt0TemLBet5CD6V7NaPCZZhuiehl5cGjb-Vk&dib_tag=se&keywords=usb%2Bto%2Brs232&qid=1749219557&s=computers&sprefix=USB%2Bto%2Brs%2Ccomputers%2C273&sr=1-8&th=1

Null modem

https://www.amazon.com.au/Dcnhfdsw-Serial-Connection-Converter-Accessory/dp/B0GXLHDRND/ref=sr_1_2_sspa?crid=ARTV0QC544CM&dib=eyJ2IjoiMSJ9.uyh6KazHAOPnjF8W5YyR_LO2nCKEazWpzBueG4n2b9wM5HkvK9Z6YnsXeTPAxNydhuv8QKOXbbNYz67fY3mlBDD65z213YAiX2-HypHKpr4u0dzUncfCijd59We6ydDmtTNVhRZepEaMd1pJrW0hRf4TW8MWmQVt__nM8Bhxit1KEdS7BoseVgdatxa73xSqJtoV1pY0Vr0C9sqG-7j6AH-BoYNXgZsLwnZj0upC2qnm7iIH2kSw1KfKP_ws1CyOmxyizxBG8_x_8hqQs47U37TYnxA6fbvqLSEveekDUjs.ce_api3XdlRqLeg7uH00Xumf-Yr2H1gUCNSJgXP0_Bw&dib_tag=se&keywords=null+modem&qid=1783443492&sprefix=null+modem%2Caps%2C242&sr=8-2-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&psc=1


**Hardware Connection**:

**How to run the program**:

Connect the camera to the machine, install the below apt package:

```shell
sudo apt install v4l-utils

# use the below command to find which device is the connected camera
v4l2-ctl --list-devices

# if the camera connected to /dev/video0, use the below command to check
v4l2-ctl -d /dev/video0 --info

# connect the Teensy to both computers, 
# the the USB to UART adapter is connected to th host machine
# the Teensy is connected to the machine running the game, 
# use the below command to find the device on the host machine, usually it would be /dev/ttyUSB0
ls /dev/tty*

# use git clone to download the program to ~/mygit/rl
mkdir -p ~/mygit/rl
cd ~/mygit/rl
git clone https://github.com/RongpingZhou/autorl.git

# run the below command to ensure the program inside a docker container to use the host machine's display
xhost +"local:docker@"

# I have uploaded a docker image to docker hub, so run the below command to start a docker container for the program
docker run --gpus all -u root -ti --rm -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /dev/snd:/dev/snd:rw -v /dev/ttyUSB0:/dev/ttyUSB0:rw -v /dev/video2:/dev/video0:rw -v $(realpath ~/mygit/rl/):/rl/ -v $(realpath ~/cache/datasets/):/cache/datasets -e DISPLAY=unix$DISPLAY -p 8888:8888 --privileged zrongping/ubuntu2204_cuda12-4-1_cudnn9-1-0-70-1_drl-pytorch_noah-vega:latest

# in the docker container prompt
cd /rl/autorl/deep_rl_zoo/00_project

# test program
python flappybird_test_with_sensor_actuator.py

# train program
python flappybird_learning_with_sensor_actuator.py

# or without hardware
python flappybird_test.py
python flappybird_learning.py

# inside docker container, you can also run the programs from the repositories of Deep RL Zoo and Vega
# for example
cd /rl/autorl

# AutoML CARS
python /rl/autorl/vega/tools/run_pipeline.py /rl/autorl/examples/nas/cars/cars.yml

cd /rl/autorl/deep_rl_zoo/00_project

# test wann
python wann_test.py -p p/biped.json -i champions/biped.out --nReps 3 --view True

# train wann
python wann_train.py -p p/biped.json -n 8
```
