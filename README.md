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

micro USB cable

https://www.amazon.com.au/Gopala-Android-Charger-Braided-Charging/dp/B077ZWC2BR/ref=sr_1_15?crid=37HCLJFTP1K2J&dib=eyJ2IjoiMSJ9.oaWdpf_28Urspho_IfJ66WRtd9UoYkK_T_1ZSJz2R3_5NKbCSjjI-bhrcUTUqFLHu0u_N0vjFKy3x2dpAA211c3YVSRRNYyYY9wvFrKzPDJfAT0-pGMJgvwt3jPJl5pWClU21GAZIfiyk96xy5c6eIAZNEhRyE1Vv4Yvvnk-izGGvFU-Q49NiZsRmySOvAJgfff1RrakZOsjm3kqb-rF_nLDnfIuKyiZte-YP8FtFode6OBHZSEogRKhvKsuEZHSdewC6MSPu2q5pzqDVt-BwA3giEQ14-Gn3VNA2ocXE5E.NRiKDRQILpuYffU_3P7YmpizF4fkFHaPpSjqFDdv-rc&dib_tag=se&keywords=USB%2Bmicro%2Bto%2BUSB%2BA&qid=1749222379&refinements=p_n_feature_twelve_browse-bin%3A23322471051&rnid=23322469051&s=computers&sprefix=usb%2Bmicro%2Bto%2Busb%2Ba%2Caps%2C265&sr=1-15&th=1

Aperture, focus and zoom manually controllable camera which transmits video via UVC standard, such as the camera from the below link:

https://www.amazon.com.au/ELP-Lightburn-Varifocal-Webcamera-Conference/dp/B0F3DD6V8M/ref=sr_1_1?crid=17LLC0IYNKTG7&dib=eyJ2IjoiMSJ9.BKGsq-shR9lY8ZYDeLpR9g.dxFBvqA9Hqx-8bhT_OJP5YDbmrjXGzicfgcBY6ZwUxs&dib_tag=se&keywords=ELP+12mp+USB+Camera+Manual+Zoom+Webcam+for+Computer+1080P+120fps+Variable+Focus+Lightburn+Camera+High+Speed+3.6-10mm+3X+Zoom+USB2.0+PC+Cam+Varifocal+USB+Security+Webcamera+for+Video+Conference&qid=1749221425&s=computers&sprefix=elp+12mp+usb+camera+manual+zoom+webcam+for+computer+1080p+120fps+variable+focus+lightburn+camera+high+speed+3.6-10mm+3x+zoom+usb2.0+pc+cam+varifocal+usb+security+webcamera+for+video+conference%2Ccomputers%2C231&sr=1-1

Camera stand:

https://www.amazon.com.au/OXENDURE-Suspension-Scissor-Compatible-Logitech/dp/B07RNHZYGW/ref=asc_df_B07RNHZYGW/?tag=googleshopdsk-22&linkCode=df0&hvadid=712260569166&hvpos=&hvnetw=g&hvrand=6101658522981671598&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9071771&hvtargid=pla-941430588700&psc=1&mcid=8b421920465c3cbdbf038e092d23ec2d&gad_source=1

USB to DB9/RS232 cable

https://www.amazon.com.au/Cable-Matters-RS-232-Male-Serial/dp/B00J4N9T9C/ref=sr_1_8?crid=2HTETQ8IXOOP4&dib=eyJ2IjoiMSJ9.ncWkgmSce2_EPU1klFLw2a82f3FJlxGjO1j_ccMtA066s2TgXjBt-MBBBGKZM3zo2qaNdSLJUxB3k6tT6gorIKhy8Q2wepuORPMj7mQk3BTstHWIQSkMQDliMBRKvyV1rq9LtS3dfSPytWfC1wIKqEEhbJBf8480OHpjlxA8OuCwdg8Hw_f2i4TFe4t4_xgsaQDZCBNDfqGiqv1xe6aaiYucSWDOmMlSeLKzWkZ_3jVNXkJ6DnoIqqsuKqgOuncOw1XtOvnD7vCrQLQ307lIE5pILHSQjWzIoZ5PW8Hny2s.6M28x9Krt0TemLBet5CD6V7NaPCZZhuiehl5cGjb-Vk&dib_tag=se&keywords=usb%2Bto%2Brs232&qid=1749219557&s=computers&sprefix=USB%2Bto%2Brs%2Ccomputers%2C273&sr=1-8&th=1

Null modem

https://www.amazon.com.au/Dcnhfdsw-Serial-Connection-Converter-Accessory/dp/B0GXLHDRND/ref=sr_1_2_sspa?crid=ARTV0QC544CM&dib=eyJ2IjoiMSJ9.uyh6KazHAOPnjF8W5YyR_LO2nCKEazWpzBueG4n2b9wM5HkvK9Z6YnsXeTPAxNydhuv8QKOXbbNYz67fY3mlBDD65z213YAiX2-HypHKpr4u0dzUncfCijd59We6ydDmtTNVhRZepEaMd1pJrW0hRf4TW8MWmQVt__nM8Bhxit1KEdS7BoseVgdatxa73xSqJtoV1pY0Vr0C9sqG-7j6AH-BoYNXgZsLwnZj0upC2qnm7iIH2kSw1KfKP_ws1CyOmxyizxBG8_x_8hqQs47U37TYnxA6fbvqLSEveekDUjs.ce_api3XdlRqLeg7uH00Xumf-Yr2H1gUCNSJgXP0_Bw&dib_tag=se&keywords=null+modem&qid=1783443492&sprefix=null+modem%2Caps%2C242&sr=8-2-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&psc=1


**Real-world input system**

*Hardware connection*

Attach a camera on a camera stand, connect the camera to a USB port of a Ubuntu machine, point the camera to the screen

*How to run the program*

Install docker engine on the Ubuntu mahcine if you haven't done so.

Then run the commands in the below:

```shell
sudo apt install v4l-utils

# use the below command to find which device is the connected camera, for example /dev/video0 or /dev/video1
v4l2-ctl --list-devices

# if the camera connected to /dev/video0, use the below command to check
v4l2-ctl -d /dev/video0 --info

# use git clone to download the program to ~/mygit/rl
mkdir -p ~/mygit/

cd ~/mygit/

git clone https://github.com/RongpingZhou/real_world_program.git

# allow GUI programs running inside Docker containers (on the same machine) to open windows on your host display
xhost +local:docker

# this command will automatically download the docker images from docker hub, you can replace /dev/video0 to /dev/video1 if your machine recognizes the camera as /dev/video1
docker run --gpus all -u root -ti --rm -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /dev/snd:/dev/snd:rw -v /dev/ttyUSB0:/dev/ttyUSB0:rw -v /dev/video0:/dev/video0:rw -v $(realpath ~/mygit/):/rl/ -e DISPLAY=unix$DISPLAY -p 8888:8888 --privileged zrongping/ubuntu2204_cuda12-4-1_cudnn9-1-0-70-1_drl-pytorch_noah-vega:version.20250608

# inside docker container, install gym for the huggingface model loading
pip install gym

# inside docker container, run a test program using huggingface model to see whether the program is working or not
cd /rl/real_world_program/
./rwrl_068_hf_test.sh

# inside docker container, run the test program for the real world input system
# close the game window will stop the program
./rwrl_068_sensor_hf_test.sh

# change the below variables in 
# "code_068_dqn_agent_test_for_real_world_input_system.py" 
# to change the game window position
window_x = 50 # you change it to change the game window position
window_y = 862 # you change it to change the game window position

# the original field of view will show on the screen and the cropped window is also shown on the screen
# use this program to adjust camera position, aperture, zoom, focus 
# to ensure to the game window is in the centre of the original field of view
# change the below variables in 
# "code_068_dqn_agent_test_for_real_world_input_system.py" 
# to fine tune the cropped image
REAL_WORLD_INPUT_HEIGHT_TOP = 90 # you can change it to fine tune the position
REAL_WORLD_INPUT_HEIGHT_BOTTOM = 410 # you can change it to fine tune the position
REAL_WORLD_INPUT_WIDTH_LEFT = 190 # you can change it to fine tune the position
REAL_WORLD_INPUT_WIDTH_RIGHT = 435 # you can change it to fine tune the position

# remember the position data and then use these data to update the same variables in
# "code_067_dqn_agent_training_for_real_world_input_system.py"
REAL_WORLD_INPUT_HEIGHT_TOP = 90 # you can change it to fine tune the position
REAL_WORLD_INPUT_HEIGHT_BOTTOM = 410 # you can change it to fine tune the position
REAL_WORLD_INPUT_WIDTH_LEFT = 190 # you can change it to fine tune the position
REAL_WORLD_INPUT_WIDTH_RIGHT = 435 # you can change it to fine tune the position

# now start the training program to train an agent
./rwrl_067_for_real_world_input_system.sh

# after training, run the real world input system test program to get the test data
./rwrl_068_test_for_real_world_input_system.sh
```

**Real-world system**

*Hardware connection*

Attach a camera on a camera stand, connect the camera to a USB port of a Ubuntu machine, point the camera to the screen

Follow the instructions in https://github.com/RongpingZhou/etherkey to set up and connect the hardware-emulated keyboard between two Ubuntu machines, connect the agent machine and the Teensy board via the USB to serial cable, connect the Teensy board to the game machine via micro USB cable

Use USB to DB9/RS232 cable and null modem to connect two Ubuntu machine, be careful of the null modem quality, I spent some time to figure out that the null modem didn't connect one side TX to the other side's RX, I had to cut the cable and reconnect the wires to ensure TX <--> RX and RX <--> TX

*How to run the program*

Install docker engine on the Ubuntu mahcine if you haven't done so.

Then run the commands in the below:

```shell
sudo apt install v4l-utils

# use the below command to find which device is the connected camera, for example /dev/video0 or /dev/video1
v4l2-ctl --list-devices

# if the camera connected to /dev/video0, use the below command to check
v4l2-ctl -d /dev/video0 --info

# use git clone to download the program to ~/mygit/rl
mkdir -p ~/mygit/

cd ~/mygit/

git clone https://github.com/RongpingZhou/real_world_program.git

# allow GUI programs running inside Docker containers (on the same machine) to open windows on your host display
xhost +local:docker

# this command will automatically download the docker images from docker hub, you can replace /dev/video0 to /dev/video1 if your machine recognizes the camera as /dev/video1
docker run --gpus all -u root -ti --rm -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /dev/snd:/dev/snd:rw -v /dev/ttyUSB0:/dev/ttyUSB0:rw -v /dev/video0:/dev/video0:rw -v $(realpath ~/mygit/):/rl/ -e DISPLAY=unix$DISPLAY -p 8888:8888 --privileged zrongping/ubuntu2204_cuda12-4-1_cudnn9-1-0-70-1_drl-pytorch_noah-vega:version.20250608

# inside docker container, install gym for the huggingface model loading
pip install gym

# inside docker container, run a test program using huggingface model to see whether the program is working or not
cd /rl/real_world_program/

```
