# Measure Sim-to-Real Gap: Designing an Affordable Real-World Benchmark Platform for Reinforcement Learning in AIoT Systems

Instructions is under writing:

**Hardware**:

The hardware is only tested under Linux

Teensy 4.1 Head

https://core-electronics.com.au/teensy-4-1-headers.html

Use the program in the below repository to program Teensy to be a emulated USB keyboard

https://github.com/Flowm/etherkey

use a webcam which transmit video via UVC standard, such as the camera from the below link:

https://www.amazon.com.au/dp/B0CYQ5P6T7/ref=sspa_dk_

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
