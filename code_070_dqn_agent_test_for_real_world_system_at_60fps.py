# xhost +local:docker

# docker run --gpus all -u root -ti --rm -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /dev/snd:/dev/snd:rw -v /dev/ttyUSB0:/dev/ttyUSB0:rw -v /dev/video0:/dev/video0:rw -v $(realpath ~/mygit/):/rl/ -e DISPLAY=unix$DISPLAY -p 8888:8888 --privileged zrongping/ubuntu2204_cuda12-4-1_cudnn9-1-0-70-1_drl-pytorch_noah-vega:version.20250608

# docker run -u root -ti --rm -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /dev/snd:/dev/snd:rw -v /dev/ttyUSB0:/dev/ttyUSB0:rw -v /dev/ttyUSB1:/dev/ttyUSB1:rw -v /dev/video0:/dev/video0:rw -v $(realpath ~/mygit/):/rl/ -e DISPLAY=unix$DISPLAY -p 8888:8888 --privileged zrongping/ubuntu2204_cuda12-4-1_cudnn9-1-0-70-1_drl-pytorch_noah-vega:version.20250608

# action 0 needs wait time while other keys have physical hold time which cannot be changed by software
# human player won't wait for the key released, so the below wait time is removed
# NO_OP_TIME = float(noops/skip) * SLIGHTLY_MORE_THAN_KEY_HOLD_TIME

import time
import numpy as np
from collections import deque
from typing import Callable, List, Optional, Tuple
import hashlib

import threading
import queue
import json
import random

import sys
sys.path.append("domain/")
sys.path.append("mybuffer/")
sys.path.append("mylibs/")

from evaluation.atari_data import get_human_normalized_score, get_env_id

from mybuffer.replaybm import ReplayBuffer
from mylibs.commands import commands_dict

import pygame

import tkinter as tk

import os

import gymnasium as gym
from gymnasium import Env, logger
from gymnasium.wrappers import TimeLimit
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete

from stable_baselines3.common.utils import get_linear_fn, safe_mean, set_random_seed, polyak_update, get_parameters_by_name

from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.load_from_hub import download_from_hub
from rl_zoo3.utils import StoreDict, get_model_path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime

try:
    import cv2
    cv2.ocl.setUseOpenCL(False)
except ImportError:
    cv2 = None  # type: ignore[assignment]
    
import multiprocessing
import serial

import argparse
from distutils.util import strtobool

import ale_py
gym.register_envs(ale_py)
# gym.pprint_registry()

from datetime import datetime
import glob
import re
from pathlib import Path

# Import TensorBoard utilities for handling log continuity
try:
    from tensorboard_utils import create_filtered_logdir_for_checkpoint, verify_tensorboard_continuity
    TENSORBOARD_UTILS_AVAILABLE = True
except ImportError:
    TENSORBOARD_UTILS_AVAILABLE = False
    print("Warning: tensorboard_utils not available. TensorBoard log filtering disabled.")

import matplotlib
import matplotlib.pyplot as plt

from huggingface_sb3 import EnvironmentName
import yaml

NOOP_MAX = 30
ENVS = 0
FRAMES_SKIP = 4
LEARNING_RATE = 1e-4
BUFFER_SIZE = 100_000  # 100k
LEARNING_STARTS = BUFFER_SIZE  # Number of steps before starting training
GAMMA = 0.99
BATCH_SIZE = 32
MAX_GRAD_NORM = 10.0
TRAINING_FREQ = 4  # Train the agent every `TRAINING_FREQ` steps
TARGET_UPDATE_INTERVAL = 1_000  # Update the target network every `TARGET_UPDATE_FREQ` steps
EXPLORATION_FRACTION = 0.1  # Fraction of entire training period over which the exploration rate is annealed
EXPLORATION_INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy exploration
EXPLORATION_FINAL_EPSILON = 0.01  # Final value of epsilon in epsilon-greedy exploration
TEST_STEP_SIZE =  1_000_000
MAX_TEST_STEPS = 10_000_000  # 10 million steps

IMAGE_CHANNELS = 4
STACK_FRAMES = 4
IMAGE_ROWS = 84
IMAGE_COLS = 84
# SLIGHTLY_MORE_THAN_KEY_HOLD_TIME = 0.067 # elite typist speed
SLIGHTLY_MORE_THAN_KEY_HOLD_TIME = 0.117 # mean typist speed
EYES_PERCEPTION_TIME = 0.0167 # 1/60 seconds
FRAME_DURATION_TIME = 0.0167 # 1/60 seconds
print(f"SLIGHTLY_MORE_THAN_KEY_HOLD_TIME: {SLIGHTLY_MORE_THAN_KEY_HOLD_TIME} seconds")
print(f"EYES_PERCEPTION_TIME: {EYES_PERCEPTION_TIME} seconds")
print(f"FRAME_DURATION_TIME: {FRAME_DURATION_TIME} seconds")
OFFSET = 0.2 # to capture the frame after the key has pressed for 0.2 * hold time, to make sure the frame has the effect of the key press

VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_FPS = 120

# [105:425, 205:450] oringinal position for training
# [100:420, 205:450] minor change for testing
REAL_WORLD_INPUT_HEIGHT_TOP = 105
# REAL_WORLD_INPUT_HEIGHT_TOP = 100
#415
REAL_WORLD_INPUT_HEIGHT_BOTTOM = 425
# REAL_WORLD_INPUT_HEIGHT_BOTTOM = 420
REAL_WORLD_INPUT_WIDTH_LEFT = 205
REAL_WORLD_INPUT_WIDTH_RIGHT = 450

def setup_matplotlib_backend():
    """Configure matplotlib backend based on environment"""

    # Check if running in headless environment
    if os.environ.get('DISPLAY') is None or os.environ.get('SSH_CONNECTION'):
        print("Headless environment detected, using Agg backend")
        matplotlib.use('Agg')
        return 'headless'

    # Try interactive backends in order of preference
    interactive_backends = ['TkAgg', 'Qt5Agg', 'GTK3Agg']

    for backend in interactive_backends:
        try:
            matplotlib.use(backend)
            # Test if backend works
            fig = plt.figure()
            plt.close(fig)
            print(f"Using interactive backend: {backend}")
            return 'interactive'
        except ImportError:
            continue

    # Fallback to Agg if no interactive backend works
    print("No interactive backend available, falling back to Agg")
    matplotlib.use('Agg')
    return 'headless'

matplotlib_backend = setup_matplotlib_backend()

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()

    parser.add_argument('--gym-id', type=str, default="BreakoutNoFrameskip-v4",
        help='the id of the gym environment')
    parser.add_argument("--env", type=EnvironmentName, default="BreakoutNoFrameskip-v4", 
        help="the environment ID for loading huggingface model, should be the same as --gym-id")
    parser.add_argument("--folder", type=str, default="rl-trained-agents", 
        help="Log folder")
    parser.add_argument("--algo", default="dqn", type=str, required=False, choices=list(ALGOS.keys()), 
        help="RL Algorithm")
    parser.add_argument("-n", "--n-timesteps", default=1000, type=int, 
        help="number of timesteps")
    parser.add_argument("--n-episodes", default=1, type=int, 
        help="number of episodes for evaluation")
    parser.add_argument("--num-threads", default=-1, type=int, 
        help="Number of threads for PyTorch (-1 to use default)")
    parser.add_argument("--n-envs", default=1, type=int, 
        help="number of environments")
    parser.add_argument("--exp-id", default=0, type=int, 
        help="Experiment ID (default: 0: latest, -1: no exp folder)")
    parser.add_argument("--verbose", default=1, type=int, 
        help="Verbose mode (0: no output, 1: INFO)")
    parser.add_argument("--device", default="auto", type=str, 
        help="PyTorch device to be use (ex: cpu, cuda...)")
    parser.add_argument("--load-best", action="store_true", default=False, 
        help="Load best model instead of last model if available")
    parser.add_argument("--deterministic", action="store_true", default=False, 
        help="Use deterministic actions")
    parser.add_argument("--load-checkpoint", type=int, 
        help="Load checkpoint instead of last model if available, you must pass the number of timesteps corresponding to it",)
    parser.add_argument("--load-last-checkpoint", action="store_true", default=False, 
        help="Load last checkpoint instead of last model if available")
    parser.add_argument("--stochastic", action="store_true", default=False, 
        help="Use stochastic actions")
    parser.add_argument("--norm-reward", action="store_true", default=False, 
        help="Normalize reward if applicable (trained with ecNormalize)")
    parser.add_argument("--reward-log", default="", type=str, 
        help="Where to log reward")
    parser.add_argument("--gym-packages", type=str, nargs="+", default=[], 
        help="Additional external Gym environment package modules to import")
    parser.add_argument("--env-kwargs", type=str, nargs="+", action=StoreDict, 
        help="Optional keyword argument to pass to the env constructor")
    parser.add_argument("--custom-objects", action="store_true", default=False, 
        help="Use custom objects to solve loading issues")
    parser.add_argument("-P", "--progress", action="store_true", default=False, 
        help="if toggled, display a progress bar using tqdm and rich")
    parser.add_argument("--max-episode-steps", type=int, default=60000,
        help="how many steps to run in one episode in each environment")
    parser.add_argument("--model", type=int, default=4,
        help="model for the agent, 0 is random action, 1 is CNN, 2 is huggingface model, 3 is transformer, 4 is the standard CNN model")
    parser.add_argument('--model-file', type=str, default=None,
        help='the model file name for the agent to load')
    parser.add_argument('--buffer-file', type=str, default=None,
        help='the buffer file name for the agent to load replay buffer')
    parser.add_argument('--checkpoint-file', type=str, default=None,
        help='specific checkpoint file to resume from (if not specified, will find latest)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
        help='directory to save/load checkpoints')
    parser.add_argument('--filter-tensorboard', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help='automatically filter TensorBoard logs when resuming from earlier checkpoint (default: True)')
    parser.add_argument("--training", type=int, default=0,
        help="0 is not training, 1 is training, 2 is transfer training")
    parser.add_argument("--test", type=int, default=0,
        help="0 is not testing, 1 is testing")
    parser.add_argument("--debug", type=int, default=0,
        help="0 is not debugging, 1 is debugging")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--num-steps", type=int, default=5,
        help="how many steps to run in each environment per update")
    parser.add_argument("--fps", type=int, default=300,
        help="frame per second for the environment")
    parser.add_argument("--zoom", type=float, default=1.0,
        help="zoom in the environment, 1.0 is no zoom, 2.0 is double zoom")
    parser.add_argument("--seed", type=int, default=None,
        help="random seed for the environment")
    parser.add_argument("--crop", type=int, default=0,
        help="use sensor or not, 0 is not cropping the window, 1 is cropping the window")
    parser.add_argument("--display", type=int, default=0,
        help="display the screen or not, 0 is not displaying, 1 is displaying")
    parser.add_argument("--plot", type=int, default=0,
        help="show plot during training, 0 is not showing, 1 is showing")
    parser.add_argument("--forpaper", type=int, default=0,
        help="collect data for paper writing, 0 is not collecting, 1 is collecting")

    args = parser.parse_args()
    
    return args

args = parse_args()
print("args: ", args)
print(vars(args))

if args.display == 1 or args.plot == 1:
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    print(f"screen width: {screen_width}, screen height: {screen_height}")

    window_x = 1200 
    window_y = 80
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{window_x},{window_y}"


inner_loop_break = False

class NatureCNN(nn.Module):
    def __init__(self, in_channels: int = 4, features_dim: int = 512):
        super().__init__()
        
        # 1. Feature extraction layers (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=-1)
        )
        
        # 2. Fully connected projection layer
        # Note: in_features=3136 corresponds to an input frame size of 84x84
        self.linear = nn.Sequential(
            nn.Linear(in_features=64 * 7 * 7, out_features=features_dim, bias=True),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(x))

class DQNModel(nn.Module):

    def __init__(self, input_shape: Tuple[int, ...], n_actions: int):
        super(DQNModel, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions

        self.features_extractor = NatureCNN(in_channels=self.input_shape[0], features_dim=512)

        self.q_net = nn.Sequential(
            nn.Linear(512, self.n_actions),
        )
        
    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    
    def forward(self, x):
        return self.q_net(self.features_extractor.linear(self.features_extractor.cnn(x.float() / 255.0)))

class DQNAgent:

    def __init__(
        self,
        env: gym.Env,
        input_shape: Tuple[int, ...] = (IMAGE_CHANNELS, IMAGE_ROWS, IMAGE_COLS),
        device: str = "auto",
        learning_rate: float = LEARNING_RATE,
        seed: Optional[int] = None,
        exploartion_initial_epsilon: float = EXPLORATION_INITIAL_EPSILON,
        exploartion_final_epsilon: float = EXPLORATION_FINAL_EPSILON,
        exploartion_fraction: float = EXPLORATION_FRACTION,
        model_save_file: Optional[str] = None,
        replay_buffer_file: Optional[str] = None
    ):

        self.env = env
        # Environment info
        self.obs_shape = env.observation_space.shape
        self.n_actions = env.action_space.n
        self.action_space = env.action_space
        
        self.network_input_shape = input_shape
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.learning_rate = learning_rate
        
        print(f"DQN Agent Configuration:")
        print(f"  Environment: {env.spec.id if env.spec else 'Unknown'}")
        print(f"  Observation shape: {self.obs_shape}")
        print(f"  Network input shape: {self.network_input_shape}")
        print(f"  Action space: {self.n_actions}")
        print(f"  Device: {self.device}")
        
        set_random_seed(seed, using_cuda=self.device.type == torch.device("cuda").type)
        self.action_space.seed(seed)

        # Create SB3 ReplayBuffer
        self.replay_buffer = ReplayBuffer(
            buffer_size=BUFFER_SIZE,                    # Maximum transitions to store
            observation_space=Box(low =0, high= 255, shape=self.network_input_shape, dtype=np.uint8),  # Gym observation space
            action_space=env.action_space,        # Gym action space
            device=self.device,                         # PyTorch device
            n_envs=1,                            # Single environment
            optimize_memory_usage=False,          # Standard memory usage
            handle_timeout_termination=True       # Handle timeouts properly
        )
        print(f"self.replay_buffer: Opitmize Memory Usage: {self.replay_buffer.optimize_memory_usage}")
        print(f"self.replay_buffer: Handle Timeout Termination: {self.replay_buffer.handle_timeout_termination}")
        print(f"self.replay_buffer: Current Size: {self.replay_buffer.size()}")

        # Networks
        self.dQ_network = DQNModel(self.network_input_shape, self.n_actions).to(self.device)
        self.target_dQ_network = DQNModel(self.network_input_shape, self.n_actions).to(self.device)
        self.target_dQ_network.eval()
        # Copy weights to target network
        self.target_dQ_network.load_state_dict(self.dQ_network.state_dict())
        self.batch_norm_stats = get_parameters_by_name(self.dQ_network, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.target_dQ_network, ["running_"])
        
        # Get the optimizer
        self.optimizer = optim.Adam(self.dQ_network.parameters(), lr=self.learning_rate, eps=1e-8)
        print("Agent optimizer:")
        print(self.optimizer)
        for i, group in enumerate(self.optimizer.param_groups):
            print(f"Group {i}")
            for k, v in group.items():
                if k != 'params':
                    print(f"    {k}: {v}")
        self._current_progress_remaining = 1.0
        self.exploration_rate = 0.0
        self.exploration_schedule = get_linear_fn(exploartion_initial_epsilon, exploartion_final_epsilon, exploartion_fraction)
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if args.forpaper == 1:
            # save file for analysis
            np.save("original_file.npy", image)
            # need to remove in the experiment
            if args.display == 1:
                cv2.imshow('Image', image)
                # cv2.moveWindow('Image', 0, 800)
                cv2.waitKey(1)
        
        if args.crop == 1:
            # image = image[105:425, 205:450]
            cv2.imshow('Original Image', image)
            cv2.moveWindow('Original Image', 0, 0)
            image = image[REAL_WORLD_INPUT_HEIGHT_TOP:REAL_WORLD_INPUT_HEIGHT_BOTTOM, REAL_WORLD_INPUT_WIDTH_LEFT:REAL_WORLD_INPUT_WIDTH_RIGHT]

            if args.forpaper == 1:
                # save file for analysis
                np.save("cropped_file.npy", image)
                raise Exception("File saved for analysis, stop the code here")
                # need to remove in the experiment

        if args.display == 1 and args.forpaper == 0:
            cv2.imshow('Image', image)
            cv2.moveWindow('Image', 800, 0)
            cv2.waitKey(1)

        # the code is copied from 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (self.network_input_shape[1], self.network_input_shape[2]), interpolation=cv2.INTER_AREA)

        if args.forpaper == 1:
            # save file for analysis
            np.save("preprocessed_file.npy", image)
                
        return image

    def get_action(self, x: torch.Tensor) -> int:
        self.dQ_network.train(False)
        with torch.no_grad():
            q_values = self.dQ_network(x)
            action = torch.argmax(q_values, dim=1)  # Direct argmax of Q-values
        return action.cpu().detach().numpy().item()

    def get_target_action(self, x: torch.Tensor) -> int:
        with torch.no_grad():
            q_values = self.target_dQ_network(x)
            action = torch.argmax(q_values, dim=1)  # Direct argmax of Q-values
        return action.cpu().detach().numpy().item()

    def get_targt_value(self, x: torch.Tensor) -> int:
        with torch.no_grad():
            value = self.target_dQ_network(x)
        return value

    def _update_current_progress_remaining(self, total_steps: int, max_timesteps: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(total_steps) / float(max_timesteps) if 1.0 - float(total_steps) / float(max_timesteps) > 0.0 else 0.0

    def update_exploration_rate(self, 
                                total_steps: int=0):
        
        progress_remaining = 1.0 - float(total_steps / MAX_TEST_STEPS) if 1.0 - float(total_steps / MAX_TEST_STEPS) > 0.0 else 0.0
        self.exploration_rate = self.exploration_schedule(progress_remaining)
                
    def get_action_for_training(self,
                                total_steps: int=0,
                                state: torch.Tensor = None,
                                deterministic: bool = False
                                ) -> Tuple[int]:
        """
        Get an action for training based on epsilon-greedy policy.
        """
        if total_steps < LEARNING_STARTS:
            action = self.env.action_space.sample()  # Random action before learning starts
        else:
            if not deterministic and np.random.rand() < self.exploration_rate:
                action = self.env.action_space.sample()  # Random action
            else:
                action = self.get_action(state)
        return action
        
    #function to decrease the learning rate after every epoch. In this manner, the learning rate reaches 0, by 20,000 epochs
    def step_decay(self, epoch: int) -> float:
        decay = 3.2e-8
        lrate = self.learning_rate - epoch * decay
        lrate = max(lrate, 1e-8)
        return lrate

    def training_step(
        self,
        total_steps: int = 0,
        batch_size: int = BATCH_SIZE,
        gamma: float = GAMMA,
        learning_starts: int = LEARNING_STARTS,
        episode: int = 0,
        max_grad_norm: float = MAX_GRAD_NORM
        ) -> float:
        """
        Perform a training step on the agent.
        """
        
        # Not enough samples in the replay buffer
        if self.replay_buffer.size() < learning_starts:
            return  0

        self.dQ_network.train(True)
        
        # sampled data are pytorch tensors on the device
        replay_batch = self.replay_buffer.sample(batch_size)
        
        state_minibatch = replay_batch.observations
        next_state_minibatch = replay_batch.next_observations
        action_minibatch = replay_batch.actions
        reward_minibatch = replay_batch.rewards
        done_minibatch = replay_batch.dones
        
        with torch.no_grad():
            next_q_values = self.target_dQ_network(next_state_minibatch)
            next_q_values, _ = next_q_values.max(dim=1)
            next_q_values = next_q_values.reshape(-1, 1)
            
            target_q_values = reward_minibatch + (1 - done_minibatch) * gamma * next_q_values

        current_q_values = self.dQ_network(state_minibatch)
        current_q_values = torch.gather(current_q_values, dim=1, index=action_minibatch.long())
        
        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        # zero the gradient buffers
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dQ_network.parameters(), max_grad_norm)
        # Does the update
        self.optimizer.step()

        # Return the loss value for logging
        return loss.item()
        
    def update_target_network(self, 
                              target_update_freq: int = TARGET_UPDATE_INTERVAL,
                              total_steps: int=0):
        # Update target network
        if total_steps > 0 and total_steps % target_update_freq == 0:
            polyak_update(self.dQ_network.parameters(), self.target_dQ_network.parameters(), tau=1.0)
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
            print(f"***** Target network updated at total steps {total_steps}")
        
def configure_serial(port, baudrate):
    """
    configure serial port and open the port
    
    parameters:
    port -- serial port number
    baudrate
    return:
    ser -- configured serial port
    """
    ser = serial.Serial()
    ser.port = port  # serial port
    ser.baudrate = baudrate  # baud rate
    ser.bytesize = serial.EIGHTBITS  # digital byte size
    ser.parity = serial.PARITY_NONE  # parity bit
    ser.stopbits = serial.STOPBITS_ONE  # stop bit

    # open serial port
    ser.open()
    if ser.isOpen():
        print("The serial port is enabled.")
    else:
        print("The serial port can't be enabled, please check the configuration of serial port!")
        return None

    return ser

# Physical keyboard clicker
def send_specific_data(ser):
    """
    send hex number to serial port。
    
    parameters:
    ser -- serial port
    """
    if ser and ser.isOpen():
        # send the first group of data
        data1 = [0x55, 0x01, 0x00, 0x00, 0x01]  # the first group of data
        ser.write(bytes(data1))  # send data

        # wait for 80ms
        time.sleep(0.05)

        # send the second group of data
        data2 = [0x55, 0x00, 0x00, 0x00, 0x00]  # the second group of data
        ser.write(bytes(data2))  # send data
        # wait for 50ms
        time.sleep(0.05)
    else:
        print("The serial port is disabled, the motor to press the key is not working!")

def send_release_data(ser):
    """
    send specific data to the serial port
    
    parameter:
    ser -- serial port
    """
    if ser and ser.isOpen():

        # wait for 50ms
        time.sleep(0.05)

        # send the data
        data2 = [0x55, 0x00, 0x00, 0x00, 0x00]  
        ser.write(bytes(data2))  
        print("The command to release the key: ", data2)
        # wait for 50ms
        time.sleep(0.05)

    else:
        print("The serial port is disabled, the actuator is not working!")

def press_release_key(actuator_event, ser, stop_event):
    while not stop_event.is_set():
        actuator_event.wait() # Wait for the event to be set
        if actuator_event.is_set():
            try:
                send_specific_data(ser)
                # print("press and release")
            except BaseException as e:
                send_release_data(ser)
                print("release")
                actuator_event.clear()
            actuator_event.clear()

def press_key_pygame(ser):
    try:
        send_specific_data(ser)
        # print("press and release")
    except BaseException as e:
        send_release_data(ser)
        print("release")
# End of Physical keyboard clicker

# Hardware emulated keyboard
def receive_data(ser):
    if ser.in_waiting > 0:
        response = ser.readline().decode().strip()
        return response
    return None

def send_data(ser, message):
    try:
        # make sure we're in Command mode
        # send ctrl-Q, then 1
        # cmd=bytes([17])
        ser.write(message.encode())
        while True:
            if ser.in_waiting > 0:
                response = receive_data(ser)
                if response:
                    print(f"Received: {response}")
                else:
                    print("No data received.")
            else:
                break
    except serial.SerialException as e:
        print(f"Error: {e}")    

def send_up_command(ser):
    try:
        message = "Send {up}\n\r"
        ser.write(message.encode())
        while True:
            if ser.in_waiting > 0:
                response = receive_data(ser)
                if response:
                    pass
                else:
                    pass
            else:
                break
    except serial.SerialException as e:
        print(f"Error: {e}") 

def send_ser_command(ser, env_id, action):
    
    if action == 0:
        return  # No action for 0, just return

    command = commands_dict.get((env_id, action))
    
    try:
        message = 'Send ' + command + '\n'
        ser.write(message.encode())
        while True:
            if ser.in_waiting > 0:
                response = receive_data(ser)
                if response:
                    pass
                else:
                    pass
            else:
                break
    except serial.SerialException as e:
        print(f"Error: {e}")

# End of Hardware emulated keyboard functions

stop_kbthread = False

class KeyboardThread(threading.Thread):

    def __init__(self, port, baudrate, env_id):
        super().__init__(daemon=True)
        self.action = 0
        self.current_action = None
        self.press_key = False
        self.key_released = True
        self.pressed_time = None
        self.env_id = env_id
        self.steps = 0
        self.ser = configure_serial(port, baudrate)
        message = '\x11'
        send_data(self.ser, message)
        time.sleep(0.1)
        message = "1"
        send_data(self.ser, message)
        time.sleep(0.1)
        print("end of sending data")
        self.ser.flush()
        
    def run(self):

        global stop_kbthread
        global start_time

        try:
            while not stop_kbthread:
                if self.key_released and self.press_key:
                    self.current_action = self.action
                    self.pressed_time = time.time()
                    # print(f"Sending action: {self.current_action}")
                    send_ser_command(self.ser, self.env_id, self.current_action)
                    self.press_key = False
                    self.key_released = False
                if self.current_action == 0 and not self.key_released and not self.press_key:
                    # For action 0, we don't wait for a release signal, just mark it as released
                    time.sleep(SLIGHTLY_MORE_THAN_KEY_HOLD_TIME)  # Wait for the hold time
                    self.key_released = True
                    self.steps += 1
                    # print(f"Action {self.current_action} sent and marked as released.")
                if self.ser.in_waiting > 0:
                    response = receive_data(self.ser)
                    # if response:
                    #     print(f"Received: {response}")
                    if response == 'Released':
                        self.key_released = True
                        self.steps += 1
                        # print(f"Action {self.current_action} from key pressed to key released: {time.time() - self.pressed_time:.5f} seconds")
                        # print(f"Next action: {self.action}")
        except serial.SerialException as e:
            print(f"Error: {e}")
        finally:
            if self.ser and self.ser.isOpen():
                self.ser.close()
                print("KeyboardThread run: Serial port is closed.")

    def stop(self):
        global stop_kbthread
        stop_kbthread = True
        if self.ser and self.ser.isOpen():
            self.ser.close()
            print("KeyboardThread stop: Serial port is closed.")

dictionary = {}
dictionary["score"] = 0

# Create a queue for communication
data_queue = queue.Queue()
stop_thread = False
start_play = False

port = '/dev/ttyUSB1'  # serial port number based on the system setting
baudrate = 115200  # baud rate for serial communication

class SerialThread(threading.Thread):

    def __init__(self, queue, port, baudrate):
        super().__init__(daemon=True)
        self.queue = queue
        self.receiver = configure_serial(port, baudrate)
        
    def run(self):

        global stop_thread
        global start_play

        if start_play == False:
            print("waiting handshake in thread...\n\r")
            while True:
                if self.receiver.in_waiting > 0:
                    signal = self.receiver.read(self.receiver.in_waiting)
                    print(f"Received signal: {signal}")
                    if signal == b'READY\n':
                        print("Handshake signal received.")
                        self.receiver.write(b'ACK\n')
                        start_play = True
                        break

        self.receiver.flush()
        try:
            while not stop_thread:
                if self.receiver.in_waiting > 0:
                    data = self.receiver.read_until(b'}')
                    self.queue.put(data.decode('utf-8'))  # Push data to the queue
                    self.receiver.flush()
        except serial.SerialException as e:
            print(f"Error: {e}")
        finally:
            if self.receiver and self.receiver.isOpen():
                self.receiver.close()
                print("SerialThread run: Serial port is closed.")
                
    def stop(self):
        global stop_thread
        stop_thread = True
        if self.receiver and self.receiver.isOpen():
            self.receiver.close()
            print("SerialThread stop: Serial port is closed.")

# Start the receiver thread
thread = SerialThread(queue=data_queue, port=port, baudrate=baudrate)
thread.start()

def process_serial_data():
    global dictionary
    global stop_thread
    global inner_loop_break

    # Retrieve data from the queue
    received_data = data_queue.get()
    if args.debug == 1:
         print(f"Received data from serial: {received_data}")
    if '{STOP}' in received_data:
        time.sleep(1)
        stop_thread = True
        thread.join()
        thread.stop()
        inner_loop_break = True
        return None, None, None, None, None
    dictionary = json.loads(received_data)
    terminated = dictionary['terminated']
    truncated = dictionary['truncated']
    lives = lives_after = dictionary['lives']
    # Reward R_t+1
    reward = dictionary['reward']
    return terminated, truncated, lives, lives_after, reward

frame_queue = queue.Queue(maxsize=1)

class CameraThread(threading.Thread):
    def __init__(self, queue):
        super().__init__(daemon=True)
        print("open camera")
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Open the default camera
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Set the codec
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)  # Set the width
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)  # Set the height
        self.cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)  # Set the FPS
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Set the buffer size to 1 to reduce latency
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            quit()
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera opened with FPS: {actual_fps}")
        _, self.frame = self.cap.read()
        if self.frame.ndim == 2:
            print(f"Camera frame shape: {self.frame.shape}")
            self.obs_format = (VIDEO_WIDTH, VIDEO_HEIGHT)
        if self.frame.ndim == 3 and self.frame.shape[2] == 1: 
            print(f"Camera frame shape: {self.frame.shape}")
            self.obs_format = (VIDEO_WIDTH, VIDEO_HEIGHT)
        if self.frame.ndim == 3 and self.frame.shape[2] == 3:
            print(f"Camera frame shape: {self.frame.shape}")
            self.obs_format = (VIDEO_HEIGHT, VIDEO_WIDTH, 3)
        print(f"obs_format: {self.obs_format}")
        self.running = True
        self.queue = queue

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass

            # Put newest frame
            self.queue.put(frame)

    def stop(self):
        self.running = False
        self.cap.release()

cam = CameraThread(frame_queue)
cam.start()
        
def main():

    plt.ion()  # Turn on interactive mode
    fig1, ax1 = plt.subplots()
    line1, = ax1.plot([], [], 'o-', label='median') 
    line2, = ax1.plot([], [], 'x-', label='mean')

    fig3, ax3 = plt.subplots()
    line31, = ax3.plot([], [], 'o-', label='median') 
    line32, = ax3.plot([], [], 'x-', label='mean')

    ax1.legend() 
    ax1.grid(True)

    ax3.legend()
    ax3.grid(True)
    
    plt.figure(fig1.number)
    plt.figure(fig3.number)
    if args.plot == 1:
        plt.show(block=False)
    x_data = []
    y1_data = []
    y2_data = []

    x3_data = []
    y31_data = []
    y32_data = []

    device = torch.device("cuda" if (torch.cuda.is_available() and args.cuda) else "cpu")
    if args.cuda == False:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Handle seed: if None or negative, generate random seed
    if args.seed is None or args.seed < 0:
        # Seed but with a random one
        if args.seed is not None:
            print(f"Negative seed provided: {args.seed}, generating random seed")
        else:
            print(f"No seed provided, generating random seed")
        args.seed = np.random.randint(2**32 - 1, dtype="int64").item()  # type: ignore[attr-defined]
        print(f"Using random seed: {args.seed}")

    set_random_seed(args.seed)
    
    global stop_thread
    
    global stop_kbthread

    # serial port number based on the system setting
    port = '/dev/ttyUSB0'
    # baud rate for hardware emulated keyboard
    baudrate = 57600
    env_id = get_env_id(args.gym_id)

    # Start keyboard thread
    hwemulatedkbd = KeyboardThread(port=port, baudrate=baudrate, env_id=env_id)
    hwemulatedkbd.start()
    
    total_steps = 0
    topscore = 0
    skip = FRAMES_SKIP
    print(f"skip: {skip}")
    a_t = 0
    episode_reset_action = 0
    score = 0
    reward = 0.0
    frames_num = 0
    hns_scores = np.array([])
    scores = np.array([])
    steps = 0
    terminated = True
    truncated = False
    episodes = 0
    frames_num = 0
    fps_time1 = time.time()
    info = {}
    lives = 0
    lives_after = 0

    print("Using CNN model")
    env = gym.make(args.gym_id, render_mode="rgb_array")
    env_id = get_env_id(args.gym_id)
    input_shape = (IMAGE_CHANNELS, IMAGE_ROWS, IMAGE_COLS)

    obs_format = cam.obs_format
        
    frame = np.zeros(obs_format, dtype=env.observation_space.dtype)
    o_t = np.zeros((IMAGE_ROWS, IMAGE_COLS), dtype=env.observation_space.dtype)
    stacked_o_t = np.zeros((1, STACK_FRAMES, IMAGE_ROWS, IMAGE_COLS), dtype=env.observation_space.dtype)
    obs = np.zeros(obs_format, dtype=env.observation_space.dtype)
    obs_buffer = np.zeros((2, *obs_format), dtype=env.observation_space.dtype)

    def observe():
        nonlocal obs
        nonlocal frame
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            print("No frame available within 1s ...")
        obs = frame.copy()

    def wait_environment_ready():
        nonlocal score
        nonlocal steps
        nonlocal reward
        nonlocal frames_num
        nonlocal lives
        nonlocal lives_after
        nonlocal terminated
        nonlocal truncated
        nonlocal obs
        nonlocal frame

        if truncated or terminated:
            while terminated or truncated:
                if not data_queue.empty():
                    # print(f"Real Env reset: Lives: {lives}, no action")
                    terminated, truncated, _, lives_after, reward = process_serial_data()
            observe()
            score = 0
            steps = 0
            frames_num = 0
    
    def noops_reset():

        nonlocal score
        nonlocal steps
        nonlocal frames_num
        nonlocal reward
        nonlocal lives
        nonlocal lives_after
        nonlocal terminated
        nonlocal truncated
        nonlocal obs
        nonlocal frame
        nonlocal obs_buffer

        noops_time = time.time()
        score = 0
        steps = 0
        frames_num = 0
        reward = 0.0
        noops = env.unwrapped.np_random.integers(1, NOOP_MAX + 1)
        assert noops > 0, "noops should be > 0"
        i = 0
        key_pressed = False
        # NO_OP_TIME = float(noops/skip) * SLIGHTLY_MORE_THAN_KEY_HOLD_TIME
        # NO_OP_TIME = float(noops) * (FRAME_DURATION_TIME/EYES_PERCEPTION_TIME)
        NO_OPS = int(float(noops) * (FRAME_DURATION_TIME/EYES_PERCEPTION_TIME))
        # print(f"noops: {noops}, NO_OPS: {NO_OPS}, stop_kbthread: {stop_kbthread}, stop_thread: {stop_thread}")
        while not hwemulatedkbd.key_released:
            pass
        hwemulatedkbd.action = 0
        if hwemulatedkbd.key_released:
            hwemulatedkbd.press_key = True        
        # while (time.time() - noops_time) < NO_OP_TIME:
        while i < NO_OPS:
            time.sleep(EYES_PERCEPTION_TIME * OFFSET)
            stroke_time = time.time()
            observe()
            if not data_queue.empty():
                terminated, truncated, _, lives_after, reward = process_serial_data()
            score += reward
            if truncated or terminated:
                break
            while (time.time() - stroke_time) < (EYES_PERCEPTION_TIME) * (1-OFFSET):
                pass
            frames_num += 1
            i += 1
        wait_environment_ready()
        while not hwemulatedkbd.key_released:
            pass
        steps = hwemulatedkbd.steps

    def action_not_in_the_loop():
        nonlocal obs
        nonlocal episode_reset_action
        nonlocal score
        nonlocal steps
        nonlocal frame
        nonlocal frames_num
        nonlocal reward
        nonlocal lives
        nonlocal lives_after
        nonlocal terminated
        nonlocal truncated
        
        # send_ser_command(ser, env_id, episode_reset_action)
        while not hwemulatedkbd.key_released:
            pass
        hwemulatedkbd.action = episode_reset_action
        hwemulatedkbd.press_key = True
        # key_pressed_time = time.time()
        # EYES_PERCEPTION_TIME is always faster than the key hold time, we use this time to ensure the key is pressed
        time.sleep((EYES_PERCEPTION_TIME) * OFFSET)
        # time.sleep((SLIGHTLY_MORE_THAN_KEY_HOLD_TIME/skip) * OFFSET)
        stroke_time = time.time()
        observe()
        reward = 0.0
        if not data_queue.empty():
            terminated, truncated, _, lives_after, reward = process_serial_data()
        score += reward
        frames_num += 1
        obs_buffer[:-1] = obs_buffer[1:]
        obs_buffer[-1] = obs
        while (time.time() - stroke_time) < (EYES_PERCEPTION_TIME) * (1-OFFSET):
            pass
        steps = hwemulatedkbd.steps

        while not hwemulatedkbd.key_released:
            # half of the key hold time for fire action, divided by skip to spread out the frames               
            time.sleep((EYES_PERCEPTION_TIME) * OFFSET)
            # time.sleep((SLIGHTLY_MORE_THAN_KEY_HOLD_TIME/skip) * OFFSET)
            stroke_time = time.time()
            observe()
            reward = 0.0
            if not data_queue.empty():
                terminated, truncated, _, lives_after, reward = process_serial_data()
            score += reward
            frames_num += 1
            obs_buffer[:-1] = obs_buffer[1:]
            obs_buffer[-1] = obs
            if terminated or truncated:
                break
            while (time.time() - stroke_time) < (EYES_PERCEPTION_TIME) * (1-OFFSET):
                pass
        steps = hwemulatedkbd.steps
        obs = obs_buffer.max(axis=0)
        
    def reset_action():

        nonlocal score
        nonlocal steps
        nonlocal frames_num
        nonlocal reward
        nonlocal lives
        nonlocal lives_after
        nonlocal terminated
        nonlocal truncated
        nonlocal episode_reset_action
        nonlocal obs_buffer

        action_not_in_the_loop()
        
        livesm1 = False
        if 0 < lives_after < lives:
            livesm1 = True
            if terminated or truncated:
                livesm1 = False
        lives = lives_after
        # human player won't wait for the key released, so the below wait time is removed
        # while (time.time() - skip_time) < SLIGHTLY_MORE_THAN_KEY_HOLD_TIME:
        #     pass            
        if livesm1:
            episode_reset_action = 0
            action_not_in_the_loop()
        if terminated or truncated:
            noops_reset()

    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.moveWindow('Image', 0, 600)

    global start_play
    print("waiting for handshake signal\n\r")
    while True:
        if start_play == False:
            pass
        else:
            print("start to play game")
            time.sleep(2)
            break

    file_num = 0

    if args.model == 1:
        env_name: EnvironmentName = args.env
        algo = args.algo
        folder = args.folder

        try:
            _, model_path, log_path = get_model_path(
                args.exp_id,
                folder,
                algo,
                env_name,
                args.load_best,
                args.load_checkpoint,
                args.load_last_checkpoint,
            )
            print(f"***** model_path: {model_path}, log_path: {log_path}")
        except (AssertionError, ValueError) as e:
            # Special case for rl-trained agents
            # auto-download from the hub
            if "rl-trained-agents" not in folder:
                raise e
            else:
                print("Pretrained model not found, trying to download it from sb3 Huggingface hub: https://huggingface.co/sb3")
                # Auto-download
                download_from_hub(
                    algo=algo,
                    env_name=env_name,
                    exp_id=args.exp_id,
                    folder=folder,
                    organization="sb3",
                    repo_name=None,
                    force=False,
                )
                # Try again
                _, model_path, log_path = get_model_path(
                    args.exp_id,
                    folder,
                    algo,
                    env_name,
                    args.load_best,
                    args.load_checkpoint,
                    args.load_last_checkpoint,
                )

        # Off-policy algorithm only support one env for now
        off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]
        
        stats_path = os.path.join(log_path, env_name)
        print(f"stats_path: {stats_path}")
        hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)
        print(f"***** hyperparams: {hyperparams}")
        
        args_path = os.path.join(log_path, env_name, "args.yml")
        if os.path.isfile(args_path):
            with open(args_path) as f:
                loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
        print(f"***** loaded_args: {loaded_args} ")
        
        kwargs = dict(seed=args.seed)
        if algo in off_policy_algos:
            # Dummy buffer size as we don't need memory to enjoy the trained agent
            kwargs.update(dict(buffer_size=1))
            # Hack due to breaking change in v1.6
            # handle_timeout_termination cannot be at the same time
            # with optimize_memory_usage
            if "optimize_memory_usage" in hyperparams:
                kwargs.update(optimize_memory_usage=False)
                
        print(f"***** kwargs: {kwargs}")
        
        # Check if we are running python 3.8+
        # we need to patch saved model under python 3.6/3.7 to load them
        newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

        custom_objects = {}
        if newer_python_version or args.custom_objects:
            custom_objects = {
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.0,
                # load models with different obs bounds
                # Note: doesn't work with channel last envs
                # "observation_space": env.observation_space,
            }
            
        print(f"***** custom_objects: {custom_objects}")

        print(f"***** Loading the model with the following kwargs: {kwargs}")
        model = ALGOS[algo].load(model_path, custom_objects=custom_objects, device=args.device, **kwargs)
        
        files = [model_path, model_path]
        label = "_huggingface"
        labels = [label +'_0', label +'_1']
        print(f"files: {files}, labels: {labels}")

    if args.model == 2:
        
        assert args.algo == "dqn", "dqn is the algorithm for the trained model that are being loaded right now"
        
        files = ['../data/saved_models/model_updates_dqn_breakout_0.pth',
                 '../data/saved_models/model_updates_dqn_breakout_1000000.pth',
                 '../data/saved_models/model_updates_dqn_breakout_2000000.pth',
                 '../data/saved_models/model_updates_dqn_breakout_3000000.pth',
                 '../data/saved_models/model_updates_dqn_breakout_4000000.pth',
                 '../data/saved_models/model_updates_dqn_breakout_5000000.pth',
                 '../data/saved_models/model_updates_dqn_breakout_6000000.pth',
                 '../data/saved_models/model_updates_dqn_breakout_7000000.pth',
                 '../data/saved_models/model_updates_dqn_breakout_8000000.pth',
                 '../data/saved_models/model_updates_dqn_breakout_9000000.pth',
                 '../data/saved_models/model_updates_dqn_breakout_10000000.pth']
        
        labels = ['0_000_000', '1_000_000', '2_000_000', '3_000_000', '4_000_000', '5_000_000', '6_000_000', '7_000_000', '8_000_000', '9_000_000', '10_000_000']
       
    if args.model == 3:
        
        assert args.algo == "dqn", "dqn is the algorithm for the trained model that are being loaded right now"
        
        files = ['../data/saved_models/code_061_model_updates_dqn_breakout_0.pth',
                 '../data/saved_models/code_061_model_updates_dqn_breakout_1000000.pth',
                 '../data/saved_models/code_061_model_updates_dqn_breakout_2000000.pth',
                 '../data/saved_models/code_061_model_updates_dqn_breakout_3000000.pth',
                 '../data/saved_models/code_061_model_updates_dqn_breakout_4000000.pth',
                 '../data/saved_models/code_061_model_updates_dqn_breakout_5000000.pth',
                 '../data/saved_models/code_061_model_updates_dqn_breakout_6000000.pth',
                 '../data/saved_models/code_061_model_updates_dqn_breakout_7000000.pth',
                 '../data/saved_models/code_061_model_updates_dqn_breakout_8000000.pth',
                 '../data/saved_models/code_061_model_updates_dqn_breakout_9000000.pth',
                 '../data/saved_models/code_061_model_updates_dqn_breakout_10000000.pth']
        
        labels = ['0_000_000', '1_000_000', '2_000_000', '3_000_000', '4_000_000', '5_000_000', '6_000_000', '7_000_000', '8_000_000', '9_000_000', '10_000_000']

    if args.model == 4:
        
        assert args.algo == "dqn", "dqn is the algorithm for the trained model that are being loaded right now"
        
        files = ['../data/saved_models/code_069_model_updates_dqn_breakout_1000000.pth',
                 '../data/saved_models/code_069_model_updates_dqn_breakout_1000000.pth',
                 '../data/saved_models/code_069_model_updates_dqn_breakout_2000000.pth',
                 '../data/saved_models/code_069_model_updates_dqn_breakout_3000000.pth',
                 '../data/saved_models/code_069_model_updates_dqn_breakout_4000000.pth',
                 '../data/saved_models/code_069_model_updates_dqn_breakout_5000000.pth',
                 '../data/saved_models/code_069_model_updates_dqn_breakout_6000000.pth',
                 '../data/saved_models/code_069_model_updates_dqn_breakout_7000000.pth',
                 '../data/saved_models/code_069_model_updates_dqn_breakout_8000000.pth',
                 '../data/saved_models/code_069_model_updates_dqn_breakout_9000000.pth',
                 '../data/saved_models/code_069_model_updates_dqn_breakout_10000000.pth']
                
        labels = ['0_000_000', '1_000_000', '2_000_000', '3_000_000', '4_000_000', '5_000_000', '6_000_000', '7_000_000', '8_000_000', '9_000_000', '10_000_000']

    for file in files:
        
        print("file name is ", file)

        input_shape = (IMAGE_CHANNELS, IMAGE_ROWS, IMAGE_COLS)
        agent = DQNAgent(env, device=device, input_shape=input_shape, seed=args.seed)

        if args.model != 1:
            print("Using CNN model")
            loaded_state_dict = torch.load(file, map_location=torch.device('cpu'), weights_only=True)
            print(loaded_state_dict.keys())
            agent.dQ_network.load_state_dict(torch.load(file, map_location=torch.device('cpu'), weights_only=True))
            print(f"load file {file}")
            agent.dQ_network.eval()

        episodes = 0
        frames_num = 0
        fps_time1 = time.time()
        info = {}
        lives = 0
        lives_after = 0

        stacked_o_t[...] = 0
        obs[...] = 0
        obs_buffer[...] = 0
        a_t = 0

        print(f"start all episodes for the current model: env.observation_space.shape: {env.observation_space.shape}, dtype: {stacked_o_t.dtype}")

        while episodes < args.n_episodes:

            frames_num = 0
            total_steps = 0
            fps_time1 = time.time()
            info: dict = {}
            lives = 0
            lives_after = 0

            score = 0
            steps = 0
            terminated = False
            truncated = False
            episode_end = False
            psudo_episode_end = False
            
            # print(f"Start of an episodes")

            terminated = True
            truncated = True            
            while terminated or truncated:
                if not data_queue.empty():
                    print(f"Start a new episode: Env reset inside Noop reset: Lives: {lives}, no action")
                    terminated, truncated, lives, lives_after, reward = process_serial_data()

            if args.forpaper == 1:
                # save file for analysis
                start_time = time.time()
                for i in range(1000):
                    observe()
                    o_t = agent.preprocess(obs)
                print(f"time taken to get the first observation after reset: {time.time() - start_time:.2f} seconds")
                print("start to preprocess the first observation after reset")
                o_t = agent.preprocess(obs)
                time.sleep(2)
                    
                raise Exception("stop here for debug")
                # need to remove in the experiment
            
            lives = lives_after
            observe()
            o_t = agent.preprocess(obs)

            noops_reset()
            
            lives = lives_after

            # Reset Action 1
            episode_reset_action = 1
            reset_action()
            
            lives = lives_after

            # Reset Action 2
            episode_reset_action = 2
            reset_action()

            lives = lives_after
            
            o_t = agent.preprocess(obs)
            stacked_o_t[0, -1, :, :] = o_t
            s_t = stacked_o_t.copy()

            # if there is no action decision, then the action is 0, which is no-op
            a_t = 0
            s_t_tensor = torch.as_tensor(s_t, device=device)
            if file_num == 0:
                a_t = env.action_space.sample()
            else:
                if args.model == 1:
                    a_t = model.predict(s_t, deterministic=True)[0][0]
                elif args.model != 1:
                    a_t = agent.get_action(s_t_tensor)

            while not hwemulatedkbd.key_released:
                pass
            steps = hwemulatedkbd.steps

            # press the key after the key is released
            if hwemulatedkbd.key_released:
                total_steps += 1
                hwemulatedkbd.action = a_t
                hwemulatedkbd.press_key = True

            while True:

                # duplicate the atari wrapper MaxAndSkipEnv functionality here
                terminated = False
                truncated = False
                time.sleep((EYES_PERCEPTION_TIME) * OFFSET)
                stroke_time = time.time()
                observe()
                reward = 0.0
                if not data_queue.empty():
                    terminated, truncated, _, lives_after, reward = process_serial_data()
                if inner_loop_break:
                    break        
                score += reward
                frames_num += 1
                obs_buffer[:-1] = obs_buffer[1:]
                obs_buffer[-1] = obs
                if terminated or truncated:
                    episode_end = True
                obs = obs_buffer.max(axis=0)
                o_t = agent.preprocess(obs)

                if 0 < lives_after < lives:
                    psudo_episode_end = True
                    if episode_end:
                        psudo_episode_end = False
                if psudo_episode_end or episode_end:
                    while (time.time() - stroke_time) < (EYES_PERCEPTION_TIME) * (1-OFFSET):
                        pass
                    while not hwemulatedkbd.key_released:
                        pass

                if hwemulatedkbd.key_released and (psudo_episode_end or episode_end):
                    fps_time2 = time.time()
                    time_elapsed = fps_time2 - fps_time1
                    Calculated_FPS = frames_num / time_elapsed if time_elapsed > 0 else float('inf')
                    frames_num = 0
                    fps_time1 = time.time()

                    if psudo_episode_end == True:

                        terminated = False
                        truncated = False

                        episode_reset_action = 0
                        reset_action()

                        if terminated or truncated:
                            episode_end = True

                    # Start of getting the state from the environment
                    if episode_end == True:
                        if score > topscore:
                            topscore = score

                        print(f"***** FPS: {Calculated_FPS:>8.2f} Episodes: {episodes:>5d} total_steps: {total_steps:>8d} score: {score:>5.2f} topscore: {topscore:>5.2f}")

                        scores = np.append(scores, score)
                        hns = get_human_normalized_score(env_id, score)
                        hns_scores = np.append(hns_scores, hns)

                        score = 0
                        steps = 0
                        terminated = False
                        truncated = False
                        frames_num = 0
                        obs[...] = 0
                        obs_buffer[...] = 0
                        o_t[...] = 0
                        stacked_o_t[...] = 0
                        
                        fps_time1 = time.time()

                        episodes += 1
                        print(f"epsiodes increased to {episodes}, the end of the real episode, reset the environment and start a new episode")
                        print(f"break out of the while loop for the episode end")
                        break
                        
                    lives = lives_after
                    
                    # Reset Action 1
                    episode_reset_action = 1
                    reset_action()
                    
                    lives = lives_after
                        
                    # Reset Action 2
                    episode_reset_action = 2
                    reset_action()
                        
                    lives = lives_after

                    o_t = agent.preprocess(obs)
                    stacked_o_t[...] = 0
                    
                    terminated = False
                    truncated = False
                    episode_end = False
                    psudo_episode_end = False
                
                if episode_end == True:
                    print(f"break out of the while loop for the episode end")
                    break

                stacked_o_t = np.roll(stacked_o_t, shift=-1, axis=1)
                stacked_o_t[0, -1, :, :] = o_t
                # End of taking action and getting the reward from the environment   
                
                s_t = stacked_o_t.copy()
                s_t_tensor = torch.as_tensor(s_t, device=device)
                # Start of getting the action from the model or the player
                if file_num == 0:
                    a_t = env.action_space.sample()
                else:
                    if args.model == 1:
                        a_t = model.predict(s_t, deterministic=True)[0][0]
                    elif args.model != 1:
                        a_t = agent.get_action(s_t_tensor)

                while (time.time() - stroke_time) < (EYES_PERCEPTION_TIME) * (1-OFFSET):
                    pass
                # End of action decision and getting the reward from the environment
                # action decision within the perception time

                # press the key after the key is released
                if hwemulatedkbd.key_released:
                    total_steps += 1
                    hwemulatedkbd.action = a_t
                    hwemulatedkbd.press_key = True

                if inner_loop_break:
                    print("breaking out of the while loop of an episode")
                    break

                if psudo_episode_end:
                    info: dict = {}
                    obs[...] = 0

            if inner_loop_break:
                print("breaking out of the loop for episodes")
                break

        if inner_loop_break:
            print("breaking out of the while loop for files")
            break

        file_path = env_id + '-data-' + args.algo + '-model-' + labels[file_num] + '.npz'
        file_path3 = env_id + '-hns-data-' + args.algo + '-model-'+ labels[file_num] +'.npz'
        
        if file_num == 0:
            array_for_dict = scores
            array_for_hns = hns_scores
        else:
            array_for_dict = np.vstack((array_for_dict, scores))
            array_for_hns = np.vstack((array_for_hns, hns_scores))
        
        np.savez(file_path, array=scores)
        print("*"*5 + " Test results were saved to ", file_path)

        # Load the existing data from the .npz file
        loaded_data = np.load(file_path)
        
        # Retrieve the existing array and datetime
        existing_array = loaded_data['array']
        print("Loaded Scores: ", existing_array)
        min_score = existing_array.min()
        max_score = existing_array.max()
        median = np.median(existing_array)                    
        average = np.mean(existing_array)
        
        print("Loaded scores min: " + str(min_score) + " max: " + str(max_score) + " median: " + str(median) + " average: " + str(average))
        x_data.append(file_num)
        y1_data.append(median)
        y2_data.append(average)
        plt.figure(fig1.number)

        line1.set_data(x_data, y1_data)
        line2.set_data(x_data, y2_data)
        ax1.set_xlim(-1, max(x_data) + 1)
        ax1.set_ylim(-1, max(max(y1_data), max(y2_data)) + 10)
        plt.draw()
        if args.plot == 1:
            plt.pause(0.1)
            plt.show()

        np.savez(file_path3, array=hns_scores)
        print("*"*5 + " Test results (HNS) were saved to ", file_path3)

        # Load the existing data from the .npz file
        loaded_data = np.load(file_path3)
        
        # Retrieve the existing array and datetime
        existing_array = loaded_data['array']
        print("Loaded HNS:", existing_array)
        min_score = existing_array.min()
        max_score = existing_array.max()
        median = np.median(existing_array)        
        average = np.mean(existing_array)
        
        print("Loaded HNS min: " + str(min_score) + " max: " + str(max_score) + " median: " + str(median) + " average: " + str(average))
        
        x3_data.append(file_num)
        y31_data.append(median)
        y32_data.append(average)

        plt.figure(fig3.number)
        line31.set_data(x3_data, y31_data)
        line32.set_data(x3_data, y32_data)
        ax3.set_xlim(-1, max(x3_data) + 1)
        ax3.set_ylim(-1, max(max(y31_data), max(y32_data)) + 10)
        plt.draw()
        if args.plot == 1:
            plt.pause(0.1)
            plt.show()

        scores = np.array([])
        hns_scores = np.array([])
        file_num += 1

    # Save the training progress plot after all test steps are completed
    fig1.savefig("training_progress_final.png", dpi=300, bbox_inches='tight')
    fig3.savefig("training_progress_final3.png", dpi=300, bbox_inches='tight')
    print("*"*5 + " Training progress plot saved as 'training_progress_final.png'")

    print(f"steps: {steps}")
    print(f"info: {info}")
    pygame.quit()

    print("end of evalution, start to close the resources")

    time.sleep(0.5)

    print("*"*5 + " Releasing camera resources...")
    cam.stop()
    cv2.destroyAllWindows()
    print("*"*5 + " Camera resources released.")

    env.close()
    stop_kbthread = True
    stop_thread = True
    # thread.stop()
    print("*"*5 + " Thread is closed.")
        
    time.sleep(0.5)

    return

if __name__ == "__main__":
    main()
