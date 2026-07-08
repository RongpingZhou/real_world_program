# xhost +local:docker

# docker run --gpus all -u root -ti --rm -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /dev/snd:/dev/snd:rw -v /dev/ttyUSB0:/dev/ttyUSB0:rw -v /dev/video0:/dev/video0:rw -v $(realpath ~/mygit/):/rl/ -e DISPLAY=unix$DISPLAY -p 8888:8888 --privileged zrongping/ubuntu2204_cuda12-4-1_cudnn9-1-0-70-1_drl-pytorch_noah-vega:version.20250608

import time
import numpy as np
from typing import Callable, List, Optional, Tuple

import threading
import queue
import yaml

import sys
sys.path.append("domain/")
sys.path.append("mybuffer/")

from evaluation.atari_data import get_human_normalized_score, get_env_id

from mybuffer.replaybm import ReplayBuffer

import pygame
from pygame import Surface

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

import matplotlib
import matplotlib.pyplot as plt

from huggingface_sb3 import EnvironmentName

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
    parser.add_argument("--sensor", type=int, default=0,
        help="use sensor or not, 0 is not using sensor, 1 is using sensor")
    parser.add_argument("--play", type=int, default=0,
        help="0 is not playing, 1 is playing")
    parser.add_argument("--training", type=int, default=0,
        help="0 is not training, 1 is training, 2 is transfer training")
    parser.add_argument("--test", type=int, default=0,
        help="0 is not testing, 1 is testing")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--num-steps", type=int, default=5,
        help="how many steps to run in each environment per update")
    parser.add_argument("--bptime", type=int, default=0,
        help="use sensor or not, 0 is not showing back propagation time, 1 is showing time")
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
    args = parser.parse_args()
    
    return args

args = parse_args()
print("args: ", args)
print(vars(args))

if args.display == 1:
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    print(f"screen width: {screen_width}, screen height: {screen_height}")

    # for lab computer setting
    window_x = 50 
    window_y = 862
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{window_x},{window_y}"

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

VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_FPS = 120

#95
REAL_WORLD_INPUT_HEIGHT_TOP = 90
# REAL_WORLD_INPUT_HEIGHT_TOP = 95
#415
REAL_WORLD_INPUT_HEIGHT_BOTTOM = 410
# REAL_WORLD_INPUT_HEIGHT_BOTTOM = 415
REAL_WORLD_INPUT_WIDTH_LEFT = 190
REAL_WORLD_INPUT_WIDTH_RIGHT = 435

inner_loop_break = False

class MissingKeysToAction(Exception):
    """Raised when the environment does not have a default ``keys_to_action`` mapping."""

# from gymnasium.utils.play import play
# from domain.play import play
class PlayableGame:
    """Wraps an environment allowing keyboard inputs to interact with the environment."""

    def __init__(
        self,
        env: Env,
        keys_to_action: dict[tuple[int, ...], int] | None = None,
        zoom: float | None = None,
    ):
        """Wraps an environment with a dictionary of keyboard buttons to action and if to zoom in on the environment.

        Args:
            env: The environment to play
            keys_to_action: The dictionary of keyboard tuples and action value
            zoom: If to zoom in on the environment render
        """
        if env.render_mode not in {"rgb_array", "rgb_array_list"}:
            raise ValueError(
                "PlayableGame wrapper works only with rgb_array and rgb_array_list render modes, "
                f"but your environment render_mode = {env.render_mode}."
            )

        self.env = env
        self.relevant_keys = self._get_relevant_keys(keys_to_action)
        # self.video_size is the size of the video that is being displayed.
        # The window size may be larger, in that case we will add black bars
        self.video_size = self._get_video_size(zoom)
        self.screen = pygame.display.set_mode(self.video_size, pygame.RESIZABLE)
        self.pressed_keys = []
        self.running = True

    def _get_relevant_keys(
        self, keys_to_action: dict[tuple[int], int] | None = None
    ) -> set:
        if keys_to_action is None:
            if self.env.has_wrapper_attr("get_keys_to_action"):
                keys_to_action = self.env.get_wrapper_attr("get_keys_to_action")()
            else:
                assert self.env.spec is not None
                raise MissingKeysToAction(
                    f"{self.env.spec.id} does not have explicit key to action mapping, "
                    "please specify one manually, `play(env, keys_to_action=...)`"
                )
        assert isinstance(keys_to_action, dict)
        relevant_keys = set(sum((list(k) for k in keys_to_action.keys()), []))
        return relevant_keys

    def _get_video_size(self, zoom: float | None = None) -> tuple[int, int]:
        rendered = self.env.render()
        if isinstance(rendered, List):
            rendered = rendered[-1]
        assert rendered is not None and isinstance(rendered, np.ndarray)
        video_size = (rendered.shape[1], rendered.shape[0])
        print(f"video size: {video_size}")

        if zoom is not None:
            video_size = (int(video_size[0] * zoom), int(video_size[1] * zoom))

        return video_size

def display_arr(
    screen: Surface, arr: np.ndarray, video_size: tuple[int, int], transpose: bool
):
    """Displays a numpy array on screen.

    Args:
        screen: The screen to show the array on
        arr: The array to show
        video_size: The video size of the screen
        transpose: If to transpose the array on the screen
    """
    assert isinstance(arr, np.ndarray) and arr.dtype == np.uint8
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    
    # We might have to add black bars if surface_size is larger than video_size
    surface_size = screen.get_size()
    width_offset = (surface_size[0] - video_size[0]) / 2
    height_offset = (surface_size[1] - video_size[1]) / 2
    screen.fill((0, 0, 0))
    screen.blit(pyg_img, (width_offset, height_offset))

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
        exploartion_fraction: float = EXPLORATION_FRACTION
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
        if args.crop == 1:
            # image = image[95:415, 190:435]
            cropped_image = image[REAL_WORLD_INPUT_HEIGHT_TOP:REAL_WORLD_INPUT_HEIGHT_BOTTOM, REAL_WORLD_INPUT_WIDTH_LEFT:REAL_WORLD_INPUT_WIDTH_RIGHT]
            # np.save("env_obs_file.npy", image)
            # raise
            cv2.imshow('Cropped Image', cropped_image)

        if args.display == 1:
            if args.sensor == 0:
                cv2.imshow('Image', image[:, :, [2, 1, 0]])
            else:
                cv2.imshow('Image', image)
            cv2.moveWindow('Image', 350, 0)
            cv2.waitKey(1)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (self.network_input_shape[1], self.network_input_shape[2]), interpolation=cv2.INTER_AREA)

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
        if not hasattr(self, 'action_call'):
            self.action_call = 0
        self.action_call += 1

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
                
        if self.replay_buffer.size() < learning_starts:
            return  0   # Not enough samples in the replay buffer

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
        self.optimizer.step()    # Does the update

        return loss.item()  # Return the loss value for logging
        
    def update_target_network(self, 
                              target_update_freq: int = TARGET_UPDATE_INTERVAL,
                              total_steps: int=0):
        # Update target network
        if total_steps > 0 and total_steps % target_update_freq == 0:
            polyak_update(self.dQ_network.parameters(), self.target_dQ_network.parameters(), tau=1.0)
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
            print(f"***** Target network updated at total steps {total_steps}")

if args.sensor == 1:
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

                # If queue is full, remove old frame
                if self.queue.full():
                    try:
                        self.queue.get_nowait()
                    except:
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

    if args.seed < 0:
        # Seed but with a random one
        print(f"before args.seed: {args.seed}")
        args.seed = np.random.randint(2**32 - 1, dtype="int64").item()  # type: ignore[attr-defined]
        print(f"after args.seed: {args.seed}")

    set_random_seed(args.seed)
    
    has_fire = False
    env = gym.make(args.gym_id, render_mode="rgb_array")
    
    if args.sensor == 0:
        obs_format = env.observation_space.shape

    if args.sensor == 1:
        obs_format = cam.obs_format
        
    # Define key-to-action mapping
    if args.display == 1:
        keys_to_action = {
            (pygame.K_LEFT,): 3,  # Move left
            (pygame.K_RIGHT,): 2,  # Move right
            (pygame.K_SPACE,): 1,  # Fire (release ball)
        }
    env_id = get_env_id(args.gym_id)
    print(f"env_id: {env_id}")
    if env_id == "breakout":
        env = TimeLimit(env, max_episode_steps=args.max_episode_steps)
    if "FIRE" in env.unwrapped.get_action_meanings():
        has_fire = True
        print("Environment has FIRE action, will use FireResetEnv wrapper")
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    # if args.capture_video:
    #     env = gym.wrappers.RecordVideo(env, "videos", step_trigger=lambda step: step % 1000 == 0)

    print(f"env spec: {env.spec}")
    print(f"env metadata: {env.metadata}")
    env.metadata["render_fps"] = args.fps  # Set FPS to 30
    print(f"env action space: {env.action_space}")
    print(f"env action space shape: {env.action_space.shape}")
    print(f"env observation space: {env.observation_space}")
    print(f"env observation space shape: {env.observation_space.shape}")
    print(f"FPS: {args.fps}")
    print("env.unwrapped: ", env.unwrapped)
    print("env.unwrapped.ale: ", env.unwrapped.ale)
    has_lives = False
    lives = 0
    if hasattr(env.unwrapped, 'ale') and hasattr(env.unwrapped.ale, 'lives'):
        has_lives = True
        print("env.unwrapped.ale.lives: ", env.unwrapped.ale.lives())
        lives = env.unwrapped.ale.lives()

    if args.display == 1:
        key_code_to_action = {}
        for key_combination, action in keys_to_action.items():
            key_code = tuple(
                sorted(ord(key) if isinstance(key, str) else key for key in key_combination)
            )
            key_code_to_action[key_code] = action

        print(f"key_code_to_action: {key_code_to_action}")
    
    if args.display == 1:
        env.reset(seed=args.seed)
        game = PlayableGame(env, key_code_to_action, zoom=args.zoom)
        clock = pygame.time.Clock()
        clock.tick(args.fps)
            
    # Initialize training variables
    # index 0 = no operation
    episode_reset_action = 0
    total_reward = 0.0
    obs = np.zeros(0)
            
    inner_loop_break = False

    EPISODE = 0
    skip = FRAMES_SKIP
    print(f"skip: {skip}")
    a_t = 0
    topscore = 0
    score = 0
    hns_scores = np.array([])
    scores = np.array([])
    steps = 0
    total_steps = 0

    frames_num = 0
    fps_time1 = time.time()
    info: dict = {}
    reset_info: dict = {}
    o_t = np.zeros((IMAGE_ROWS, IMAGE_COLS), dtype=env.observation_space.dtype)
    stacked_o_t = np.zeros((1, STACK_FRAMES, IMAGE_ROWS, IMAGE_COLS), dtype=env.observation_space.dtype)
    print(f"Step {total_steps}: env.observation_space.shape: {env.observation_space.shape}, dtype: {stacked_o_t.dtype}")
    terminal_stacked_o_t = np.zeros((1, STACK_FRAMES, IMAGE_ROWS, IMAGE_COLS), dtype=env.observation_space.dtype)
    if args.sensor == 0:
        obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=env.observation_space.dtype)
    else:
        obs_buffer = np.zeros((2, *obs_format), dtype=env.observation_space.dtype)
        frame = np.zeros(obs_format, dtype=env.observation_space.dtype)

    print(f"*****env spec: {env.spec}")

    # --- Helper functions for init and training loop ---

    def render_and_display():
        nonlocal obs
        if args.display == 1:
            if obs is not None:
                rendered = env.render()
                if isinstance(rendered, List):
                    rendered = rendered[-1]
                assert rendered is not None and isinstance(rendered, np.ndarray)
                display_arr(
                    game.screen, rendered, transpose=True, video_size=game.video_size
                )
            pygame.display.flip()
            clock.tick(args.fps)

    def capture_video():
        nonlocal obs
        nonlocal frame
        if args.sensor == 1:
            try:
                frame = frame_queue.get(timeout=1.0)
            except queue.Empty:
                print("No frame available within 1s ...")
            obs = frame.copy()

    def observe():
        render_and_display()
        capture_video()

    def direct_env_reset():
        nonlocal obs
        nonlocal score
        nonlocal steps
        nonlocal frames_num
        nonlocal terminated
        nonlocal truncated
        nonlocal info

        obs, info = env.reset(seed=None)
        observe()
        # terminated = False
        # truncated = False
        # score = 0
        # steps = 0
        # frames_num = 0

    def noop_reset_action(msg_prefix=""):
        """Run random noops after env has already been reset. Handle mid-noop terminal resets.

        Caller must call env.reset() and observe() before this function.

        """
        nonlocal obs
        nonlocal score
        nonlocal steps
        nonlocal frames_num
        nonlocal terminated
        nonlocal truncated
        nonlocal total_steps
        nonlocal lives
        nonlocal lives_after
        nonlocal info

        score = 0
        steps = 0
        frames_num = 0
        noops = env.unwrapped.np_random.integers(1, NOOP_MAX + 1)
        assert noops > 0, "noops should be > 0"
        print(f"{msg_prefix}Lives: {lives} -> {env.unwrapped.ale.lives()}，Step {total_steps}: No-ops after episode end: {noops}")
        for _ in range(noops):
            obs, reward, terminated, truncated, info = env.step(0)
            observe()
            score += reward
            steps += 1
            frames_num += 1
            if terminated or truncated:
                direct_env_reset()

    def action_not_in_the_loop():
        """
        Execute action with frame skipping and max pooling.

        """
        nonlocal obs
        nonlocal obs_buffer
        nonlocal episode_reset_action
        nonlocal skip
        nonlocal score
        nonlocal steps
        nonlocal frames_num
        nonlocal terminated
        nonlocal truncated
        nonlocal total_reward
        nonlocal info
        total_reward = 0.0
        terminated = False
        truncated = False
        
        for i in range(skip):
            obs, reward, terminated, truncated, info = env.step(episode_reset_action)
            observe()
            score += reward
            steps += 1
            frames_num += 1
            if i == skip - 2:
                obs_buffer[0] = obs
            if i == skip - 1:
                obs_buffer[1] = obs
            total_reward += float(reward)
            if terminated or truncated:
                break
        obs = obs_buffer.max(axis=0)
        
    def reset_action():
        nonlocal obs
        nonlocal score
        nonlocal steps
        nonlocal frames_num
        nonlocal lives
        nonlocal lives_after
        nonlocal has_lives
        nonlocal terminated
        nonlocal truncated
        nonlocal episode_reset_action

        print(f"Reset Action: {episode_reset_action}, lives: {lives} -> {lives_after}")
        
        action_not_in_the_loop()

        lives_after = env.unwrapped.ale.lives()
        livesm1 = False
        if 0 < lives_after < lives and has_lives:
            livesm1 = True
            if terminated or truncated:
                livesm1 = False
        lives = lives_after

        if livesm1:
            episode_reset_action = 0
            action_not_in_the_loop()
        if terminated or truncated:
            obs, _ = env.reset(seed=None)
            observe()
            noop_reset_action(msg_prefix="Terminal reset during noop sequence: ")
        
    file_num = 0

    if args.model == 1:
        env_name: EnvironmentName = args.env
        algo = args.algo
        folder = args.folder
        
        print(f"***** Loading model for env: {env_name}, algo: {algo}, folder: {folder}")

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
        print(f"args_path: {args_path}")
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

        # files = ['saved_models/model_updates_dqn_breakout_0.pth',
        #          'saved_models/model_updates_dqn_breakout_2000000.pth']
        
        files = ['saved_models/model_updates_dqn_breakout_0.pth',
                 'saved_models/model_updates_dqn_breakout_1000000.pth',
                 'saved_models/model_updates_dqn_breakout_2000000.pth',
                 'saved_models/model_updates_dqn_breakout_3000000.pth',
                 'saved_models/model_updates_dqn_breakout_4000000.pth',
                 'saved_models/model_updates_dqn_breakout_5000000.pth',
                 'saved_models/model_updates_dqn_breakout_6000000.pth',
                 'saved_models/model_updates_dqn_breakout_7000000.pth',
                 'saved_models/model_updates_dqn_breakout_8000000.pth',
                 'saved_models/model_updates_dqn_breakout_9000000.pth',
                 'saved_models/model_updates_dqn_breakout_10000000.pth']
        
        # files = ['saved_models/model_updates_dqn_frostbite_1000000.pth',
        #         'saved_models/model_updates_dqn_frostbite_2000000.pth',
        #         'saved_models/model_updates_dqn_frostbite_3000000.pth',
        #         'saved_models/model_updates_dqn_frostbite_4000000.pth',
        #         'saved_models/model_updates_dqn_frostbite_5000000.pth',
        #         'saved_models/model_updates_dqn_frostbite_6000000.pth',
        #         'saved_models/model_updates_dqn_frostbite_7000000.pth',
        #         'saved_models/model_updates_dqn_frostbite_8000000.pth',
        #         'saved_models/model_updates_dqn_frostbite_9000000.pth',
        #         'saved_models/model_updates_dqn_frostbite_10000000.pth']
        
        labels = ['0_000_000', '1_000_000', '2_000_000', '3_000_000', '4_000_000', '5_000_000', '6_000_000', '7_000_000', '8_000_000', '9_000_000', '10_000_000']

    if args.model == 3:
        
        assert args.algo == "dqn", "dqn is the algorithm for the trained model that are being loaded right now"

        # files = ['saved_models/model_updates_dqn_breakout_0.pth',
        #          'saved_models/model_updates_dqn_breakout_2000000.pth']
        
        files = ['saved_models_real_input/model_updates_dqn_breakout_0.pth',
                 'saved_models_real_input/model_updates_dqn_breakout_1000000.pth',
                 'saved_models_real_input/model_updates_dqn_breakout_2000000.pth',
                 'saved_models_real_input/model_updates_dqn_breakout_3000000.pth',
                 'saved_models_real_input/model_updates_dqn_breakout_4000000.pth',
                 'saved_models_real_input/model_updates_dqn_breakout_5000000.pth',
                 'saved_models_real_input/model_updates_dqn_breakout_6000000.pth',
                 'saved_models_real_input/model_updates_dqn_breakout_7000000.pth',
                 'saved_models_real_input/model_updates_dqn_breakout_8000000.pth',
                 'saved_models_real_input/model_updates_dqn_breakout_9000000.pth',
                 'saved_models_real_input/model_updates_dqn_breakout_10000000.pth']
        
        labels = ['0_000_000', '1_000_000', '2_000_000', '3_000_000', '4_000_000', '5_000_000', '6_000_000', '7_000_000', '8_000_000', '9_000_000', '10_000_000']
        
    for file in files:
        
        print("file name is ", file)

        input_shape = (IMAGE_CHANNELS, IMAGE_ROWS, IMAGE_COLS)
        agent = DQNAgent(env, device=device, input_shape=input_shape, seed=args.seed)

        if args.model != 1:
            print("Using CNN model")
            if args.test == 1:
                loaded_state_dict = torch.load(file)
                print(loaded_state_dict.keys())
                agent.dQ_network.load_state_dict(torch.load(file, map_location=device, weights_only=True))
                print(f"load file {file}")
                agent.dQ_network.eval()

        episodes = 0
        frames_num = 0
        fps_time1 = time.time()
        info = {}
        lives = 0
        lives_after = 0
        
        # while episodes < 100:
        while episodes < args.n_episodes:
            
            score = 0
            steps = 0
            done = False
            pdone = False
            terminated = False
            truncated = False
            psudo_episode_end = False
            episode_end = False

            # initial reset
            if episodes == 0:
                obs, info = env.reset(seed=args.seed)
            else:
                obs, info = env.reset(seed=None)
        
            lives_after = lives = env.unwrapped.ale.lives()
        
            # --- Initial noop reset ---
            # FireResetEnv reset start
            # EpisodicLifeEnv reset start
            # NoopResetEnv reset start
            observe()
            obs = np.zeros(0)
            info: dict = {}
            noop_reset_action(msg_prefix="First Env Reset: ")
            # NoopResetEnv reset end

            lives = env.unwrapped.ale.lives()
            # EpisodicLifeEnv reset end

            if has_fire:
                episode_reset_action = 1
                reset_action()
                lives = env.unwrapped.ale.lives()
                
                episode_reset_action = 2
                reset_action()
                lives = env.unwrapped.ale.lives()
            # FireResetEnv reset end

            o_t = agent.preprocess(obs)
            stacked_o_t[0, -1, :, :] = o_t
            s_t = stacked_o_t.copy()
            print(f"s_t, {type(s_t)}, {s_t.shape}")
            
            while True:
                
                s_t_tensor = torch.as_tensor(s_t, device=device)

                # Start of getting the action from the model or the player
                if args.display == 1:
                    
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT: 
                            print("quit")
                            inner_loop_break = True
                            break
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                print("escape")
                                inner_loop_break = True
                                break
                        elif event.type == pygame.WINDOWRESIZED:
                            # Compute the maximum video size that fits into the new window
                            scale_width = event.x / game.video_size[0]
                            scale_height = event.y / game.video_size[1]
                            scale = min(scale_height, scale_width)
                            game.video_size = (scale * game.video_size[0], scale * game.video_size[1])

                if inner_loop_break:
                    break

                if file_num == 0:
                    a_t = agent.env.action_space.sample()
                else:
                    if args.model == 1:
                        a_t = model.predict(s_t, deterministic=True)[0][0]
                    elif args.model == 2:
                        a_t = agent.get_action(s_t_tensor)

                # Start of taking action and getting the reward from the environment
                total_reward = 0.0
                terminated = False
                truncated = False
                # 4 is skipping frames and doing max pooling over the last 2 frames
                for i in range(skip):
                    obs, reward, terminated, truncated, info = env.step(a_t)
                    observe()
                    lives_after = env.unwrapped.ale.lives()

                    score += reward
                    steps += 1
                    frames_num += 1

                    if i == skip - 2:
                        obs_buffer[0] = obs
                    if i == skip - 1:
                        obs_buffer[1] = obs

                    total_reward += float(reward)

                    if terminated or truncated:
                        episode_end = True
                        if terminated:
                            done = True
                        break

                obs = obs_buffer.max(axis=0)
                r_t = np.sign(float(total_reward))
                o_t = agent.preprocess(obs)

                if 0 < lives_after < lives and has_lives:
                    pdone = True
                    psudo_episode_end = True
                    if episode_end:
                        psudo_episode_end = False
                    if done:
                        pdone = False
                    terminated = True

                if lives_after != lives or terminated or truncated:
                    print(f"lives: {lives} -> {lives_after}, steps: {steps}, terminated: {terminated}, truncated: {truncated}, pdone: {pdone}, done: {done}")

                lives = lives_after
                
                total_steps += 1

                if psudo_episode_end or episode_end:
                    terminal_stacked_o_t = np.roll(stacked_o_t, shift=-1, axis=1)
                    terminal_stacked_o_t[0, -1, :, :] = o_t.copy()
                    fps_time2 = time.time()
                    time_elapsed = fps_time2 - fps_time1
                    Calculated_FPS = frames_num / time_elapsed if time_elapsed > 0 else float('inf')
                    frames_num = 0
                    fps_time1 = time.time()
                    
                if psudo_episode_end:
                    
                    # no-op action EpisodicLifeEnv reset
                    print(f"Lives loss: Action: 0, lives: {lives} -> {env.unwrapped.ale.lives()}")
                    obs = np.zeros(0)
                    info: dict = {}
                    episode_reset_action = 0
                    action_not_in_the_loop()
                    lives = lives_after = env.unwrapped.ale.lives()

                    if terminated or truncated:
                        episode_end = True
                    # EpisodicLifeEnv reset end
                        
                    if not episode_end:
                        if has_fire:
                            episode_reset_action = 1
                            reset_action()
                            lives = lives_after = env.unwrapped.ale.lives()

                            if terminated or truncated:
                                episode_end = True
                            
                            if not episode_end:

                                episode_reset_action = 2
                                reset_action()
                                lives = lives_after = env.unwrapped.ale.lives()
                            # FireResetEnv reset end

                    lives = lives_after
                
                    o_t = agent.preprocess(obs)

                    stacked_o_t[...] = 0

                    EPISODE += 1

                if episode_end == True:
                    terminated = False
                    truncated = False

                    if score > topscore:
                        topscore = score

                    stacked_o_t[...] = 0
                    
                    print(f"***** Calculated_FPS: {Calculated_FPS:>8.2f} Episodes: {episodes:>5d} EPISODE: {EPISODE:>5d} total_steps: {total_steps:>8d} replay_buffer.size: {agent.replay_buffer.size():>5d} score: {score:>5.2f} topscore: {topscore:>5.2f}")

                    scores = np.append(scores, score)
                    hns = get_human_normalized_score(env_id, score)
                    hns_scores = np.append(hns_scores, hns)

                    score = 0
                    steps = 0
                    terminated = False
                    truncated = False
                    frames_num = 0
                    fps_time1 = time.time()
                    
                    EPISODE += 1
                    episodes += 1
                    break

                stacked_o_t = np.roll(stacked_o_t, shift=-1, axis=1)
                stacked_o_t[0, -1, :, :] = o_t.copy()
                # End of taking action and getting the reward from the environment        
                
                s_t = stacked_o_t.copy()
                
                total_steps += 1
                
                if psudo_episode_end:
                    terminated = False
                    truncated = False
                    pdone = False
                    psudo_episode_end = False
                    
                if episode_end:                
                    terminated = False
                    truncated = False
                    done = False
                    pdone = False
                    episode_end = False
                    psudo_episode_end = False
                                                            
                if inner_loop_break:
                    print("break 2")
                    break

            if inner_loop_break:
                print("break 3")
                break

        if inner_loop_break:
            print(" break 4")
            break
        
        print(f"file_num: {file_num}, labels: {labels}")
        print(f"label: {labels[file_num]}, scores: {scores}")
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

    print("end of evaluation")

    time.sleep(0.5)

    if args.sensor == 1:
        print("*"*5 + " Releasing camera resources...")
        cam.stop()
        cv2.destroyAllWindows()
        print("*"*5 + " Camera resources released.")
    
    env.close()
    print("*"*5 + " Environment is closed.")

    time.sleep(0.5)

    return

if __name__ == "__main__":
    main()
