# Running with docker container with pytorch 2.5.1 and necessary cuda and cudnn

# xhost +local:docker

# docker run --gpus all -u root -ti --rm -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /dev/snd:/dev/snd:rw -v /dev/ttyUSB0:/dev/ttyUSB0:rw -v /dev/video0:/dev/video0:rw -v $(realpath ~/mygit/):/rl/ -e DISPLAY=unix$DISPLAY -p 8888:8888 --privileged zrongping/ubuntu2204_cuda12-4-1_cudnn9-1-0-70-1_drl-pytorch_noah-vega:version.20250608

import time
import numpy as np
from collections import deque
from typing import List, Optional, Tuple
import random

import threading
import queue

import sys

from wandb import env
sys.path.append("domain/")
sys.path.append("mybuffer/")

from evaluation.atari_data import get_env_id

from mybuffer.replaybm import ReplayBuffer

import pygame
from pygame import Surface

import tkinter as tk

import os

import gymnasium as gym
from gymnasium import Env, logger
from gymnasium.wrappers import TimeLimit
from gymnasium.core import ActType, ObsType
from gymnasium.error import DependencyNotInstalled
from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete

from stable_baselines3.common.utils import get_linear_fn, safe_mean, set_random_seed, polyak_update, get_parameters_by_name
from stable_baselines3.common.logger import configure

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
    
import argparse
from distutils.util import strtobool

import ale_py
gym.register_envs(ale_py)
# gym.pprint_registry()

from datetime import datetime
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--gym-id', type=str, default="BreakoutNoFrameskip-v4",
        help='the id of the gym environment')
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
    parser.add_argument("--max-episode-steps", type=int, default=60000,
        help="how many steps to run in one episode in each environment")
    parser.add_argument("--model", type=int, default=4,
        help="model for the agent, 0 is random action, 1 is CNN, 2 is CNN trained with ppo, 3 is transformer, 4 is the standard CNN model")
    parser.add_argument("--sensor", type=int, default=0,
        help="use sensor or not, 0 is not using sensor, 1 is using sensor")
    parser.add_argument("--training", type=int, default=0,
        help="0 is not training, 1 is training, 2 is transfer training")
    parser.add_argument("--resume", type=int, default=0,
        help="0 is not resuming, 1 is resuming from last checkpoint, 2 is resuming from specific checkpoint-file")
    parser.add_argument("--test", type=int, default=0,
        help="0 is not testing, 1 is testing")
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

SB3_FIRE_RESET = True

inner_loop_break = False

class MissingKeysToAction(Exception):
    """Raised when the environment does not have a default ``keys_to_action`` mapping."""

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
        
        self.rnd_calls = 0

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
            image = image[95:415, 190:435]
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
            self.rnd_calls += 1
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

def save_checkpoint(
    filepath: str,
    agent: 'DQNAgent',
    total_steps: int,
    episode: int,
    topscore: float,
    tested_steps: list,
    ep_info_buffer: deque,
    env_id: str,
    args,
    env=None
):
    """
    Save a comprehensive training checkpoint that includes all state needed for resumption.

    Args:
        filepath: Path where checkpoint will be saved
        agent: The DQN agent instance
        total_steps: Current total training steps
        episode: Current episode number
        topscore: Best score achieved so far
        tested_steps: Array tracking which test milestones have been completed
        ep_info_buffer: Deque of recent episode statistics
        env_id: Environment identifier
        args: Command line arguments
    """
    checkpoint = {
        # Model state
        'model_state_dict': agent.dQ_network.state_dict(),
        'target_model_state_dict': agent.target_dQ_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),

        # Training state
        'total_steps': total_steps,
        'episode': episode,
        'topscore': topscore,
        'exploration_rate': agent.exploration_rate,
        'tested_steps': tested_steps,

        # Episode info buffer (convert deque to list for serialization)
        'ep_info_buffer': list(ep_info_buffer),

        # Random states for reproducibility
        # Note: RNG states are always saved on CPU for portability
        'rng_state': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state().cpu(),  # Explicitly ensure CPU
            'torch_cuda': [state.cpu() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
            # CRITICAL: Save environment's RNG state for exact reproducibility
            'env_np_random': env.unwrapped.np_random.bit_generator.state if env is not None else None,
            # CRITICAL: Save action_space RNG state (separate from env RNG!)
            'action_space_np_random': env.action_space.np_random.bit_generator.state if env is not None and hasattr(env.action_space, 'np_random') else None,
            # CRITICAL: Save observation_space RNG state (also separate!)
            'observation_space_np_random': env.observation_space.np_random.bit_generator.state if env is not None and hasattr(env.observation_space, 'np_random') else None,
        },

        # Metadata
        'env_id': env_id,
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),

        # Hyperparameters (for verification)
        'hyperparameters': {
            'learning_rate': LEARNING_RATE,
            'buffer_size': BUFFER_SIZE,
            'gamma': GAMMA,
            'batch_size': BATCH_SIZE,
            'target_update_interval': TARGET_UPDATE_INTERVAL,
            'exploration_fraction': EXPLORATION_FRACTION,
            'exploration_initial': EXPLORATION_INITIAL_EPSILON,
            'exploration_final': EXPLORATION_FINAL_EPSILON,
        }
    }

    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")
    print(f"  Episode: {episode}, Steps: {total_steps}, Top Score: {topscore}")

def load_checkpoint(
    filepath: str,
    agent: 'DQNAgent',
    device: torch.device
) -> dict:
    """
    Load a training checkpoint and restore all state.

    Args:
        filepath: Path to the checkpoint file
        agent: The DQN agent instance to load state into
        device: PyTorch device

    Returns:
        Dictionary containing restored training state
    """
    print(f"Loading checkpoint from {filepath}")
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    # BACKWARD COMPATIBILITY: Detect old-style vs new-style checkpoint
    if 'model_state_dict' not in checkpoint:
        print("⚠ Loading OLD-STYLE checkpoint (model weights only)")
        print("  This checkpoint doesn't have full training state.")
        print("  Loading model weights, but other state will be initialized fresh.")

        # Old-style checkpoint is just the model state_dict
        agent.dQ_network.load_state_dict(checkpoint)
        agent.target_dQ_network.load_state_dict(agent.dQ_network.state_dict())

        # Return minimal checkpoint data structure
        return {
            'episode': 0,
            'total_steps': 0,
            'topscore': 0,
            'exploration_rate': agent.exploration_rate,
            'tested_steps': None,
            'ep_info_buffer': [],
            'rng_state': None,
            'args': {},
        }

    # NEW-STYLE CHECKPOINT: Full training state
    print("✓ Loading NEW-STYLE checkpoint (full training state)")

    # Restore model state
    agent.dQ_network.load_state_dict(checkpoint['model_state_dict'])
    agent.target_dQ_network.load_state_dict(checkpoint['target_model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.exploration_rate = checkpoint['exploration_rate']

    # Restore random states for reproducibility
    if 'rng_state' in checkpoint:
        random.setstate(checkpoint['rng_state']['python'])
        np.random.set_state(checkpoint['rng_state']['numpy'])

        # PyTorch RNG state must be a ByteTensor on CPU
        torch_rng_state = checkpoint['rng_state']['torch']
        if isinstance(torch_rng_state, torch.Tensor):
            # Ensure it's on CPU (in case checkpoint was loaded with map_location=cuda)
            torch_rng_state = torch_rng_state.cpu()
        torch.set_rng_state(torch_rng_state)

        # CUDA RNG state (if available)
        if checkpoint['rng_state']['torch_cuda'] is not None and torch.cuda.is_available():
            cuda_rng_states = checkpoint['rng_state']['torch_cuda']
            # Ensure all CUDA RNG states are on CPU before setting
            if isinstance(cuda_rng_states, list):
                cuda_rng_states = [state.cpu() if isinstance(state, torch.Tensor) else state
                                  for state in cuda_rng_states]
            torch.cuda.set_rng_state_all(cuda_rng_states)

    print(f"Checkpoint loaded successfully")
    print(f"  Episode: {checkpoint['episode']}, Steps: {checkpoint['total_steps']}, Top Score: {checkpoint['topscore']}")
    print(f"  Exploration Rate: {checkpoint['exploration_rate']:.4f}")
    print(f"  Saved at: {checkpoint.get('timestamp', 'unknown')}")

    return checkpoint

def find_latest_checkpoint(checkpoint_dir: str = "checkpoints", prefix: str = "checkpoint_") -> str:
    """
    Find the most recent checkpoint file in the specified directory.

    Args:
        checkpoint_dir: Directory to search for checkpoints
        prefix: Filename prefix for checkpoint files

    Returns:
        Path to the latest checkpoint, or None if no checkpoints found
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None

    # Find all checkpoint files
    checkpoints = list(checkpoint_path.glob(f"{prefix}*.pth"))

    if not checkpoints:
        return None

    # Sort by modification time and return the most recent
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return str(latest)

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

    device = torch.device("cuda" if (torch.cuda.is_available() and args.cuda) else "cpu")

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
    # env = TimeLimit(env, max_episode_steps=args.max_episode_steps)
    if "FIRE" in env.unwrapped.get_action_meanings():
        has_fire = True
        print("Environment has FIRE action, will use FireResetEnv wrapper")
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    # if args.capture_video:
    #     env = gym.wrappers.RecordVideo(env, "videos", step_trigger=lambda step: step % 1000 == 0)

    env_id = get_env_id(args.gym_id)
    print(f"env spec: {env.spec}")
    print(f"env metadata: {env.metadata}")
    if args.display == 1:
        env.metadata["render_fps"] = args.fps
        print(f"FPS: {args.fps}")
    print(f"env action space: {env.action_space}")
    print(f"env action space shape: {env.action_space.shape}")
    print(f"env observation space: {env.observation_space}")
    print(f"env observation space shape: {env.observation_space.shape}")
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
    EPISODE = 0
    total_steps = 0
    topscore = 0
    loss_val = 0.0
                    
    print("Using CNN model")
    input_shape = (IMAGE_CHANNELS, IMAGE_ROWS, IMAGE_COLS)
    agent = DQNAgent(env, input_shape=input_shape, device = device, seed =args.seed)
    # save a randomly initialized model for testing loading
    torch.save(agent.dQ_network.state_dict(), "saved_models/model_updates_dqn_" + env_id + "_" + str(total_steps) + ".pth")
    # Initialize checkpoint-related variables
    checkpoint_data = None
    env_rng_state_to_restore = None  # Will be set if resuming from checkpoint
    action_space_rng_state_to_restore = None  # Will be set if resuming from checkpoint
    observation_space_rng_state_to_restore = None  # Will be set if resuming from checkpoint

    if args.resume >= 1:
        # Determine which checkpoint to load
        checkpoint_file = None

        if args.resume == 2 and args.checkpoint_file:
            # Load specific checkpoint file
            checkpoint_file = args.checkpoint_file
        elif args.resume == 1:
            # Try to find latest checkpoint automatically
            checkpoint_file = find_latest_checkpoint(
                checkpoint_dir=args.checkpoint_dir,
                prefix=f"checkpoint_{env_id}_"
            )

            # Fallback to old-style loading if no checkpoint found
            if checkpoint_file is None and args.model_file:
                print(f"No checkpoint found, attempting old-style model loading...")
                file = args.model_file
                loaded_state_dict = torch.load(file)
                print(loaded_state_dict.keys())
                agent.dQ_network.load_state_dict(torch.load(file, map_location=device, weights_only=True))
                agent.target_dQ_network.load_state_dict(agent.dQ_network.state_dict())
                print(f"Loaded model from {file}")

                if args.buffer_file:
                    bm_file = args.buffer_file
                    agent.replay_buffer.load(bm_file)
                    print(f"Loaded replay buffer from {bm_file}")

        # Load comprehensive checkpoint if found
        if checkpoint_file:
            checkpoint_data = load_checkpoint(checkpoint_file, agent, device)

            # Load replay buffer if it exists
            buffer_file = checkpoint_file.replace('.pth', '_buffer.pkl')
            if Path(buffer_file).exists():
                agent.replay_buffer.load(buffer_file)
                print(f"Step {total_steps}: Loaded replay buffer from {buffer_file}")
            else:
                print(f"Step {total_steps}: Warning: Replay buffer not found at {buffer_file}")

            # CRITICAL: Store environment RNG state to restore later
            # (after environment is created/reset)
            env_rng_state_to_restore = checkpoint_data.get('rng_state', {}).get('env_np_random', None)
            # CRITICAL: Also store action_space RNG state (separate from env RNG!)
            action_space_rng_state_to_restore = checkpoint_data.get('rng_state', {}).get('action_space_np_random', None)
            # CRITICAL: Also store observation_space RNG state (also separate!)
            observation_space_rng_state_to_restore = checkpoint_data.get('rng_state', {}).get('observation_space_np_random', None)

    agent.dQ_network.train(True)

    global inner_loop_break
    inner_loop_break = False

    # Define test_steps and ep_info_buffer first (needed for both fresh and resumed training)
    test_steps = list(range(TEST_STEP_SIZE, MAX_TEST_STEPS + TEST_STEP_SIZE, TEST_STEP_SIZE))
    
    print(f"Step {total_steps}: test_steps: {test_steps}")
    ep_info_buffer = deque(maxlen=100)

    # Restore training state from checkpoint if resuming
    if checkpoint_data is not None:
        EPISODE = checkpoint_data.get('episode', 0)
        total_steps = checkpoint_data.get('total_steps', 0)
        topscore = checkpoint_data.get('topscore', 0)
        tested_steps = checkpoint_data.get('tested_steps', [0] * len(test_steps))
        print(f"Step {total_steps}: tested_steps restored: {tested_steps}")

        # DEBUG: Log first model weights
        print(f"\n{'='*70}")
        print(f"Step {total_steps}: DEBUG: After loading checkpoint")
        print(f"Step {total_steps}: First conv weight sum: {agent.dQ_network.cnn[0].weight.sum().item():.6f}")
        print(f"Step {total_steps}: Exploration rate: {agent.exploration_rate:.6f}")
        print(f"Step {total_steps}: Optimizer lr: {agent.optimizer.param_groups[0]['lr']}")
        print(f"{'='*70}\n")

        # Restore episode info buffer
        if 'ep_info_buffer' in checkpoint_data:
            ep_info_buffer.clear()
            ep_info_buffer.extend(checkpoint_data['ep_info_buffer'])

        print(f"Step {total_steps}: Resumed training from checkpoint:")
        print(f"Step {total_steps}: Starting at Episode {EPISODE}, Step {total_steps}")
        print(f"Step {total_steps}: Top Score: {topscore}")
        print(f"Step {total_steps}: Exploration Rate: {agent.exploration_rate:.4f}")
    else:
        # Fresh training - initialize tested_steps
        tested_steps = [0] * len(test_steps)

    skip = FRAMES_SKIP
    print(f"Step {total_steps}: skip: {skip}")
    a_t = 0

    # Setup TensorBoard logger
    # Create a unique run directory based on environment and timestamp
    if checkpoint_data is not None:
        # Resume with existing log directory if available
        log_folder = checkpoint_data.get('args', {}).get('log_folder', None)
        checkpoint_step = checkpoint_data.get('total_steps', 0)

        if log_folder is None or not Path(log_folder).exists():
            # Create new folder if old one doesn't exist
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_folder = f"./runs/{env_id}_resume_{timestamp}"
            print(f"Step {total_steps}: Original log folder not found, creating new: {log_folder}")
        else:
            print(f"\n{'='*60}")
            print(f"Step {total_steps}: Resuming with existing log folder: {log_folder}")
            print(f"Step {total_steps}: Checkpoint step: {checkpoint_step}")
            print(f"{'='*60}\n")

            # Use TensorBoard utilities to handle log filtering if needed
            if args.filter_tensorboard:
                if TENSORBOARD_UTILS_AVAILABLE:
                    print("Checking TensorBoard log continuity...")
                    log_folder = create_filtered_logdir_for_checkpoint(
                        original_logdir=log_folder,
                        checkpoint_step=checkpoint_step,
                        base_dir="./runs"
                    )
                else:
                    print(f"Step {total_steps}: Warning: TensorBoard utils not available (missing tensorboard package)")
                    print(f"Step {total_steps}: Install with: pip install tensorboard")
                    print(f"Step {total_steps}: Using original log folder - this may cause data discontinuity")
            else:
                print(f"Step {total_steps}: TensorBoard filtering disabled (--filter-tensorboard False)")
                print(f"Step {total_steps}: Using original log folder - this may cause data discontinuity")
    else:
        # Fresh training - create new log folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_folder = f"./runs/{env_id}_{timestamp}"
        print(f"Step {total_steps}: Creating new log folder: {log_folder}")

    # Store log folder in args for checkpointing
    args.log_folder = log_folder

    data_logger = configure(folder=log_folder, format_strings=['stdout', 'csv', 'tensorboard'])
    print(f"\nStep {total_steps}: TensorBoard logging to: {log_folder}")
    print(f"Step {total_steps}: View with: tensorboard --logdir {log_folder}\n")

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

    score = 0
    steps = 0
    episode_end = False
    psudo_episode_end = False
    done = False
    pdone = False
    terminated = False
    truncated = False
    total_reward = 0.0
    a_t = 0
    episode_reset_action = 0
    obs = np.zeros(0)
    info: dict = {}
    k = 0

    # initial reset
    # CRITICAL FIX: Don't reset environment with fixed seed when resuming!
    # When resuming, restore RNG state BEFORE reset so it uses correct random sequence
    if env_rng_state_to_restore is not None:
        # CRITICAL: Restore RNG state BEFORE calling reset!
        # Otherwise reset() will consume random numbers from wrong RNG state
        env.unwrapped.np_random.bit_generator.state = env_rng_state_to_restore
        print(f"Restored environment RNG state from checkpoint")

        # CRITICAL: Also restore action_space RNG state (separate from env RNG!)
        if action_space_rng_state_to_restore is not None and hasattr(env.action_space, 'np_random'):
            env.action_space.np_random.bit_generator.state = action_space_rng_state_to_restore
            print(f"Restored action_space RNG state from checkpoint")

        # CRITICAL: Also restore observation_space RNG state (also separate!)
        if observation_space_rng_state_to_restore is not None and hasattr(env.observation_space, 'np_random'):
            env.observation_space.np_random.bit_generator.state = observation_space_rng_state_to_restore
            print(f"Restored observation_space RNG state from checkpoint")

        # Now reset - this will use the restored RNG states
        obs, info = env.reset(seed=None)
        print(f"Environment reset using restored RNG states")
        
        # DEBUG: Log observation after reset (before noops)
        print(f"\nDEBUG: After reset, before noops")
        print(f"  Observation sum: {obs.sum()}, mean: {obs.mean():.2f}")        
    else:
        # Fresh training: reset with seed for reproducibility
        obs, info = env.reset(seed=args.seed)
        print(f"Environment reset with seed={args.seed}")

    checkpoint_saved = False
                
    lives_after = lives = env.unwrapped.ale.lives()
    
    print(f"env spec: {env.spec}")

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
        terminated = False
        truncated = False
        score = 0
        steps = 0
        frames_num = 0

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
        # print(f"{msg_prefix}Lives: {lives} -> {env.unwrapped.ale.lives()}，Step {total_steps}: No-ops after episode end: {noops}")
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

    def capture_video_no_obs():
        # nonlocal obs
        nonlocal frame
        if args.sensor == 1:
            try:
                frame = frame_queue.get(timeout=1.0)
            except queue.Empty:
                print("No frame available within 1s ...")

    def observe_no_obs():
        render_and_display()
        capture_video_no_obs()

    def action_no_obs():
        """
        Execute action with frame skipping and max pooling.

        """
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
        internal_obs = None
        total_reward = 0.0
        terminated = False
        truncated = False
        
        for i in range(skip):
            internal_obs, reward, terminated, truncated, info = env.step(episode_reset_action)
            observe()
            score += reward
            steps += 1
            frames_num += 1
            if i == skip - 2:
                obs_buffer[0] = internal_obs
            if i == skip - 1:
                obs_buffer[1] = internal_obs
            total_reward += float(reward)
            if terminated or truncated:
                break
        internal_obs = obs_buffer.max(axis=0)

    def direct_env_reset_no_obs():
        nonlocal score
        nonlocal steps
        nonlocal frames_num
        nonlocal terminated
        nonlocal truncated
        nonlocal info

        _, info = env.reset(seed=None)
        observe_no_obs()
        terminated = False
        truncated = False
        score = 0
        steps = 0
        frames_num = 0

    def noop_reset_action_no_obs():
        """Run random noops after env has already been reset. Handle mid-noop terminal resets.

        Caller must call env.reset() and observe() before this function.

        """
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
        info = {}
        for _ in range(noops):
            _, reward, terminated, truncated, info = env.step(0)
            observe_no_obs()
            score += reward
            steps += 1
            frames_num += 1
            if terminated or truncated:
                direct_env_reset_no_obs()

    def fire_reset_action():
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
            action_no_obs()
        if terminated or truncated:
            env.reset(seed=None)
            observe_no_obs()
            noop_reset_action_no_obs()

    # pause_moment = False
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
        if SB3_FIRE_RESET:
            fire_reset_action()
        else:
            reset_action()
        lives = env.unwrapped.ale.lives()
        
        episode_reset_action = 2
        if SB3_FIRE_RESET:
            fire_reset_action()
        else:
            reset_action()
        lives = env.unwrapped.ale.lives()
    # FireResetEnv reset end
    
    o_t = agent.preprocess(obs)
    stacked_o_t[0, -1, :, :] = o_t
    s_t = stacked_o_t.copy()
    print(f"s_t, {type(s_t)}, {s_t.shape}")

    if checkpoint_data is not None:
        print(f"Step {total_steps}: Episode info buffer (last 100 episodes):")
        print([ep_info["r"] for ep_info in ep_info_buffer])

        agent.update_target_network(
            total_steps=total_steps,
            target_update_freq=TARGET_UPDATE_INTERVAL
        )
        agent._update_current_progress_remaining(total_steps=total_steps, max_timesteps=MAX_TEST_STEPS)
        agent.exploration_rate = agent.exploration_schedule(agent._current_progress_remaining)

        EPISODE += 1

        if EPISODE > 1 and EPISODE % 4 == 0:
                                    
            ep_rew_mean = safe_mean([ep_info["r"] for ep_info in ep_info_buffer])
            ep_len_mean = safe_mean([ep_info["l"] for ep_info in ep_info_buffer])
            
            if len(ep_info_buffer) > 0 and len(ep_info_buffer[0]) > 0:
                data_logger.record("rollout/ep_rew_mean", ep_rew_mean)
                data_logger.record("rollout/ep_len_mean", ep_len_mean)
            data_logger.record("rollout/top_score", topscore)
            data_logger.record("rollout/exploration_rate", agent.exploration_rate)
            data_logger.record("time/fps", Calculated_FPS)
            data_logger.record("time/episodes", EPISODE)
            data_logger.record("time/total_steps", total_steps)
            data_logger.record("train/loss", loss_val)
            data_logger.dump(step=total_steps)

        # Training process start after LEARNS_START frames or the end of an episode
        if args.training != 0 and total_steps > LEARNING_STARTS and total_steps % TRAINING_FREQ == 0:
            
            loss_val = agent.training_step(
                total_steps=total_steps,
                batch_size=BATCH_SIZE,
                gamma=GAMMA,
                episode=EPISODE
            )

        # if total_steps % 1000000 == 0:
        if total_steps % TEST_STEP_SIZE == 0:
            torch.save(agent.dQ_network.state_dict(), "saved_models/model_updates_dqn_" + env_id + "_" + str(total_steps) + ".pth")        
        # End of training process

        if total_steps > MAX_TEST_STEPS:
            terminated = False
            truncated = False
            done = False
            pdone = False
            return
            
        terminated = False
        truncated = False
        done = False
        pdone = False
        episode_end = False
        psudo_episode_end = False
        info: dict = {}
        obs = np.zeros(0)
        
    while True:
        
        # Check if the thread crashed or was stopped
        if args.sensor == 1:
            if not cam.is_alive():
                print("Camera thread has stopped. Exiting...")
                break

        # state tensor
        s_t_tensor = torch.as_tensor(s_t, device=device)

        # Start of getting the action from the model
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

        if args.training != 0:
            a_t = agent.get_action_for_training(total_steps=total_steps, state=s_t_tensor, deterministic=False)
        else:
            a_t = agent.get_action(s_t_tensor)

        total_reward = 0.0
        terminated = False
        truncated = False
        for i in range(skip): # 4 is skipping frames
            obs, reward, terminated, truncated, info = env.step(a_t)
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
                episode_end = True
                if terminated:
                    done = True
                break

        obs = obs_buffer.max(axis=0)        
        o_t = agent.preprocess(obs)
        r_t = np.sign(float(total_reward))
        
        lives_after = env.unwrapped.ale.lives()

        if 0 < lives_after < lives and has_lives:
            pdone = True
            psudo_episode_end = True
            if episode_end:
                psudo_episode_end = False
            if done:
                pdone = False
            terminated = True
        
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
            
            # FireResetEnv reset start
            # EpisodicLifeEnv reset start
            if psudo_episode_end:
                # no-op action EpisodicLifeEnv reset
                obs = np.zeros(0)
                info: dict = {}
                episode_reset_action = 0
                action_not_in_the_loop()
                
                if terminated or truncated:
                    episode_end = True

            if episode_end:
                if score > topscore:
                    topscore = score
                ep_info = {"r": score, "l": steps, "fps": Calculated_FPS}
                ep_info_buffer.append(ep_info)

                # Save checkpoint at test milestones
                for i, (test_step, tested) in enumerate(zip(test_steps, tested_steps)):
                    if tested == 0 and total_steps >= test_step:
                        print(f"*** Saving checkpoint at test milestone: Episode {EPISODE}, when Step {total_steps} >= {test_step} ***")

                        # CRITICAL FIX: Mark as tested BEFORE saving checkpoint
                        # so the checkpoint contains the correct tested_steps state
                        tested_steps[i] = 1

                        checkpoint_filename = f"{args.checkpoint_dir}/checkpoint_{env_id}_ep_{EPISODE}_step_{total_steps}.pth"
                        save_checkpoint(
                            filepath=checkpoint_filename,
                            agent=agent,
                            total_steps=total_steps,
                            episode=EPISODE,
                            topscore=topscore,
                            tested_steps=tested_steps,
                            ep_info_buffer=ep_info_buffer,
                            env_id=env_id,
                            args=args,
                            env=env
                        )

                        torch.save(agent.dQ_network.state_dict(),
                                    f"{args.checkpoint_dir}/model_tests_dqn_{env_id}_ep_{EPISODE}_step_{total_steps}.pth")

                        print(f"Test milestone {test_step} marked as completed (checkpoint saved at step {total_steps})")
                        checkpoint_saved = True

                # Noop reset
                lives = lives_after
                obs, _ = env.reset(seed=None)
                observe()
                obs = np.zeros(0)
                info: dict = {}
                noop_reset_action(msg_prefix="Env Reset: ")
                
            lives = lives_after = env.unwrapped.ale.lives()
            # EpisodicLifeEnv reset end

            if has_fire:
                episode_reset_action = 1
                if SB3_FIRE_RESET:
                    fire_reset_action()
                else:
                    reset_action()
                lives = lives_after = env.unwrapped.ale.lives()
                
                episode_reset_action = 2
                if SB3_FIRE_RESET:
                    fire_reset_action()
                else:
                    reset_action()
                lives = lives_after = env.unwrapped.ale.lives()
            # FireResetEnv reset end

            o_t = agent.preprocess(obs)
            stacked_o_t[...] = 0

        stacked_o_t = np.roll(stacked_o_t, shift=-1, axis=1)
        stacked_o_t[0, -1, :, :] = o_t.copy()
        # End of taking action and getting the reward from the environment        

        # Start of updating the replay buffer
        s_t_batch = s_t.copy()
        if psudo_episode_end or episode_end:
            s_t_next_batch = terminal_stacked_o_t.copy()
        else:
            s_t_next_batch = stacked_o_t.copy()
        a_t_batch = np.array([a_t])
        r_t_batch = np.array([r_t])
        done_batch = np.array([done or pdone])
        info_batch = [info]

        agent.replay_buffer.add(s_t_batch, s_t_next_batch, a_t_batch, r_t_batch, done_batch, info_batch)
        # End of updating the replay buffer
        
        if checkpoint_saved:
            # Save replay buffer
            buffer_filename = checkpoint_filename.replace('.pth', '_buffer.pkl')
            agent.replay_buffer.save_buffer(buffer_filename)
            print(f"Replay buffer saved to {buffer_filename}")
            checkpoint_saved = False
        
        s_t = stacked_o_t.copy()
        
        agent.update_target_network(
            total_steps=total_steps,
            target_update_freq=TARGET_UPDATE_INTERVAL
        )
        agent._update_current_progress_remaining(total_steps=total_steps, max_timesteps=MAX_TEST_STEPS)
        agent.exploration_rate = agent.exploration_schedule(agent._current_progress_remaining)

        # if pdone or done:
        if psudo_episode_end or episode_end:
                
            EPISODE += 1

            if EPISODE > 1 and EPISODE % 4 == 0:
                                        
                ep_rew_mean = safe_mean([ep_info["r"] for ep_info in ep_info_buffer])
                ep_len_mean = safe_mean([ep_info["l"] for ep_info in ep_info_buffer])
                
                if len(ep_info_buffer) > 0 and len(ep_info_buffer[0]) > 0:
                    data_logger.record("rollout/ep_rew_mean", ep_rew_mean)
                    data_logger.record("rollout/ep_len_mean", ep_len_mean)
                data_logger.record("rollout/top_score", topscore)
                data_logger.record("rollout/exploration_rate", agent.exploration_rate)
                data_logger.record("time/fps", Calculated_FPS)
                data_logger.record("time/episodes", EPISODE)
                data_logger.record("time/total_steps", total_steps)
                data_logger.record("train/loss", loss_val)
                data_logger.dump(step=total_steps)

        # Training process start after LEARNS_START frames or the end of an episode
        if args.training != 0 and total_steps > LEARNING_STARTS and total_steps % TRAINING_FREQ == 0:
            loss_val = agent.training_step(
                total_steps=total_steps,
                batch_size=BATCH_SIZE,
                gamma=GAMMA,
                episode=EPISODE
            )

        if total_steps % TEST_STEP_SIZE == 0:
            torch.save(agent.dQ_network.state_dict(), "saved_models/model_updates_dqn_" + env_id + "_" + str(total_steps) + ".pth")        

        if total_steps > MAX_TEST_STEPS and done == True:
            terminated = False
            truncated = False
            done = False
            pdone = False
            break
            
        if psudo_episode_end or episode_end:
            terminated = False
            truncated = False
            done = False
            pdone = False
            episode_end = False
            psudo_episode_end = False
            # pause_moment = True
            info: dict = {}
            obs = np.zeros(0)
        # End of training process
        
    print(f"steps: {steps}")
    print(f"info: {info}")
    pygame.quit()
    print("end of training")
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
