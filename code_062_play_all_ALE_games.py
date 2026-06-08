"""Utilities of visualising an environment."""
# from code 022_play_and_respond.py

from __future__ import annotations

from collections import deque
from typing import Callable, List

import sys
sys.path.append("domain/")

import cv2
import cv2
import gymnasium as gym
from gymnasium import Env, logger
from gymnasium.wrappers import TimeLimit
from gymnasium.core import ActType, ObsType
from gymnasium.error import DependencyNotInstalled

try:
    import pygame
    from pygame import Surface
    from pygame.event import Event
except ImportError as e:
    raise gym.error.DependencyNotInstalled(
        'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
    ) from e

import domain.flappy_bird as flappy_bird
from domain.flappy_bird import FlappyBirdEnv
# print(f"flappy bird window size: {flappy_bird.SCREENWIDTH} x {flappy_bird.SCREENHEIGHT}")

import serial
import time

import argparse
from distutils.util import strtobool

import numpy as np

from evaluation.atari_data import get_env_id

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
except ImportError:
    logger.warn('matplotlib is not installed, run `pip install "gymnasium[other]"`')
    matplotlib, plt = None, None

# import matplotlib
# # Use 'TkAgg', 'Qt5Agg', 'Qt4Agg', etc.
# # matplotlib.use('TkAgg')

# import matplotlib.pyplot as plt

import tkinter as tk
root = tk.Tk()
root.withdraw()  # Hide the root window
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
print(f"screen width: {screen_width}, screen height: {screen_height}")

import os

class MissingKeysToAction(Exception):
    """Raised when the environment does not have a default ``keys_to_action`` mapping."""

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--gym-id', type=str, default="BreakoutNoFrameskip-v4",
        help='the id of the gym environment')
    parser.add_argument("--max-episode-steps", type=int, default=60000,
        help="how many steps to run in one episode in each environment")
    parser.add_argument("--fps", type=int, default=30,
        help="frame per second for the environment")
    parser.add_argument("--zoom", type=float, default=1.0,
        help="zoom in the environment, 1.0 is no zoom, 2.0 is double zoom")
    parser.add_argument("--seed", type=int, default=None,
        help="random seed for the environment")
    parser.add_argument("--model", type=int, default=1,
        help="model for the agent, 1 is CNN, 2 is CNN trained with ppo, 3 is transformer")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--sensor", type=int, default=0,
        help="use sensor or not, 0 is not using sensor, 1 is using sensor")
    parser.add_argument("--actuator", type=int, default=0,
        help="use actuator or not, 0 is not using actuator, 1 is using hardware emulated keyboard, 2 is using physical keyboard clicker")
    parser.add_argument("--bptime", type=int, default=0,
        help="use sensor or not, 0 is not showing back propagation time, 1 is showing time")
    parser.add_argument("--crop", type=int, default=0,
        help="use sensor or not, 0 is not cropping the window, 1 is cropping the window")
    parser.add_argument("--forpaper", type=int, default=0,
        help="collect data for paper writing, 0 is not collecting, 1 is collecting")
    args = parser.parse_args()
    return args

args = parse_args()
print(args)
print(vars(args))

# device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
device = "cpu"

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
        # print("inside send_data")
        ser.write(message.encode())
        ser.flush()
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
                    print(f"Received: {response}")
                else:
                    print("No data received.")
            else:
                break
    except serial.SerialException as e:
        print(f"Error: {e}") 
        
def send_info(ser, info):
    try:
        message = "Send {}\n\r".format(info)
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

# End of Hardware emulated keyboard functions

plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
line1, = ax.plot([], [], 'o-', label='Line 1') 
line2, = ax.plot([], [], 'x-', label='Line 2')

# ax.set_xlim(0, 10)
# ax.set_ylim(0, 100)
plt.legend() 
plt.grid(True)
x_data = []
y1_data = []
y2_data = []

# import gymnasium as gym
# import pygame
# from gymnasium.utils.play import play
# from domain.play import play
import ale_py
gym.register_envs(ale_py)

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
        # self.screen = pygame.display.set_mode(self.video_size, pygame.RESIZABLE)
        self.screen = None
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
        print(f"keys_to_action: {keys_to_action}")
        relevant_keys = set(sum((list(k) for k in keys_to_action.keys()), []))
        print(f"relevant_keys: {relevant_keys}")
        return relevant_keys

    def _get_video_size(self, zoom: float | None = None) -> tuple[int, int]:
        rendered = self.env.render()
        if isinstance(rendered, List):
            rendered = rendered[-1]
        assert rendered is not None and isinstance(rendered, np.ndarray)
        video_size = (rendered.shape[1], rendered.shape[0])

        if zoom is not None:
            video_size = (int(video_size[0] * zoom), int(video_size[1] * zoom))

        return video_size

    def process_event(self, event: Event):
        """Processes a PyGame event.

        In particular, this function is used to keep track of which buttons are currently pressed
        and to exit the :func:`play` function when the PyGame window is closed.

        Args:
            event: The event to process
        """
        if event.type == pygame.KEYDOWN:
            # print(f"Key pressed: {pygame.key.name(event.key)}")
            if event.key in self.relevant_keys:
                self.pressed_keys.append(event.key)
            elif event.key == pygame.K_ESCAPE:
                self.running = False
        elif event.type == pygame.KEYUP:
            # print(f"Key released: {pygame.key.name(event.key)}")
            if event.key in self.relevant_keys:
                self.pressed_keys.remove(event.key)
        elif event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.WINDOWRESIZED:
            # Compute the maximum video size that fits into the new window
            scale_width = event.x / self.video_size[0]
            scale_height = event.y / self.video_size[1]
            scale = min(scale_height, scale_width)
            self.video_size = (scale * self.video_size[0], scale * self.video_size[1])

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
    # print(f"surface_size: {surface_size}, video_size: {video_size}")
    width_offset = (surface_size[0] - video_size[0]) / 2
    height_offset = (surface_size[1] - video_size[1]) / 2
    screen.fill((0, 0, 0))
    screen.blit(pyg_img, (width_offset, height_offset))

class PlayPlot:
    """Provides a callback to create live plots of arbitrary metrics when using :func:`play`.

    This class is instantiated with a function that accepts information about a single environment transition:
        - obs_t: observation before performing action
        - obs_tp1: observation after performing action
        - action: action that was executed
        - rew: reward that was received
        - terminated: whether the environment is terminated or not
        - truncated: whether the environment is truncated or not
        - info: debug info

    It should return a list of metrics that are computed from this data.
    For instance, the function may look like this::

        >>> def compute_metrics(obs_t, obs_tp, action, reward, terminated, truncated, info):
        ...     return [reward, info["cumulative_reward"], np.linalg.norm(action)]

    :class:`PlayPlot` provides the method :meth:`callback` which will pass its arguments along to that function
    and uses the returned values to update live plots of the metrics.

    Typically, this :meth:`callback` will be used in conjunction with :func:`play` to see how the metrics evolve as you play::

        >>> plotter = PlayPlot(compute_metrics, horizon_timesteps=200,                               # doctest: +SKIP
        ...                    plot_names=["Immediate Rew.", "Cumulative Rew.", "Action Magnitude"])
        >>> play(your_env, callback=plotter.callback)                                                # doctest: +SKIP
    """

    def __init__(
        self, callback: Callable, horizon_timesteps: int, plot_names: list[str]
    ):
        """Constructor of :class:`PlayPlot`.

        The function ``callback`` that is passed to this constructor should return
        a list of metrics that is of length ``len(plot_names)``.

        Args:
            callback: Function that computes metrics from environment transitions
            horizon_timesteps: The time horizon used for the live plots
            plot_names: List of plot titles

        Raises:
            DependencyNotInstalled: If matplotlib is not installed
        """
        self.data_callback = callback
        self.horizon_timesteps = horizon_timesteps
        self.plot_names = plot_names

        if plt is None:
            raise DependencyNotInstalled(
                'matplotlib is not installed, run `pip install "gymnasium[other]"`'
            )

        num_plots = len(self.plot_names)
        self.fig, self.ax = plt.subplots(num_plots)
        if num_plots == 1:
            self.ax = [self.ax]
        for axis, name in zip(self.ax, plot_names):
            axis.set_title(name)
        self.t = 0
        self.cur_plot: list[plt.Axes | None] = [None for _ in range(num_plots)]
        self.data = [deque(maxlen=horizon_timesteps) for _ in range(num_plots)]

    def callback(
        self,
        obs_t: ObsType,
        obs_tp1: ObsType,
        action: ActType,
        rew: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ):
        """The callback that calls the provided data callback and adds the data to the plots.

        Args:
            obs_t: The observation at time step t
            obs_tp1: The observation at time step t+1
            action: The action
            rew: The reward
            terminated: If the environment is terminated
            truncated: If the environment is truncated
            info: The information from the environment
        """
        points = self.data_callback(
            obs_t, obs_tp1, action, rew, terminated, truncated, info
        )
        for point, data_series in zip(points, self.data):
            data_series.append(point)
        self.t += 1

        xmin, xmax = max(0, self.t - self.horizon_timesteps), self.t

        for i, plot in enumerate(self.cur_plot):
            if plot is not None:
                plot.remove()
            self.cur_plot[i] = self.ax[i].scatter(
                range(xmin, xmax), list(self.data[i]), c="blue"
            )
            self.ax[i].set_xlim(xmin, xmax)

        if plt is None:
            raise DependencyNotInstalled(
                'matplotlib is not installed, run `pip install "gymnasium[other]"`'
            )
        plt.pause(0.000001)

def main():

    if args.actuator == 1:
        # serial port number based on the system setting
        port = '/dev/ttyUSB0'
        # baud rate for serial communication
        baudrate = 115200
        # serial port configuration
        ser = configure_serial(port, baudrate)

        time.sleep(0.5)
        ser.write(b'READY\n')
        print("Handshake signal sent")
        time.sleep(1)
        response = ser.read(ser.in_waiting)
        print(f"Received: {response}")
        if response == b'ACK\n':
            print("Handshake successful!")
        else:
            print("Handshake failed!")
            return

    sps_time1 = time.time()
    sps_time2 = time.time()
    steps = 0
    currentScore = 0.0
    topScore = 0.0
    has_lives = False
    lives = 0
    
    # Create environment
    if args.gym_id == "FlappyBird":
        env = FlappyBirdEnv(FPS = args.fps, render_mode = "rgb_array")
    else:
        # env = gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array")
        # env = gym.make(args.gym_id, full_action_space=True, render_mode="rgb_array")
        env = gym.make(args.gym_id, render_mode="rgb_array")
        env = TimeLimit(env, max_episode_steps=args.max_episode_steps)

    env_id = get_env_id(args.gym_id)
    
    match env_id:
        
        case "flappybird":        
            keys_to_action = {
                (pygame.K_UP,): 1,  # FLAP
            }

        case "breakout":
            keys_to_action = {
                # NOOP is 0, no action
                (pygame.K_SPACE,):  1,  # Fire (release ball)
                (pygame.K_RIGHT,):  2,  # Move right
                (pygame.K_LEFT,):   3,  # Move left
            }

        case "space_invaders":    
            keys_to_action = {
                # NOOP is 0, no action
                # (pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9, pygame.K_COMMA, pygame.K_PERIOD, pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET): 0,
                (pygame.K_SPACE,): 1,  # FIRE
                (pygame.K_UP,): 2,  # UP
                (pygame.K_RIGHT,): 3, # RIGHT
                (pygame.K_LEFT,): 4,  # LEFT
                (pygame.K_DOWN,): 5,  # DOWN
                (pygame.K_UP, pygame.K_RIGHT): 6,  # UPRIGHT
                (pygame.K_UP, pygame.K_LEFT): 7,  # UPLEFT
                (pygame.K_DOWN, pygame.K_RIGHT): 8,  # DOWNRIGHT
                (pygame.K_DOWN, pygame.K_LEFT): 9,  # DOWNLEFT
                (pygame.K_UP, pygame.K_SPACE): 10,  # UPFIRE
                (pygame.K_RIGHT, pygame.K_SPACE): 11,  # RIGHTFIRE
                (pygame.K_LEFT, pygame.K_SPACE): 12,  # LEFTFIRE
                (pygame.K_DOWN, pygame.K_SPACE): 13,  # DOWNFIRE
                (pygame.K_UP, pygame.K_RIGHT, pygame.K_SPACE): 14,  # UPRIGHTFIRE
                (pygame.K_UP, pygame.K_LEFT, pygame.K_SPACE): 15,  # UPLEFTFIRE
                (pygame.K_DOWN, pygame.K_RIGHT, pygame.K_SPACE): 16,  # DOWNRIGHTFIRE
                (pygame.K_DOWN, pygame.K_LEFT, pygame.K_SPACE): 17,  # DOWNLEFTFIRE
            }

        # keys_to_action = {
        #     (pygame.K_a,): 1,  # Fire
        #     (pygame.K_b,): 2,  # move up
        #     (pygame.K_c,): 3, # Move right
        #     (pygame.K_d,): 4,  # Move left
        #     (pygame.K_e,): 5,  # Fire
        #     (pygame.K_b, pygame.K_c): 6,  # Fire
        #     (pygame.K_b, pygame.K_d): 7,  # Fire
        #     (pygame.K_e, pygame.K_c): 8,  # Fire
        #     (pygame.K_e, pygame.K_d): 9,  # Fire
        #     (pygame.K_b, pygame.K_a): 10,  # Fire
        #     (pygame.K_b, pygame.K_a): 11,  # Fire
        #     (pygame.K_d, pygame.K_a): 12,  # Fire
        #     (pygame.K_e, pygame.K_a): 13,  # Fire
        #     (pygame.K_b, pygame.K_c, pygame.K_a): 14,  # Fire
        #     (pygame.K_b, pygame.K_d, pygame.K_a): 15,  # Fire
        #     (pygame.K_e, pygame.K_c, pygame.K_a): 16,  # Fire
        #     (pygame.K_e, pygame.K_d, pygame.K_a): 17,  # Fire
        # }


    if hasattr(env.unwrapped, 'ale') and hasattr(env.unwrapped.ale, 'lives'):
        has_lives = True
    if has_lives:
        lives = env.unwrapped.ale.lives()
    print("lives: ", lives)

    env.reset(seed=args.seed)

    # keys_to_action = {
    #     (pygame.K_UP,): 1,  # FLAP
    # }

    if keys_to_action is None:
        if env.has_wrapper_attr("get_keys_to_action"):
            keys_to_action = env.get_wrapper_attr("get_keys_to_action")()
        else:
            assert env.spec is not None
            raise MissingKeysToAction(
                f"{env.spec.id} does not have explicit key to action mapping, "
                "please specify one manually"
            )

    assert keys_to_action is not None

    # validate the `keys_to_action` set provided
    assert isinstance(keys_to_action, dict)
    for key, action in keys_to_action.items():
        if isinstance(key, tuple):
            assert len(key) > 0
            assert all(isinstance(k, (str, int)) for k in key)
        else:
            assert isinstance(key, (str, int))

        assert action in env.action_space

    key_code_to_action = {}
    for key_combination, action in keys_to_action.items():
        key_code = tuple(
            sorted(ord(key) if isinstance(key, str) else key for key in key_combination)
        )
        key_code_to_action[key_code] = action

    print(f"key_code_to_action: {key_code_to_action}")
    game = PlayableGame(env, key_code_to_action, zoom=args.zoom)
    print(f"game.video_size: {game.video_size}")

    # window_x = 50
    # window_y = screen_height - game.video_size[1] - 120

    window_x = 1200
    window_y = 80

    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{window_x},{window_y}"
    game.screen = pygame.display.set_mode(game.video_size, pygame.RESIZABLE)

    wait_on_player = False
    
    print(f"env.metadata: {env.metadata}")

    # if fps is None:
    fps = env.metadata.get("render_fps", 30)
    fps = args.fps if args.fps is not None else fps
    
    print(f"fps: {fps}")
    time.sleep(2)

    done, obs = True, None
    clock = pygame.time.Clock()
    # episode = 0
    
    # i = 0    
    # cycle_start_time = time.time()
    
    if args.forpaper == 1:
        i = 0
    
    while game.running:

        if done:
            done = False
            terminated = False
            truncated = False
            action = 0
            rew = 0.0
            # obs, _ = env.reset(seed=args.seed)
            obs, _ = env.reset(seed=None)

            currentScore = 0.0
            print("Env reset current Score: " + str(currentScore))
            if has_lives:
                lives = env.unwrapped.ale.lives()
                # print("lives: ", lives)
                if args.actuator == 1:
                    message = '{"action": ' + str(action) + ', "reward": ' + str(rew) + ', "lives": ' + str(lives) + ', "truncated": ' + str(truncated).lower() + ', "terminated": ' + str(terminated).lower() + ', "score": ' + str(currentScore) + '}'
                    send_data(ser, message)
            sps_time1 = time.time()
            steps = 0
            # internal_steps = 0

        elif wait_on_player is False or len(game.pressed_keys) > 0:
            action = key_code_to_action.get(tuple(sorted(game.pressed_keys)), 0)
            # if game.pressed_keys:
            #     print(f"Action taken: {action}")
            # internal_steps += 1
            # print(f"steps: {internal_steps}, Action taken: {action}")
            obs, rew, terminated, truncated, info = env.step(action)

            steps += 1
            done = terminated or truncated
            if has_lives:
                new_lives = env.unwrapped.ale.lives()
                if new_lives < lives:
                    # internal_steps = 0
                    lives = new_lives
                    # print("lives: ", lives)
                    if args.actuator == 1:
                        message = '{"action": ' + str(action) + ', "reward": ' + str(0) + ', "lives": ' + str(lives) + ', "truncated": ' + str(truncated).lower() + ', "terminated": ' + str(terminated).lower() + ', "score": ' + str(currentScore) + '}'
                        send_data(ser, message)
                    # time.sleep(0.1)
                    # episode += 1
                if(rew > 0.0):
                    currentScore += rew
                    topScore = max(topScore, currentScore)
                    if args.actuator == 1:
                        # print(f"obs shape: {obs.shape}, Action taken: {action}, Reward: {rew}, Terminated: {terminated}, Truncated: {truncated}, info: {info}")
                        message = '{"action": ' + str(action) + ', "reward": ' + str(rew) + ', "lives": ' + str(lives) + ', "truncated": ' + str(truncated).lower() + ', "terminated": ' + str(terminated).lower() + ', "score": ' + str(currentScore) + '}'
                        send_data(ser, message)
                if done:
                    sps_time2 = time.time()
                    SPS = steps/(sps_time2-sps_time1)
                    print(f"terminated: {terminated} steps: {steps}")
                    print(f"steps per second: {SPS:>5.2f}")
                    print("Current Score: " + str(currentScore) + " Top Score: " + str(topScore))
                    if args.actuator == 1:
                        # print(f"obs shape: {obs.shape}, Action taken: {action}, Reward: {rew}, Terminated: {terminated}, Truncated: {truncated}, info: {info}")
                        message = '{"action": ' + str(action) + ', "reward": ' + str(rew) + ', "lives": ' + str(lives) + ', "truncated": ' + str(truncated).lower() + ', "terminated": ' + str(terminated).lower() + ', "score": ' + str(currentScore) + '}'
                        send_data(ser, message)
                    time.sleep(0.1)
                    # episode += 1               
            else:
                if(rew > 0.0):
                    currentScore += rew
                    topScore = max(topScore, currentScore)
                    if args.actuator == 1:
                        # print(f"obs shape: {obs.shape}, Action taken: {action}, Reward: {rew}, Terminated: {terminated}, Truncated: {truncated}, info: {info}")
                        message = '{"action": ' + str(action) + ', "reward": ' + str(rew) + ', "lives": ' + str(lives) + ', "truncated": ' + str(truncated).lower() + ', "terminated": ' + str(terminated).lower() + ', "score": ' + str(currentScore) + '}'
                        send_data(ser, message)            
                if done:
                    sps_time2 = time.time()
                    SPS = steps/(sps_time2-sps_time1)
                    print(f"terminated: {terminated} steps: {steps}")
                    print(f"steps per second: {SPS:>5.2f}")
                    print("Current Score: " + str(currentScore) + " Top Score: " + str(topScore))
                    if args.actuator == 1:
                        # print(f"obs shape: {obs.shape}, Action taken: {action}, Reward: {rew}, Terminated: {terminated}, Truncated: {truncated}, info: {info}")
                        message = '{"action": ' + str(action) + ', "reward": ' + str(rew) + ', "lives": ' + str(lives) + ', "truncated": ' + str(truncated).lower() + ', "terminated": ' + str(terminated).lower() + ', "score": ' + str(currentScore) + '}'
                        send_data(ser, message)
                    # print("steps: ", steps)
                    time.sleep(0.1)

        if obs is not None:
            rendered = env.render()
            if isinstance(rendered, List):
                rendered = rendered[-1]
            assert rendered is not None and isinstance(rendered, np.ndarray)

            display_arr(
                game.screen, rendered, transpose=True, video_size=game.video_size
            )

            if args.forpaper == 1:
                # save file for analysis
                print(type(obs))
                print(f"obs shape: {obs.shape}, obs dtype: {obs.dtype}")
                # obs_array = np.array(obs)
                np.save("env_obs_file.npy", obs)
                arr = pygame.surfarray.array3d(game.screen)
                arr = np.transpose(arr, (1, 0, 2))
                np.save("game_screen_file.npy", arr)
                if i > 0:
                    time.sleep(30)
                    image = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
                    image = cv2.resize(image, (84, 84), interpolation=cv2.INTER_AREA)
                    np.save("obs_resized_file.npy", image)
                    raise Exception("stop here for debug")
                i += 1
                # need to remove in the experiment

        # process pygame events
        for event in pygame.event.get():
            game.process_event(event)
        
        # time.sleep(0.1)
        pygame.display.flip()
        clock.tick(fps)
        # print(f"current episode: {episode}, current fps is: {clock.get_fps()}")

        # if game.pressed_keys:
        #     print(f"62 pressed_keys: {game.pressed_keys}")
                
        # i += 1
        # if i == 1000:
        #     cycle_end_time = time.time()
        #     cycle_time = (cycle_end_time - cycle_start_time)/1000
        #     print(f"Time taken for each cycles: {cycle_time:.5f} seconds")
        #     i = 0
        #     cycle_start_time = time.time()

    if args.actuator == 1:
        if ser and ser.isOpen():
            message = '{STOP}'
            send_data(ser, message)
            time.sleep(0.1)
            ser.close()
            print("Serial port is closed.")
    
    pygame.quit()

    quit()

if __name__ == "__main__":
    main()