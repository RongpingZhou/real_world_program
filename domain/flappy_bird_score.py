from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding

import numpy as np
import sys
import random
import pygame
import flappy_bird_utils
import pygame.surfarray as surfarray
from pygame.locals import *
from itertools import cycle
import skimage
from typing import Any, Literal
from enum import Enum

SCREENWIDTH  = 288  #use 256 (power of 2)
SCREENHEIGHT = 405

PIPEGAPSIZE = 100 # gap between upper and lower part of pipe
BASEY = SCREENHEIGHT * 1
PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])
SHOW_SCORE = False

# pygame.init()

# FPSCLOCK = pygame.time.Clock()

# SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT), pygame.HIDDEN)

# pygame.display.set_caption('Flappy Bird')


# IMAGES, SOUNDS, HITMASKS = FlappyBirdEnv.load()
# PLAYER_WIDTH = IMAGES['player'][0].get_width()  #images array is obtained from flappy_bird_utils above
# PLAYER_HEIGHT = IMAGES['player'][0].get_height()
# PIPE_WIDTH = IMAGES['pipe'][0].get_width()
# PIPE_HEIGHT = IMAGES['pipe'][0].get_height()
# BACKGROUND_WIDTH = IMAGES['background'].get_width()

class Actions(Enum):
    FLAP = 0
    NO_FLAP = 1

class FlappyBirdEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, FPS: int = 10, render_mode: Literal["human", "rgb_array"] | None = None,):
        

        # PLAYER_WIDTH = IMAGES['player'][0].get_width()  #images array is obtained from flappy_bird_utils above
        # PLAYER_HEIGHT = IMAGES['player'][0].get_height()
        # PIPE_WIDTH = IMAGES['pipe'][0].get_width()
        # PIPE_HEIGHT = IMAGES['pipe'][0].get_height()
        # BACKGROUND_WIDTH = IMAGES['background'].get_width()

        
        # self.fpsclock = pygame.time.Clock()
        # if render_mode == "human":
        #     SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
        # else:
        #     SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT), pygame.HIDDEN)

        # pygame.display.set_caption('Flappy Bird')
        self.screenwidth = SCREENWIDTH
        self.screenheight = SCREENHEIGHT
        self.basey = BASEY
        pygame.init()
        self.canvas = pygame.Surface((self.screenwidth, self.screenheight))
        self.window = pygame.display.set_mode((self.screenwidth, self.screenheight))
        
        self.IMAGES, self.SOUNDS, self.HITMASKS = self.load()
    
        self.player_height = self.IMAGES['player'][0].get_height()
        self.player_width = self.IMAGES['player'][0].get_width()  #images array is obtained from flappy_bird_utils above
        self.pipe_width = self.IMAGES['pipe'][0].get_width()
        self.pipe_height = self.IMAGES['pipe'][0].get_height()
        self.background_width = self.IMAGES['background'].get_width()
        
        

        self.score = self.playerIndex = self.loopIter = 0
        self.playerx = int(self.screenwidth * 0.2)
        self.playery = int((self.screenheight - self.player_height) / 2)
        self.basex = 0
        self.baseShift = self.IMAGES['base'].get_width() - self.background_width
        self.FPS = FPS

        newPipe1 = self.getRandomPipe()
        newPipe2 = self.getRandomPipe()
        self.upperPipes = [
            {'x': self.screenwidth, 'y': newPipe1[0]['y']},
            {'x': self.screenwidth + (self.screenwidth / 2), 'y': newPipe2[0]['y']},
        ]
        self.lowerPipes = [
            {'x': self.screenwidth, 'y': newPipe1[1]['y']},
            {'x': self.screenwidth + (self.screenwidth / 2), 'y': newPipe2[1]['y']},
        ]

        # player velocity, max velocity, downward accleration, accleration on flap
        self.pipeVelX = -4
        self.playerVelY    =  0    # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY =  10   # max vel along Y, max descend speed
        self.playerMinVelY =  -8   #-8   # min vel along Y, max ascend speed
        self.playerAccY    =   2   # players downward accleration
        self.playerFlapAcc =  -10   #-10  players speed on flapping
        self.playerFlapped = False # True when player flaps

        self.action_space = spaces.Discrete(2)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._actions = {
            Actions.FLAP.value: np.array([1, 0]),
            Actions.NO_FLAP.value: np.array([0, 1]),
        }

        image_shape = (
                self.screenheight,
                self.screenwidth,
                3,
        )

        self.observation_space = spaces.Box(
                low=0, high=255, dtype=np.uint8, shape=image_shape
        )

        if render_mode is not None and render_mode not in {"rgb_array", "human"}:
            raise error.Error(
                f"Render mode {render_mode} not supported (rgb_array, human)."
            )
        self.render_mode = render_mode
        
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """        
        
        self.window = None
        self.clock = None

        # # draw sprites
        # # self.SCREEN.blit(IMAGES['background'], (0,0))
        # SCREEN.blit(IMAGES['background'], (0,0))

        # for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
        #     # self.SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
        #     SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
        #     # self.SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))
        #     SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        # # self.SCREEN.blit(IMAGES['base'], (self.basex, BASEY))
        # SCREEN.blit(IMAGES['base'], (self.basex, BASEY))
        # # print score so player overlaps the score
        # # self.SCREEN.blit(IMAGES['player'][self.playerIndex],
        # SCREEN.blit(IMAGES['player'][self.playerIndex],
        #             (self.playerx, self.playery))

        # pygame.display.update()
        # # self.fpsclock.tick(self.FPS)
        # FPSCLOCK.tick(self.FPS)
        
    def _get_obs(self):
        # image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        image_data = np.transpose(
                np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2)
            )

        return image_data

    def _get_info(self):
        return {
            "agent_x": self.playerx,
            "agent_y": self.playery,
            "score": self.score,
            }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.score = self.playerIndex = self.loopIter = 0
        self.playerx = int(self.screenwidth * 0.2)
        self.playery = int((self.screenheight - self.player_height) / 2)
        self.basex = 0
        self.baseShift = self.IMAGES['base'].get_width() - self.background_width
        # self.FPS = FPS

        newPipe1 = self.getRandomPipe()
        newPipe2 = self.getRandomPipe()
        self.upperPipes = [
            {'x': self.screenwidth, 'y': newPipe1[0]['y']},
            {'x': self.screenwidth + (self.screenwidth / 2), 'y': newPipe2[0]['y']},
        ]
        self.lowerPipes = [
            {'x': self.screenwidth, 'y': newPipe1[1]['y']},
            {'x': self.screenwidth + (self.screenwidth / 2), 'y': newPipe2[1]['y']},
        ]

        # player velocity, max velocity, downward accleration, accleration on flap
        self.pipeVelX = -4
        self.playerVelY    =  0    # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY =  10   # max vel along Y, max descend speed
        self.playerMinVelY =  -8   #-8   # min vel along Y, max ascend speed
        self.playerAccY    =   2   # players downward accleration
        self.playerFlapAcc =  -10   #-10  players speed on flapping
        self.playerFlapped = False # True when player flaps

        observation = self._get_obs()
        info = self._get_info()
        # info = None

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, act):
        pygame.event.pump()

        reward = 0.1
        terminated = False

        action = self._actions[act]

        if sum(action) != 1:
            raise ValueError('Multiple input actions!')

        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: flap the bird
        if action[1] == 1:
            if self.playery > -2 * self.player_height:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True
                #SOUNDS['wing'].play()

        # check for score
        playerMidPos = self.playerx + self.player_width / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + self.pipe_width / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1
                reward = 1

        # playerIndex basex change
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        # player's movement
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
        self.playery += min(self.playerVelY, self.basey - self.playery - self.player_height)
        if self.playery < 0:
            self.playery = 0

        # move pipes to left
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = self.getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if self.upperPipes[0]['x'] < -self.pipe_width:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        # check if crash here
        isCrash= self.checkCrash({'x': self.playerx, 'y': self.playery,
                             'index': self.playerIndex},
                            self.upperPipes, self.lowerPipes)
        if isCrash:
            #SOUNDS['hit'].play()
            #SOUNDS['die'].play()
            terminated = True
            # self.__init__(self.FPS)
            # self.reset()
            reward = -1

        if self.render_mode == "human":
            self._render_frame()

        observation = self._get_obs()       
        info = self._get_info()
        truncated = None
        
        return observation, reward, terminated, truncated, info

    def draw_canvas(self):
        self.canvas.fill((0, 0, 0))
        self.canvas.blit(self.IMAGES['background'], (0,0))
        
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            self.canvas.blit(self.IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            self.canvas.blit(self.IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))
        
        self.canvas.blit(self.IMAGES['base'], (self.basex, self.basey))
        self.canvas.blit(self.IMAGES['player'][self.playerIndex],
                    (self.playerx, self.playery))
        if SHOW_SCORE:
            self.showScore(self.score)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.screenwidth, self.screenheight))
            pygame.display.set_caption('Flappy Bird')         
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
            
        self.draw_canvas()

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(self.canvas, self.canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2)
            )
    
    def showScore(self, score):
        """displays score in center of screen"""
        scoreDigits = [int(x) for x in list(str(score))]
        totalWidth = 0 # total width of all numbers to be printed

        for digit in scoreDigits:
            totalWidth += self.IMAGES['numbers'][digit].get_width()

        Xoffset = (self.screenwidth - totalWidth) / 2

        for digit in scoreDigits:
            # self.SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, self.screenheight * 0.1))
            self.canvas.blit(self.IMAGES['numbers'][digit], (Xoffset, self.screenheight * 0.1))
            Xoffset += self.IMAGES['numbers'][digit].get_width()

    def close(self):
        # if self.SCREEN is not None:
        if self.canvas is not None:
            pygame.display.quit()
            pygame.quit()
            print("flappy_bird.py: FlappyBirdEnv: close(): quit pygame")

    def getRandomPipe(self):
        """returns a randomly generated pipe"""
        # y of gap between upper and lower pipe
        gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
        index = random.randint(0, len(gapYs)-1)
        gapY = gapYs[index]

        gapY += int(BASEY * 0.2)
        pipeX = SCREENWIDTH + 10

        return [
            {'x': pipeX, 'y': gapY - self.pipe_height},  # upper pipe
            {'x': pipeX, 'y': gapY + PIPEGAPSIZE},  # lower pipe
        ]

    def checkCrash(self, player, upperPipes, lowerPipes):
        """returns True if player collders with base or pipes."""
        pi = player['index']
        player['w'] = self.IMAGES['player'][0].get_width()
        player['h'] = self.IMAGES['player'][0].get_height()

        if player['y'] <= 1:
            return True
        # if player crashes into ground
        if player['y'] + player['h'] >= BASEY - 1:
            return True
        else:

            playerRect = pygame.Rect(player['x'], player['y'],
                        player['w'], player['h'])

            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                # upper and lower pipe rects
                uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], self.pipe_width, self.pipe_height)
                lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], self.pipe_width, self.pipe_height)

                # player and upper/lower pipe hitmasks
                pHitMask = self.HITMASKS['player'][pi]
                uHitmask = self.HITMASKS['pipe'][0]
                lHitmask = self.HITMASKS['pipe'][1]

                # if bird collided with upipe or lpipe
                uCollide = self.pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
                lCollide = self.pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

                if uCollide or lCollide:
                    return True

        return False

    def pixelCollision(self, rect1, rect2, hitmask1, hitmask2):
        """Checks if two objects collide and not just their rects"""
        rect = rect1.clip(rect2)

        if rect.width == 0 or rect.height == 0:
            return False

        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y

        for x in range(rect.width):
            for y in range(rect.height):
                if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                    return True
        return False

    def getHitmask(self, image):
        """returns a hitmask using an image's alpha."""
        mask = []
        for x in range(image.get_width()):
            mask.append([])
            for y in range(image.get_height()):
                mask[x].append(bool(image.get_at((x,y))[3]))
        return mask

    def load(self):
        # path of player with different states
        PLAYER_PATH = (
                'assets/sprites/bird.png',
                'assets/sprites/bird.png',
                'assets/sprites/bird.png'
        )

        # path of background
        BACKGROUND_PATH = 'assets/sprites/background-black.png'

        # path of pipe
        PIPE_PATH = 'assets/sprites/pipe-green.png'

        IMAGES, SOUNDS, HITMASKS = {}, {}, {}

        # numbers sprites for score display
        IMAGES['numbers'] = (
            pygame.image.load('assets/sprites/0.png').convert_alpha(),
            pygame.image.load('assets/sprites/1.png').convert_alpha(),
            pygame.image.load('assets/sprites/2.png').convert_alpha(),
            pygame.image.load('assets/sprites/3.png').convert_alpha(),
            pygame.image.load('assets/sprites/4.png').convert_alpha(),
            pygame.image.load('assets/sprites/5.png').convert_alpha(),
            pygame.image.load('assets/sprites/6.png').convert_alpha(),
            pygame.image.load('assets/sprites/7.png').convert_alpha(),
            pygame.image.load('assets/sprites/8.png').convert_alpha(),
            pygame.image.load('assets/sprites/9.png').convert_alpha()
        )

        # base (ground) sprite
        IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

        # sounds
        if 'win' in sys.platform:
            soundExt = '.wav'
        else:
            soundExt = '.ogg'

    #    SOUNDS['die']    = pygame.mixer.Sound('assets/audio/die' + soundExt)
    #    SOUNDS['hit']    = pygame.mixer.Sound('assets/audio/hit' + soundExt)
    #    SOUNDS['point']  = pygame.mixer.Sound('assets/audio/point' + soundExt)
    #    SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
    #    SOUNDS['wing']   = pygame.mixer.Sound('assets/audio/wing' + soundExt)

        # select random background sprites
        IMAGES['background'] = pygame.image.load(BACKGROUND_PATH).convert()

        # select random player sprites
        IMAGES['player'] = (
            pygame.image.load(PLAYER_PATH[0]).convert_alpha(),
            pygame.image.load(PLAYER_PATH[1]).convert_alpha(),
            pygame.image.load(PLAYER_PATH[2]).convert_alpha(),
        )

        # select random pipe sprites
        IMAGES['pipe'] = (
            pygame.transform.rotate(
                pygame.image.load(PIPE_PATH).convert_alpha(), 180),
            pygame.image.load(PIPE_PATH).convert_alpha(),
        )

        # hismask for pipes
        HITMASKS['pipe'] = (
            self.getHitmask(IMAGES['pipe'][0]),
            self.getHitmask(IMAGES['pipe'][1]),
        )

        # hitmask for player
        HITMASKS['player'] = (
            self.getHitmask(IMAGES['player'][0]),
            self.getHitmask(IMAGES['player'][1]),
            self.getHitmask(IMAGES['player'][2]),
        )

        return IMAGES, SOUNDS, HITMASKS

if __name__ == "__main__":
    env = FlappyBirdEnv(FPS = 10, render_mode = "human")
    # env = FlappyBirdEnv(FPS = 10, render_mode = "rgb_array")
    state, info = env.reset()
    done = False

    # while not done:
    #     action = env.action_space.sample()  # Random action
    #     state, reward, done, truncated, info = env.step(action)
    #     env.render()

    # i = 0
    # while i < 100:
    #     while not done:
    #         action = env.action_space.sample()  # Random action
    #         state, reward, done, truncated, info = env.step(action)
    #     print(f"info: {info}")
    #     state, info = env.reset()
    #     done = False
    #     i += 1
    #     print(f"i: {i}")
    #     # env.render()


    # action_mapping = {pygame.K_UP: 1}

    # episode_over = False
    # while not episode_over:
    #     action = env.action_space.sample()  # agent policy that uses the observation and info
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     episode_over = terminated or truncated

    # Run the game loop
    quit_the_game = False
    # done = False
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_the_game = True
                break
                running = False
                # env.close()

            # Check for key presses
            if event.type == pygame.KEYDOWN and (event.key == pygame.K_SPACE or event.key == pygame.K_UP):
                print("space")
                action = 1
                break
            else:
                action = 0
                # if event.key in action_mapping:
                #     action = action_mapping[event.key]
        if quit_the_game:
            break
                    # If the episode ends, reset the environment
        if done:
            state, info = env.reset()
            # print(f"info: {info}")
            done = False
        else:
            state, reward, done, truncated, info = env.step(action)
            if done:
                print(f"info: {info}")
            
        action = 0

    env.close()
