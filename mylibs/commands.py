import gymnasium as gym

try:
    import pygame
    from pygame import Surface
    from pygame.event import Event
except ImportError as e:
    raise gym.error.DependencyNotInstalled(
        'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
    ) from e

# commands_dict = {
#     # alien
#     ("alien", 1): '{' + 'space' + '}',
#     ("alien", 2): '{' + 'right' + '}',
#     ("alien", 3): '{' + 'left' + '}',
#     # breakout
#     ("breakout", 1): '{' + 'space' + '}',
#     ("breakout", 2): '{' + 'right' + '}',
#     ("breakout", 3): '{' + 'left' + '}',
#     # ms pacman
#     ("ms_pacman", 1): '{' + 'up' + '}',
#     ("ms_pacman", 2): '{' + 'right' + '}',
#     ("ms_pacman", 3): '{' + 'left' + '}',
#     ("ms_pacman", 4): '{' + 'down' + '}',
#     ("ms_pacman", 5): '{' + 'up' + '}' + '{' + 'right' + '}',
#     ("ms_pacman", 6): '{' + 'up' + '}' + '{' + 'left' + '}',
#     ("ms_pacman", 7): '{' + 'down' + '}' + '{' + 'right' + '}',
#     ("ms_pacman", 8): '{' + 'down' + '}' + '{' + 'left' + '}',
# }

commands_dict = {
    # alien
    "alient": {
        1: '{' + 'space' + '}',
        2: '{' + 'right' + '}',
        3: '{' + 'left' + '}',
    },
    # breakout
    "breakout": {
        1: '{' + 'space' + '}',
        2: '{' + 'right' + '}',
        3: '{' + 'left' + '}',
    },
    # ms pacman
    "ms_pacman": {
        1: '{' + 'up' + '}',
        2: '{' + 'right' + '}',
        3: '{' + 'left' + '}',
        4: '{' + 'down' + '}',
        5: '{' + 'up' + '}' + '{' + 'right' + '}',
        6: '{' + 'up' + '}' + '{' + 'left' + '}',
        7: '{' + 'down' + '}' + '{' + 'right' + '}',
        8: '{' + 'down' + '}' + '{' + 'left' + '}',
    },
    # flappybird
    "flappybird": {
        1: '{' + 'up' + '}',
    },
    # pong
    "pong": {
        1: '{' + 'space' + '}',
        2: '{' + 'right' + '}',
        3: '{' + 'left' + '}',
        4: '{' + 'right' + '}' + '{' + 'space' + '}',
        5: '{' + 'left' + '}' + '{' + 'space' + '}',
    },
    # space invaders
    "space_invaders": {
        1: '{' + 'space' + '}',
        2: '{' + 'up' + '}',
        3: '{' + 'right' + '}',
        4: '{' + 'left' + '}',
        5: '{' + 'down' + '}',
        6: '{' + 'up' + '}' + '{' + 'right' + '}',
        7: '{' + 'up' + '}' + '{' + 'left' + '}',
        8: '{' + 'down' + '}' + '{' + 'right' + '}',
        9: '{' + 'down' + '}' + '{' + 'left' + '}',
        10: '{' + 'up' + '}' + '{' + 'space' + '}',
        11: '{' + 'right' + '}' + '{' + 'space' + '}',
        12: '{' + 'left' + '}' + '{' + 'space' + '}',
        13: '{' + 'down' + '}' + '{' + 'space' + '}',
        14: '{' + 'up' + '}' + '{' + 'right' + '}' + '{' + 'space' + '}',
        15: '{' + 'up' + '}' + '{' + 'left' + '}' + '{' + 'space' + '}',
        16: '{' + 'down' + '}' + '{' + 'right' + '}' + '{' + 'space' + '}',
        17: '{' + 'down' + '}' + '{' + 'left' + '}' + '{' + 'space' + '}',
    },
}

keys_to_action_dict = {
    # flappybird
    "flappybird": {
        (pygame.K_UP,): 1,  # FLAP
    },
    
    # # alien
    # ("alien", 2): '{' + 'right' + '}',
    # ("alien", 3): '{' + 'left' + '}',
    
    # breakout
    "breakout": {
        # NOOP is 0, no action
        (pygame.K_SPACE,):  1,  # Fire (release ball)
        (pygame.K_RIGHT,):  2,  # Move right
        (pygame.K_LEFT,):   3,  # Move left
    },

    # ms pacman
    "ms_pacman": {
        # NOOP is 0, no action
        (pygame.K_UP,): 1,  # UP
        (pygame.K_RIGHT,): 2, # RIGHT
        (pygame.K_LEFT,): 3,  # LEFT
        (pygame.K_DOWN,): 4,  # DOWN
        (pygame.K_UP, pygame.K_RIGHT): 5,  # UPRIGHT
        (pygame.K_UP, pygame.K_LEFT): 6,  # UPLEFT
        (pygame.K_DOWN, pygame.K_RIGHT): 7,  # DOWNRIGHT
        (pygame.K_DOWN, pygame.K_LEFT): 8,  # DOWNLEFT
    },

    # pong
    "pong": {
        # NOOP is 0, no action
        (pygame.K_SPACE,):  1,  # Fire
        (pygame.K_RIGHT,): 2, # RIGHT
        (pygame.K_LEFT,): 3,  # LEFT
        (pygame.K_RIGHT, pygame.K_SPACE): 4,  # RIGHTFIRE
        (pygame.K_LEFT, pygame.K_SPACE): 5,  # LEFTFIRE
    },
        
    # space invaders
    "space_invaders": {
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
    },
}
