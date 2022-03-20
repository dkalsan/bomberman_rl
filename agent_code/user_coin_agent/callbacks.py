import os
import pickle
import random

import settings

from .utils import StateSpace

import numpy as np


ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']


def setup(self):

    self.state_space = StateSpace({
        "tile_up": ['free_tile', 'coin', 'invalid'],
        "tile_down": ['free_tile', 'coin', 'invalid'],
        "tile_left": ['free_tile', 'coin', 'invalid'],
        "tile_right": ['free_tile', 'coin', 'invalid'],
        "closest_coin": ["up", "down", "left", "right", "empty"]
    })

    if self.train or not os.path.isfile("q_table.pt"):
        self.logger.info("Setting up Q-Table from scratch")
        self.q_table = None
    else:
        self.logger.info("Loading saved model")
        with open("q_table.pt", "rb") as file:
            self.q_table = pickle.load(file)


def act(self, game_state: dict) -> str:

    agent_state = game_state_to_feature(self, game_state)

    if self.q_table is None:
        self.q_table = np.zeros((len(self.state_space), len(ACTIONS)))

    # Decaying epsilon
    epsilon_initial = 1.0
    epsilon_decay = 0.999
    epsilon_min = 0.01
    epsilon = max(epsilon_min,
                  epsilon_initial * epsilon_decay**(game_state["round"]-1))

    # Act randomly according to epsilon-greedy
    if self.train and random.random() < epsilon:
        chosen_action = np.random.choice(ACTIONS, p=[.25, .25, .25, .25])
        self.logger.info(f"Discovery time! Let's try {chosen_action}...")
        return chosen_action  # Explore action space
    else:
        chosen_action = ACTIONS[np.argmax(self.q_table[agent_state])]
        self.logger.info(f"I'm going with {chosen_action}, because it's the best!")
        return chosen_action  # Exploit learned values


def get_q_index(game_state):
    if game_state is None:
        return None

    col, row = game_state["self"][-1]
    return (col-1) + (settings.COLS-2) * (row-1)


def game_state_to_feature(self, game_state):
    if game_state is None:
        return None

    agent_state = dict()

    # Gather information about the game state
    x, y = game_state['self'][3]
    arena = game_state['field']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # Check if there are free tiles in the adjacent space
    # (x, y) is the current position
    directions = [(x, y-1),  # UP
                  (x, y+1),  # DOWN
                  (x-1, y),  # LEFT
                  (x+1, y)]  # RIGHT

    feature_keys = ["tile_up", "tile_down", "tile_left", "tile_right"]
    for (key, d) in zip(feature_keys, directions):
        if d in coins:
            agent_state[key] = "coin"
        elif arena[d] == 0:
            agent_state[key] = "free_tile"
        elif arena[d] == -1:
            agent_state[key] = "invalid"

    # Look for the shortest distance between the agent and coins
    # Target feature
    free_space = arena == 0
    for o in others:
        free_space[o] = False
    targets = coins

    best_direction = look_for_targets(free_space, (x, y), targets, self.logger)

    if best_direction == (x, y-1):
        agent_state["closest_coin"] = "up"
    elif best_direction == (x, y+1):
        agent_state["closest_coin"] = "down"
    elif best_direction == (x-1, y):
        agent_state["closest_coin"] = "left"
    elif best_direction == (x+1, y):
        agent_state["closest_coin"] = "right"
    else:
        agent_state["closest_coin"] = "empty"

    # NOTE: How do we index the action and then next_tile_state?

    # Find index of the current state
    agent_state_ix = self.state_space.get_index(agent_state)

    return agent_state_ix


def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of the closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards the closest target or towards tile closest to any target.
    """
    if len(targets) == 0:
        return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        random.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]