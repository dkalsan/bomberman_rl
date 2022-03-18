import os
import pickle
import random

import settings

from .utils import StateSpace

import numpy as np


ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']


def setup(self):

    # TODO: Crate is opponent + crate
    self.state_space = StateSpace({
        "tile_up": ['free_tile', 'coin', 'invalid', 'crate', 'danger'],
        "tile_down": ['free_tile', 'coin', 'invalid', 'crate', 'danger'],
        "tile_left": ['free_tile', 'coin', 'invalid', 'crate', 'danger'],
        "tile_right": ['free_tile', 'coin', 'invalid', 'crate', 'danger'],
        "compass": ["N", "S", "W", "E", "NP"],
        "compass_mode": ["coin", "crate", "escape"]
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
        chosen_action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
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

    """
    Precompute information that is commonly used
    """

    x, y = game_state['self'][3]
    arena = game_state['field']
    bombs = game_state['bombs']
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    explosion_map = game_state['explosion_map']
    bomb_map = compute_bomb_map(arena, bombs)

    """
    Compute 'tile_*' features
    """

    # Check if there are free tiles in the adjacent space
    # (x, y) is the current position
    directions = [(x, y-1),  # UP
                  (x, y+1),  # DOWN
                  (x-1, y),  # LEFT
                  (x+1, y)]  # RIGHT

    feature_keys = ["tile_up", "tile_down", "tile_left", "tile_right"]
    for (key, d) in zip(feature_keys, directions):
        if bomb_map[d] != 5 or explosion_map[d] != 0:
            agent_state[key] = "danger"
        elif d in coins:
            agent_state[key] = "coin"
        elif arena[d] == 0:
            agent_state[key] = "free_tile"
        elif arena[d] == 1:
            agent_state[key] = "crate"
        elif arena[d] == -1:
            agent_state[key] = "invalid"  # TODO: danger could also be invalid?

    """
    Activate correct compass mode and directions
    """

    # Reach hyperparameters
    coin_reach = min((game_state["step"] // 40) + 4, 7)
    crate_reach = min((game_state["step"] // 40) + 1, 5)
    # enemy_reach = min((game_state["step"] // 100) + 2, 5)

    if bomb_map[x, y] != 5:
        """
        Compute 'ticking_bomb' feature,
        which points the compass toward the bomb
        """

        # Set compass mode to escape
        agent_state["compass_mode"] = "escape"

        my_position = (x, y)
        ticking_bomb_direction = compute_ticking_bomb_feature(bombs, my_position)
        agent_state["compass"] = ticking_bomb_direction

    elif (
        len(coins) > 0
        and (np.max(np.abs(np.array(coins) - (x, y)), axis=1) <= coin_reach).any()
    ):
        """
        Compute 'closest_coin' feature
        """

        # Set compass mode to coin collecting
        agent_state["compass_mode"] = "coin"

        # Look for the shortest distance between the agent and coins
        # Target feature
        free_space = arena == 0
        for o in others:
            free_space[o] = False
        targets = coins

        best_direction = look_for_targets(free_space, (x, y), targets, self.logger)

        if best_direction == (x, y-1):
            agent_state["compass"] = "N"
        elif best_direction == (x, y+1):
            agent_state["compass"] = "S"
        elif best_direction == (x-1, y):
            agent_state["compass"] = "W"
        elif best_direction == (x+1, y):
            agent_state["compass"] = "E"
        else:
            agent_state["compass"] = "NP"

    else:
        """
        Compute 'closest_crate' feature
        """

        # Set compass mode to coin collecting
        agent_state["compass_mode"] = "crate"

        best_crate_direction = compute_crate_feature(arena, bomb_map, others, (x, y), crate_reach)
        agent_state["compass"] = best_crate_direction

    """
    Find the index of the computed state
    """

    agent_state_ix = self.state_space.get_index(agent_state)

    return agent_state_ix


def compute_bomb_map(arena, bombs):
    bomb_map = np.ones(arena.shape) * 5
    for (x_bomb, y_bomb), t in bombs:

        x_blast, y_blast = compute_blast(arena,
                                         (x_bomb, y_bomb),
                                         settings.BOMB_POWER)

        bomb_map[x_blast, y_bomb] = t
        bomb_map[x_bomb, y_blast] = t

    return bomb_map


def compute_blast(arena, bomb_position, bomb_power):
    x_bomb, y_bomb = bomb_position

    l_spread = -bomb_power
    r_spread = bomb_power + 1
    if arena[x_bomb-1, y_bomb] == -1:  # Wall to the left
        l_spread = 0
    if arena[x_bomb+1, y_bomb] == -1:  # Wall to the right
        r_spread = 1

    u_spread = -bomb_power
    d_spread = bomb_power + 1
    if arena[x_bomb, y_bomb-1] == -1:  # Wall above
        u_spread = 0
    if arena[x_bomb, y_bomb+1] == -1:  # Wall below
        d_spread = 1

    x_blast_spread = np.expand_dims(np.arange(l_spread, r_spread), axis=0)
    y_blast_spread = np.expand_dims(np.arange(u_spread, d_spread), axis=0)

    x_blast = np.array([[x_bomb]]) + x_blast_spread
    x_blast = x_blast[(x_blast > 0) & (x_blast < arena.shape[0]-1)]

    y_blast = np.array([[y_bomb]]) + y_blast_spread
    y_blast = y_blast[(y_blast > 0) & (y_blast < arena.shape[1]-1)]

    return x_blast, y_blast


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


def compute_num_crates_exploded(arena, bomb_location, bomb_power):
    crates = arena == 1
    crates_exploded = 0

    x_bomb, y_bomb = bomb_location

    x_blast, y_blast = compute_blast(arena,
                                     (x_bomb, y_bomb),
                                     bomb_power)

    crates_exploded += np.count_nonzero(crates[x_blast, y_bomb])
    crates_exploded += np.count_nonzero(crates[x_bomb, y_blast])

    return crates_exploded


def compute_crate_feature(arena, bomb_map, others, my_location, reach):
    """
    Looks at all possible locations in a given vicinity controlled
    by hyperparameter 'reach', where a bomb could be placed. The
    location where the most crates can be exploded is then computed,
    where crates which are already about to be blown up by a ticking bomb
    are excluded.
    """

    possible_locations = (arena == 0) & (bomb_map == 5)
    for o in others:
        possible_locations[o] = False

    # We only look in our nearby area ({reach} number of tiles to each direction)
    x, y = my_location
    possible_locations[min(x+reach+1, possible_locations.shape[0]):, :] = False
    possible_locations[:max(x-reach, 0), :] = False
    possible_locations[:, min(y+reach+1, possible_locations.shape[1]):] = False
    possible_locations[:, :max(y-reach, 0)] = False

    # TODO: Discount by steps
    optimal_location = (-1, -1)
    max_crates_exploded = 0
    for px, py in np.transpose(possible_locations.nonzero()):

        # Consider only positions directly next to crates
        if (
            arena[px-1, py] == 1 or arena[px+1, py] == 1
            or arena[px, py-1] == 1 or arena[px, py+1] == 1
        ):
            crates_exploded = compute_num_crates_exploded(arena, (px, py), settings.BOMB_POWER)

            if crates_exploded > max_crates_exploded:
                optimal_location = (px, py)
                max_crates_exploded = crates_exploded

    # Return mannhattan directions to optimal location
    if max_crates_exploded == 0:
        return "NP"  # TODO: We had the 6th option 'none' here before, rethink this

    if optimal_location[1] > my_location[1]:
        return "S"
    elif optimal_location[1] < my_location[1]:
        return "N"

    if optimal_location[0] > my_location[0]:
        return "E"
    elif optimal_location[0] < my_location[0]:
        return "W"
    else:
        return "NP"


def compute_ticking_bomb_feature(bombs, my_position):
    if len(bombs) == 0:
        return "NP"

    reach = 3
    bomb_positions = np.array([b[0] for b in bombs])
    x_bomb_positions, y_bomb_positions = bomb_positions[:, 0], bomb_positions[:, 1]

    # Mask bombs that are in the same row or column as our agent
    mask_x = np.abs(x_bomb_positions - my_position[0]) == 0
    mask_y = np.abs(y_bomb_positions - my_position[1]) == 0
    dangerous_bomb_positions = bomb_positions[mask_x | mask_y, :]

    if len(dangerous_bomb_positions) == 0:
        return "NP"

    closest_dangerous_bomb = dangerous_bomb_positions[
        np.argmin(np.sum(np.abs(dangerous_bomb_positions - my_position), axis=1), axis=0)
    ]

    diff = closest_dangerous_bomb - my_position

    # We only care about bombs {reach} tiles away from us
    if np.max(np.abs(diff)) > reach:
        return "NP"

    # Compute in which direction the bomb is, relative to us
    if diff[0] < 0:  # x-axis
        return "W"
    elif diff[0] > 0:  # x-axis
        return "E"
    elif diff[1] < 0:  # y-axis
        return "N"
    elif diff[1] > 0:  # y-axis
        return "S"
    else:
        return "NP"  # We are standing on it
