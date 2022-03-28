import os
import random
import joblib

import numpy as np
import pandas as pd

from typing import List, Tuple

import settings

from .model import RFAgent


def setup(self):

    if self.train or not os.path.isfile("rf_model.joblib"):
        self.logger.info("Initializing RF model from scratch")
        self.model = RFAgent(
            categorical_features={
                "coin_compass": ["N", "S", "W", "E", "NP"],
                "enemy_compass": ["N", "S", "W", "E", "NP"],
                "bomb_compass": ["N", "S", "W", "E", "NP"],
                "tile_up": ["loop", "shallow-deadend", "l-shallow-deadend", "deep-deadend",
                            "coin", "invalid", "explodable"],
                "tile_down": ["loop", "shallow-deadend", "l-shallow-deadend", "deep-deadend",
                              "coin", "invalid", "explodable"],
                "tile_left": ["loop", "shallow-deadend", "l-shallow-deadend", "deep-deadend",
                              "coin", "invalid", "explodable"],
                "tile_right": ["loop", "shallow-deadend", "l-shallow-deadend", "deep-deadend",
                               "coin", "invalid", "explodable"]
            },
            actions=['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB'],
            config={
                "n_estimators": 1,
                "n_jobs": 1
            }
        )
    else:
        self.logger.info("Loading saved RF model")
        self.model = joblib.load("rf_model.joblib")


def act(self, game_state: dict) -> str:

    # Returns a pandas dataframe
    agent_state = game_state_to_feature(self, game_state)

    # Decaying epsilon
    # NOTE: Maybe we should decay epsilon also based on step,
    #       otherwise the agent is mostly exploring only around
    #       the starting positions (because he might kill himself
    #       before coming to the states more common further in the
    #       game)
    epsilon_initial = 1.0
    epsilon_decay = 0.999
    epsilon_min = 0.01
    epsilon = max(epsilon_min,
                  epsilon_initial * epsilon_decay**(game_state["round"]-1))

    # Act randomly according to epsilon-greedy
    if self.train and random.random() < epsilon:
        chosen_action = np.random.choice(self.model.actions, p=[.2, .2, .2, .2, .1, .1])
        self.logger.debug(f"[{game_state['round']:06d}|{game_state['step']:03d}]"
                          f" Agent state: {agent_state.to_dict()}")
        self.logger.info(f"[{game_state['round']:06d}|{game_state['step']:03d}] Exploring {chosen_action}")
        return chosen_action  # Explore action space
    else:
        chosen_action = self.model.predict_action(agent_state)[0]
        self.logger.debug(f"[{game_state['round']:06d}|{game_state['step']:03d}]"
                          f" Agent state: {agent_state.to_dict()}")
        self.logger.debug(f"[{game_state['round']:06d}|{game_state['step']:03d}]"
                          f" QTable: {self.model.predict(agent_state)[0]}")
        self.logger.debug(f"[{game_state['round']:06d}|{game_state['step']:03d}]"
                          f" Action {chosen_action}")
        return chosen_action  # Exploit learned values


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
    bomb_locations = [b[0] for b in bombs]

    """
    Compute direction to closest coin
    """

    free_space = arena == 0
    for o in others:
        free_space[o] = False
    for b in bomb_locations:
        free_space[b] = False
    targets = coins

    # If there is no coin, the model should learn to read that from "NP"
    best_direction_coin, _, coin_dist = breadth_first_search(free_space, (x, y), targets)

    if best_direction_coin == (x, y-1):
        agent_state["coin_compass"] = ["N"]
    elif best_direction_coin == (x, y+1):
        agent_state["coin_compass"] = ["S"]
    elif best_direction_coin == (x-1, y):
        agent_state["coin_compass"] = ["W"]
    elif best_direction_coin == (x+1, y):
        agent_state["coin_compass"] = ["E"]
    else:
        agent_state["coin_compass"] = ["NP"]

    """
    Number of crates exploded feature
    """

    num_explodable_crates = compute_num_crates_exploded(arena, (x, y), settings.BOMB_POWER)
    agent_state["num_explodable_crates"] = [num_explodable_crates]

    """
    Compute direction to closest enemy
    """

    # If there are no enemies, the model should learn to read that from "NP"
    best_direction_enemy, _, enemy_dist = breadth_first_search(arena == 0, (x, y), others)

    if best_direction_enemy == (x, y-1):
        agent_state["enemy_compass"] = ["N"]
    elif best_direction_enemy == (x, y+1):
        agent_state["enemy_compass"] = ["S"]
    elif best_direction_enemy == (x-1, y):
        agent_state["enemy_compass"] = ["W"]
    elif best_direction_enemy == (x+1, y):
        agent_state["enemy_compass"] = ["E"]
    else:
        agent_state["enemy_compass"] = ["NP"]

    """
    Compute difference in distance between enemy and coin.
    """

    # The model can see which one is closer to it
    # agent_state["coin_enemy_dist_diff"] = [coin_dist - enemy_dist]

    # Useful for picking up coins in dead-ends.
    # If greater than 0, our agent can get in and out
    # of the dead end before an enemy can reach the entrance.
    # agent_state["deadend_coin_enemy_dist_diff"] = [2*coin_dist - enemy_dist]

    """
    Compute deadend feature for trapping and escape purposes.
    """

    # Check if there are free tiles in the adjacent space
    # (x, y) is the current position
    directions = [(x, y-1),  # UP
                  (x, y+1),  # DOWN
                  (x-1, y),  # LEFT
                  (x+1, y)]  # RIGHT

    feature_keys = ["tile_up", "tile_down", "tile_left", "tile_right"]
    for (key, d) in zip(feature_keys, directions):

        if d in coins:
            agent_state[key] = ["coin"]

        # Crates and opponents
        elif (arena[d] == 1) or (d in others):
            agent_state[key] = ["explodable"]

        # Free tile, compute the route feature
        elif (arena[d] == 0) and not (d in bomb_locations):
            furthest_node, dist = deadend_breadth_first_search(free_space, (x, y), d)

            if furthest_node is None:
                agent_state[key] = ["invalid"]
            elif furthest_node == (x, y):
                agent_state[key] = ["loop"]
            elif dist <= 3:
                if furthest_node[0] == x or furthest_node[1] == y:
                    agent_state[key] = ["shallow-deadend"]
                else:
                    agent_state[key] = ["l-shallow-deadend"]
            else:
                agent_state[key] = ["deep-deadend"]

        # Not walkable (wall + bomb)
        else:
            agent_state[key] = ["invalid"]

    """
    Compute direction to the bomb if we are in blast-zone,
    for escape purposes
    """

    ticking_bomb_direction = compute_ticking_bomb_feature(bombs, (x, y))
    agent_state["bomb_compass"] = [ticking_bomb_direction]

    """
    Compute danger level for current and neighboring tiles
    """
    # {0, 1, 2, 3} bomb ticking
    # {-1} how many more steps explosion is present
    # {5} no bombs or explosions

    danger_map = bomb_map.copy()
    danger_map[explosion_map > 0] = -explosion_map[explosion_map > 0]

    feature_keys = ["tile_up_danger", "tile_down_danger", "tile_left_danger", "tile_right_danger"]
    for (key, d) in zip(feature_keys, directions):
        agent_state[key] = [danger_map[d]]

    agent_state["tile_current"] = [danger_map[(x, y)]]

    """
    Create a pd.DataFrame which can then be fed directly into the model
    """

    return pd.DataFrame.from_dict(agent_state)


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


def breadth_first_search(free_space, start: Tuple[int, int], targets: List[Tuple[int, int]]):
    """
    Finds the first target reachable through tiles denoted as free_space.
    Returns the direction node next to the start node, which the agent
    should take to reach the closest target node. Also returns the position
    of the closest target node and the distance to it. If no target can be
    reached, then we return Mannhattan directions and distance to the closest
    target.
    """

    if len(targets) == 0:
        return (None, None, 0)

    # Initialization
    queue = [start]
    explored = {start: None}

    # Find the first target using BFS
    closest_node = None
    while len(queue) > 0:
        current_node = queue.pop(0)

        if current_node in targets:
            closest_node = current_node
            break

        x, y = current_node
        neighbor_nodes = [(x, y+1), (x, y-1), (x-1, y), (x+1, y)]
        walkable_neighbour_nodes = [n for n in neighbor_nodes if free_space[n]]

        for neighbor_node in walkable_neighbour_nodes:
            if neighbor_node not in explored.keys():
                queue.append(neighbor_node)
                explored[neighbor_node] = current_node

    # No target can be reached, we return
    # Mannhattan direction and distance to
    # closest target.
    if closest_node is None:
        dists = np.sum(np.abs((np.array(targets) - start)), axis=1)
        closest_node_ix = np.argmin(dists)
        closest_node = targets[closest_node_ix]
        dist = dists[closest_node_ix]

        if closest_node[1] > start[1]:
            direction_node = (start[0], start[1]+1)
        elif closest_node[1] < start[1]:
            direction_node = (start[0], start[1]-1)
        elif closest_node[0] > start[0]:
            direction_node = (start[0]+1, start[1])
        elif closest_node[0] < start[0]:
            direction_node = (start[0]-1, start[1])
        else:
            # This should never be the case anyways
            direction_node = None
            closest_node = None
            dist = 0

    # We are standing on a coin
    elif closest_node == start:
        dist = 0
        direction_node = start

    # Target can be reached, backtrack it
    else:
        dist = 1
        direction_node = closest_node
        while explored[direction_node] != start:
            direction_node = explored[direction_node]
            dist += 1

    return (direction_node, closest_node, dist)


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


def deadend_breadth_first_search(free_space, my_location: Tuple[int, int], start: Tuple[int, int]):
    """
    Adaptation of breadth_first_search for evaluating dead ends for a
    given neighbouring tile.

    my_location: where the agent is located at the time of function call
    start: one of the neighbouring fields of the agent (up, down, left, right)

    We return either:
    - my_location, if there is a loop, together with its distance (length)
    - current_node, if there is no loop, which represents the furthest
      reachable tile in the deadend, and the distance to it.
    """

    # This direction is not walkable
    if not free_space[start]:
        return (None, -1)

    # Initialization
    queue = [start]
    explored = {start: None}

    # Find the first target using BFS
    while len(queue) > 0:
        current_node = queue.pop(0)

        # We found a loop
        if current_node == my_location:
            break

        x, y = current_node
        neighbor_nodes = [(x, y+1), (x, y-1), (x-1, y), (x+1, y)]
        walkable_neighbour_nodes = [n for n in neighbor_nodes if free_space[n]]

        # The search shouldn't go directly back to the my_location
        # in the first iteration.
        if len(explored) == 1 and my_location in walkable_neighbour_nodes:
            walkable_neighbour_nodes.remove(my_location)

        for neighbor_node in walkable_neighbour_nodes:
            if neighbor_node not in explored.keys():
                queue.append(neighbor_node)
                explored[neighbor_node] = current_node

    # Backtrack the location to compute distance
    dist = 1
    direction_node = current_node
    while explored[direction_node] is not None:
        direction_node = explored[direction_node]
        dist += 1

    return (current_node, dist)


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
