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
                "coin_compass": ["N", "S", "W", "E", "NP"]
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

    """
    Compute distance to closest coin in each direction
    """

    free_space = arena == 0
    for o in others:
        free_space[o] = False
    targets = coins

    best_direction, _, dist = breadth_first_search(free_space, (x, y), targets)

    if best_direction == (x, y-1):
        agent_state["coin_compass"] = ["N"]
    elif best_direction == (x, y+1):
        agent_state["coin_compass"] = ["S"]
    elif best_direction == (x-1, y):
        agent_state["coin_compass"] = ["W"]
    elif best_direction == (x+1, y):
        agent_state["coin_compass"] = ["E"]
    else:
        agent_state["coin_compass"] = ["NP"]

    # agent_state["coin_distance"] = [dist]

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