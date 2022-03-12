import os
import pickle
import random

import settings

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Example of Q-Table implementation on a 5x5 board with 6 actions.
    Our features represent the Q-Table and consist of simple position
    on the board (row, column)

    +-------+------+------+------+-------+------+------+
    | (r,c) |  up  | down | left | right | wait | bomb |
    +-------+------+------+------+-------+------+------+
    | (0,0) | 0.50 | 0.80 | 0.95 | 1.20  | 2.50 | -3.0 |
    +-------+------+------+------+-------+------+------+
    | (0,1) | 3.10 | 1.26 | -2.5 | -1.0  | 0.50 |  0.0 |
    +-------+------+------+------+-------+------+------+
    |  ...  |  ... |  ... |  ... |  ...  |  ... |  ... |
    +-------+------+------+------+-------+------+------+
    | (4,4) | 0.10 | 2.56 | -7.0 |  0.0  | -3.5 | 10.0 |
    +-------+------+------+------+-------+------+------+
    """

    if self.train or not os.path.isfile("q_table.pt"):
        self.logger.info("Setting up Q-Table from scratch")
        self.q_table = np.zeros(((settings.ROWS-2)*(settings.COLS-2), len(ACTIONS)))
    else:
        self.logger.info("Loading saved model")
        with open("q_table.pt", "rb") as file:
            self.q_table = pickle.load(file)


def act(self, game_state: dict) -> str:
    epsilon = 0.1

    # Act randomly according to epsilon-greedy
    if self.train and random.random() < epsilon:
        chosen_action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        self.logger.info(f"Discovery time! Let's try {chosen_action}...")
        return chosen_action

    # Extract our current position and choose the action with
    # maximal value according to our Q-Table.
    state_ix = get_q_index(game_state)
    chosen_action = ACTIONS[self.q_table[state_ix].argmax()]
    self.logger.info(f"I'm going with {chosen_action}, because it's the best!")
    return chosen_action


def get_q_index(game_state):
    if game_state is None:
        return None

    col, row = game_state["self"][-1]
    return (col-1) + (settings.COLS-2) * (row-1)
