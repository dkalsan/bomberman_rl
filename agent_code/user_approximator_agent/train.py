import joblib
import itertools

import numpy as np
import pandas as pd

from collections import namedtuple, deque
from typing import List

import events as e

from .callbacks import game_state_to_feature


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Events
FOLLOWED_COMPASS_DIRECTIONS_EVENT = "FOLLOWED_COMPASS_DIRECTIONS"
# FOLLOWED_CLOSEST_COIN_EVENT = "FOLLOWED_CLOSEST_COIN"

# Temporal difference of N
TD_N = 4

# Represents the memory replay size
TRANSITION_HISTORY_SIZE = 20000

# Number of rounds before retraining estimator
RETRAIN_FREQUENCY = 5000

# The number of samples we take from the transition history
SAMPLE_SUBSET_SIZE = RETRAIN_FREQUENCY - TD_N + 1


def setup_training(self):
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.model_iteration = 1
    self.retrain_counter = 0
    assert (SAMPLE_SUBSET_SIZE <= (RETRAIN_FREQUENCY - TD_N + 1)), \
        "In the first estimator retrain there are only" \
        " (RETRAIN_FREQUENCY - TD_N + 1) valid samples in the transition" \
        " buffer, because the last TD_N did not collect all the" \
        " rewards yet."


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    old_agent_state = None
    new_agent_state = None

    if old_game_state is not None or new_game_state is None:
        old_agent_state = game_state_to_feature(self, old_game_state)
        new_agent_state = game_state_to_feature(self, new_game_state)

    """
    move_actions = ["UP", "DOWN", "LEFT", "RIGHT"]
    closest_coin_directions = ["closest_coin_up", "closest_coin_down", "closest_coin_left", "closest_coin_right"]
    if old_agent_state is not None and self_action in move_actions:
        min_key = ""
        min_dist = np.inf

        for key in closest_coin_directions:
            if old_agent_state[key][0] != -1 and old_agent_state[key][0] < min_dist:
                min_key = key
                min_dist = old_agent_state[key][0]

        if self_action == move_actions[closest_coin_directions.index(min_key)]:
            events.append(FOLLOWED_CLOSEST_COIN_EVENT)
    """

    # Rewarded for following the compass directions
    action_to_compass_direction = {
        "UP": "N",
        "DOWN": "S",
        "LEFT": "W",
        "RIGHT": "E"
    }

    if (
        old_agent_state is not None and
        self_action in action_to_compass_direction.keys() and
        old_agent_state["coin_compass"][0] == action_to_compass_direction[self_action]
    ):
        events.append(FOLLOWED_COMPASS_DIRECTIONS_EVENT)

    self.transitions.append(Transition(old_agent_state,
                                       self_action,
                                       new_agent_state,
                                       reward_from_events(self, events)))

    self.retrain_counter += 1

    if self.retrain_counter == RETRAIN_FREQUENCY and len(self.transitions) >= SAMPLE_SUBSET_SIZE:
        retrain_q_estimator(self)
        self.retrain_counter = 0


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    # NOTE: We do not retrain the estimator here, because all variables persist
    #       through multiple routes. This means that our model is retrained every
    #       RETRAIN_FREQUENCY rounds. It might happen that the model won't be retrained
    #       in the very last round, hence we lose at max RETRAIN_FREQUENCY-1 rounds.
    #       With this trade-off, we guarantee that the retrain frequency is fixed.

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    last_agent_state = game_state_to_feature(self, last_game_state)

    self.transitions.append(Transition(last_agent_state,
                                       last_action,
                                       None,
                                       reward_from_events(self, events)))


def reward_from_events(self, events: List[str]) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.MOVED_UP: -0.1,
        e.MOVED_DOWN: -0.1,
        e.MOVED_RIGHT: -0.1,
        e.MOVED_LEFT: -0.1,
        e.INVALID_ACTION: -0.6,
        e.BOMB_DROPPED: -0.5,
        e.WAITED: -0.4,
        e.CRATE_DESTROYED: 1.5,
        e.GOT_KILLED: -5,
        e.KILLED_SELF: -3,
        e.SURVIVED_ROUND: 1.5,
        e.KILLED_OPPONENT: 5,
        FOLLOWED_COMPASS_DIRECTIONS_EVENT: 0.2
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def retrain_q_estimator(self):
    # TODO: Log this
    print("Retraining")

    alpha = 0.2
    gamma = 0.9

    # NOTE: We ignore last TD_N-1 transitions, because we did not collect the
    #       rewards yet, and they aren't necessarily the terminal state,
    #       hence we can't assume a reward of 0.
    subset_indices = np.random.choice(np.arange(0, len(self.transitions)-TD_N+1),
                                      size=SAMPLE_SUBSET_SIZE,
                                      replace=False)

    X, y = [], []

    for idx in subset_indices:
        transition_subset = list(itertools.islice(self.transitions, idx, idx+TD_N))

        origin_state, action, _, _ = transition_subset[0]
        _, _, end_state, _ = transition_subset[-1]

        action_ix = self.model.actions.index(action)

        # In the first step, the origin state is None.
        # We ignore this case.
        if origin_state is None:
            continue

        # Collect rewards and discount them by gamma
        # NOTE: If there's a terminal state in any of the intermediate steps,
        #       we fill the remainder of rewards with 0 and consider
        #       the remainder of the game to also be zero
        encountered_end_state = False
        discounted_reward_list = []
        for i, s in enumerate(transition_subset):
            if not encountered_end_state:
                discounted_reward_list.append(gamma**i * s[-1])
            else:
                discounted_reward_list.append(0.0)

            if s[2] is None:
                encountered_end_state = True
        discounted_rewards = np.sum(discounted_reward_list)

        old_q_values = self.model.predict(origin_state)[0]
        old_q_value = old_q_values[action_ix]

        q_remainder_estimate = 0.0 if encountered_end_state else self.model.predict(end_state)[0].max()

        updated_q_value = \
            old_q_value + alpha * (discounted_rewards
                                   + gamma**len(transition_subset) * q_remainder_estimate
                                   - old_q_value)

        updated_q_values = old_q_values.copy()
        updated_q_values[action_ix] = updated_q_value

        # Append the origin state and new desired Q-Values to the dataset
        X.append(origin_state)
        y.append(updated_q_values)

    # Retrain Q estimator
    self.model.fit(pd.concat(X, ignore_index=True), y)

    # Store new model
    joblib.dump(self.model, "rf_model.joblib")