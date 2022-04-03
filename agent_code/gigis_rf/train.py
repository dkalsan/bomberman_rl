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
FOLLOWED_COIN_COMPASS_DIRECTIONS = "FOLLOWED_COIN_COMPASS_DIRECTIONS"
FOLLOWED_ENEMY_COMPASS_DIRECTIONS = "FOLLOWED_ENEMY_COMPASS_DIRECTIONS"
FOLLOWED_BOMB_COMPASS_DIRECTIONS = "FOLLOWED_BOMB_COMPASS_DIRECTIONS"
WAY_TO_COMPASS_NP_BOMBED = "WAY_TO_COMPASS_NP_BOMBED"
CRATES_DESTROYED_1 = "CRATES_DESTROYED_1"
CRATES_DESTROYED_2 = "CRATES_DESTROYED_2"
CRATES_DESTROYED_3TO4 = "CRATES_DESTROYED_3TO4"
CRATES_DESTROYED_5ORMORE = "CRATES_DESTROYED_5ORMORE"
ENEMY_POSSIBLY_TRAPPED = "ENEMY_POSSIBLY_TRAPPED"
IGNORED_ENEMY_POSSIBLY_TRAPPED = "IGNORED_ENEMY_POSSIBLY_TRAPPED"
STANDING_ON_TICKING_FIELD = "STANDING_ON_TICKING_FIELD"

# Temporal difference of N
TD_N = 4

# Represents the memory replay size
TRANSITION_HISTORY_SIZE = 100000

# Number of rounds before retraining estimator
RETRAIN_FREQUENCY = 2500

# The number of samples we take from the transition history
SAMPLE_SUBSET_SIZE = 50000


def setup_training(self):
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.model_iteration = 1
    self.retrain_counter = 0


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    old_agent_state = None
    new_agent_state = None

    if old_game_state is not None or new_game_state is None:
        old_agent_state = game_state_to_feature(self, old_game_state)
        new_agent_state = game_state_to_feature(self, new_game_state)

    # Custom rewards
    if old_game_state is not None:

        """
        Reward for following the compass directions
        """

        # if e.INVALID_ACTION not in events:

        #     action_to_compass_direction = {
        #         "UP": "N",
        #         "DOWN": "S",
        #         "LEFT": "W",
        #         "RIGHT": "E"
        #     }

        #     action_to_bomb_compass_direction = {
        #         "DOWN": "N",
        #         "UP": "S",
        #         "RIGHT": "W",
        #         "LEFT": "E"
        #     }

        #     if (
        #         old_agent_state is not None and
        #         self_action in action_to_compass_direction.keys()
        #     ):
        #         if old_agent_state["coin_compass"][0] == action_to_compass_direction[self_action]:
        #             events.append(FOLLOWED_COIN_COMPASS_DIRECTIONS)

        #         if old_agent_state["enemy_compass"][0] == action_to_compass_direction[self_action]:
        #             events.append(FOLLOWED_ENEMY_COMPASS_DIRECTIONS)

        #         if old_agent_state["bomb_compass"][0] == action_to_bomb_compass_direction[self_action]:
        #             events.append(FOLLOWED_BOMB_COMPASS_DIRECTIONS)

        """
        Reward proportional to number of crates destroyed
        """

        if self_action == "BOMB" and old_game_state["self"][2] is True:
            num_explodable_crates = old_agent_state["num_explodable_crates"][0]
            if num_explodable_crates == 1:
                events.append(CRATES_DESTROYED_1)
            elif num_explodable_crates == 2:
                events.append(CRATES_DESTROYED_2)
            elif 3 <= num_explodable_crates <= 4:
                events.append(CRATES_DESTROYED_3TO4)
            elif num_explodable_crates >= 5:
                events.append(CRATES_DESTROYED_5ORMORE)

        """
        Reward if an enemy is in a deadend and agent moves towards him.
        Penalize for ignoring the information. The goal is to incentivize
        our agent to move into deadends until next to an enemy and then setting
        a bomb.
        """

        routes = ["tile_up", "tile_down", "tile_right", "tile_left"]

        for enemy_dir, route, action in zip(["N", "S", "E", "W"], routes, ['UP', 'DOWN', 'RIGHT', 'LEFT']):
            if (
                old_agent_state["enemy_compass"][0] == enemy_dir and
                old_agent_state[route][0] in ["shallow-deadend", "l-shallow-deadend", "deep-deadend"]
            ):
                if self_action == action and e.INVALID_ACTION not in events:
                    events.append(ENEMY_POSSIBLY_TRAPPED)
                else:
                    events.append(IGNORED_ENEMY_POSSIBLY_TRAPPED)

            if (
                old_agent_state["enemy_compass"][0] == enemy_dir and
                old_agent_state[route][0] == "explodable" and
                self_action != "BOMB" and
                old_game_state["self"][2] is True
            ):
                events.append(IGNORED_ENEMY_POSSIBLY_TRAPPED)

        """
        Reward for destroying crates to get to the compass location.
        This makes sense for compass modes where Mannhattan distance is used.
        """

        if self_action == "BOMB" and old_game_state["self"][2] is True:
            others = [xy for (n, s, b, xy) in old_game_state['others']]
            for compass_direction, shift in zip(["N", "S", "E", "W"], [[0, -1], [0, 1], [1, 0], [-1, 0]]):
                x, y = np.array(old_game_state["self"][3]) + shift
                if (
                    (old_agent_state["enemy_compass"][0] == compass_direction or
                        old_agent_state["coin_compass"][0] == compass_direction) and
                    (old_game_state["field"][x, y] == 1 or
                        (x, y) in others)
                ):
                    events.append(WAY_TO_COMPASS_NP_BOMBED)

        """
        Penalize for standing on a ticking field
        """

        if 0 <= old_agent_state["tile_current"][0] < 5:
            events.append(STANDING_ON_TICKING_FIELD)

    """
    TODO Ideas: * Add "bomb available" as a feature
                * Add reward for waiting next to an explosion if the compass is pointing in that direction
                * Reward for blowing up crates on the way to compass?
                * Add raw distance features?
    """

    self.transitions.append(Transition(old_agent_state,
                                       self_action,
                                       new_agent_state,
                                       reward_from_events(self, events)))

    self.retrain_counter += 1

    if self.retrain_counter >= RETRAIN_FREQUENCY and len(self.transitions) >= (SAMPLE_SUBSET_SIZE + TD_N - 1):
        retrain_q_estimator(self)
        self.retrain_counter = 0


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    # NOTE: We do not retrain the estimator here, because all variables persist
    #       through multiple rounds. This means that our model is retrained every
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
        e.BOMB_DROPPED: -0.3,
        e.WAITED: -0.4,
        e.GOT_KILLED: -5,
        e.KILLED_SELF: -3,
        e.SURVIVED_ROUND: 1.5,
        e.KILLED_OPPONENT: 5,
        # FOLLOWED_COIN_COMPASS_DIRECTIONS: 0.1,
        # FOLLOWED_ENEMY_COMPASS_DIRECTIONS: 0.1,
        # FOLLOWED_BOMB_COMPASS_DIRECTIONS: 0.2,
        WAY_TO_COMPASS_NP_BOMBED: 0.5,
        CRATES_DESTROYED_1: 0.5,
        CRATES_DESTROYED_2: 0.7,
        CRATES_DESTROYED_3TO4: 1.1,
        CRATES_DESTROYED_5ORMORE: 1.4,
        ENEMY_POSSIBLY_TRAPPED: 0.2,
        IGNORED_ENEMY_POSSIBLY_TRAPPED: -0.2,
        STANDING_ON_TICKING_FIELD: -0.2
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def retrain_q_estimator(self):
    self.logger.debug(f"Retraining model (iteration {self.model_iteration})...")
    print(f"Retraining model (iteration {self.model_iteration})...")
    self.model_iteration += 1

    alpha = 0.1
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
