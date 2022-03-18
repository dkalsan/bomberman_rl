from collections import namedtuple, deque

import pickle
from typing import List

import numpy as np

import events as e
from .callbacks import ACTIONS
from .callbacks import game_state_to_feature


# Events
OPTIMAL_CRATE_EVENT = "OPTIMAL_CRATE"
WAY_TO_OPTIMAL_CRATE_BOMBED_EVENT = "WAY_TO_OPTIMAL_CRATE_BOMBED"

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Represents 'N' to our TD(N) learning. Must be bigger or equal to 1.
TRANSITION_HISTORY_SIZE = 4


def setup_training(self):
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    old_agent_state = None
    new_agent_state = None

    if old_game_state is not None or new_game_state is None:
        old_agent_state = game_state_to_feature(self, old_game_state)
        new_agent_state = game_state_to_feature(self, new_game_state)

    # TODO: Add reward proportional to boxes destroyed?
    if self_action == "BOMB" and old_agent_state is not None:

        # Rewarded for destroying optimal location
        if (
            self.state_space.get_state(old_agent_state)["compass"] == "NP" and
            self.state_space.get_state(old_agent_state)["compass_mode"] == "crate"
        ):
            events.append(OPTIMAL_CRATE_EVENT)

        # Rewarded for destroying boxes, when they are in the
        # way of the optimal location
        for compass_direction, shift in zip(["N", "S", "E", "W"], [[0, -1], [0, 1], [1, 0], [-1, 0]]):
            if (
                self.state_space.get_state(old_agent_state)["compass"] == compass_direction and
                old_agent_state["arena"][np.array(old_agent_state["self"]) + shift] == 1
            ):
                events.append(WAY_TO_OPTIMAL_CRATE_BOMBED_EVENT)

        # 

    self.transitions.append(Transition(old_agent_state,
                                       self_action,
                                       new_agent_state,
                                       reward_from_events(self, events)))

    # Perform a TD(N) update when the buffer fills up
    if len(self.transitions) == self.transitions.maxlen:
        update_q_table(self)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    last_agent_state = game_state_to_feature(self, last_game_state)

    self.transitions.append(Transition(last_agent_state,
                                       last_action,
                                       None,
                                       reward_from_events(self, events)))

    # Clear the buffer performing the remaining TD(N) updates
    while len(self.transitions) > 0:
        update_q_table(self)

    with open("q_table.pt", "wb") as file:
        pickle.dump(self.q_table, file)


# TODO: Adjust rewards, so that it makes sense for him
#       to make some steps towards the optimal crate
#       and not get killed
def reward_from_events(self, events: List[str]) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.MOVED_UP: -0.1,
        e.MOVED_DOWN: -0.1,
        e.MOVED_RIGHT: -0.1,
        e.MOVED_LEFT: -0.1,
        e.INVALID_ACTION: -0.6,
        e.BOMB_DROPPED: -0.8,
        e.WAITED: -0.4,
        e.CRATE_DESTROYED: 1.5,
        e.GOT_KILLED: -5,
        e.KILLED_SELF: -3,
        e.SURVIVED_ROUND: 1.5,
        OPTIMAL_CRATE_EVENT: 0.7,
        WAY_TO_OPTIMAL_CRATE_BOMBED_EVENT: 0.3
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def update_q_table(self):
    """
    Implements TD(N) update, where the N is assumed
    from the current number of elements in self.transitions (buffer).
    Hence it is up to the caller to call this function when
    the buffer fills up to the required N. This operation removes
    the first element in the buffer after computing the update.

    Origin state is the state for which we compute a new value
    and end state is the state we ended up in after collecting
    N rewards. This state is then used to read the Q-prediction
    for the rest of the game used in the table update.

    N must be strictly greater or equal to 1.
    """

    alpha = 0.05
    gamma = 0.9

    origin_state_ix, action, _, _ = self.transitions[0]
    _, _, end_state_ix, _ = self.transitions[-1]

    # In the first step, the origin state is None.
    # We ignore this case.
    if origin_state_ix is None:
        self.transitions.popleft()
        return

    # Collect rewards and discount them by gamma
    discounted_rewards = np.sum([gamma**i * s[-1] for i, s in enumerate(self.transitions)])

    old_q_value = self.q_table[origin_state_ix, ACTIONS.index(action)]

    q_remainder_estimate = 0.0 if end_state_ix is None else self.q_table[end_state_ix].max()

    updated_q_value = \
        old_q_value + alpha * (discounted_rewards
                               + gamma**len(self.transitions) * q_remainder_estimate
                               - old_q_value)

    self.q_table[origin_state_ix, ACTIONS.index(action)] = updated_q_value

    # Remove the transition, whose state we just processed
    self.transitions.popleft()
