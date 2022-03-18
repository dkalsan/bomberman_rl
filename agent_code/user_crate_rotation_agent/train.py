from collections import namedtuple, deque

import pickle
from typing import List

import numpy as np

import events as e
from .callbacks import ACTIONS
from .callbacks import game_state_to_feature


# Events
OPTIMAL_CRATE_EVENT = "OPTIMAL_CRATE"

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
        if (
            self.state_space.get_state(old_agent_state)["compass"] == "NP" and
            self.state_space.get_state(old_agent_state)["compass_mode"] == "crate"
        ):
            events.append(OPTIMAL_CRATE_EVENT)

    self.transitions.append(Transition(old_agent_state,
                                       self_action,
                                       new_agent_state,
                                       reward_from_events(self, events)))

    # Perform a TD(N) update when the buffer fills up
    if len(self.transitions) == self.transitions.maxlen:
        rotational_update_q_table(self)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    last_agent_state = game_state_to_feature(self, last_game_state)

    self.transitions.append(Transition(last_agent_state,
                                       last_action,
                                       None,
                                       reward_from_events(self, events)))

    # Clear the buffer performing the remaining TD(N) updates
    while len(self.transitions) > 0:
        rotational_update_q_table(self)

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
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def rotational_update_q_table(self):
    """
    Implements TD(N) update for all states which are the same as
    the origin state from the first element in the buffer under
    repeated 90 degree rotations. This operation removes the first
    element from the buffer after computing the updates.

    For further details on Q-Table update, refer to update_q_table(...)
    function.
    """

    origin_state_ix, action, _, _ = self.transitions[0]

    # In the first step, the origin state is None.
    # We ignore this case.
    if origin_state_ix is None:
        self.transitions.popleft()
        return

    origin_state = self.state_space.get_state(origin_state_ix)

    # TODO: Change the dict fields to match the rotated state.
    #       Don't forget to also rotate the action!
    rotated_state_90 = dict(origin_state)
    rotated_state_90 = ...
    rotated_action_90 = ...

    # NOTE: You can also use the rotated_state_90 or any other,
    #       to create initial rotated_state_180 dictionary copy,
    #       if you find that more convenient. You might even
    #       decide to rotate one single state all along in a loop
    #       and only keep one dict called 'rotated_state'
    #       (e.g. in the loop at the end of this function).
    rotated_state_180 = dict(origin_state)
    rotated_state_180 = ...
    rotated_action_180 = ...

    rotated_state_270 = dict(origin_state)
    rotated_state_270 = ...
    rotated_action_270 = ...

    rotated_state_270 = dict(origin_state)
    rotated_state_270 = ...
    rotated_action_270 = ...

    all_states = [origin_state, rotated_state_90, rotated_state_180, rotated_state_270]
    all_actions = [action, rotated_action_90, rotated_action_180, rotated_action_270]

    # Update all rotated values
    for s, a in zip(all_states, all_actions):
        s_ix = self.state_space.get_index(s)
        update_q_table(s_ix, a)

    # Remove the transition, whose state we just processed
    self.transitions.popleft()


def update_q_table(self, origin_state_ix, action):
    """
    Implements TD(N) update, where the N is assumed
    from the current number of elements in self.transitions (buffer).
    Hence it is up to the caller to call this function when
    the buffer fills up to the required N.

    Origin state is the state for which we compute a new value
    and end state is the state we ended up in after collecting
    N rewards. This state is then used to read the Q-prediction
    for the rest of the game used in the table update.

    N must be strictly greater or equal to 1.
    """

    alpha = 0.02
    gamma = 0.9

    _, _, end_state_ix, _ = self.transitions[-1]

    # Collect rewards and discount them by gamma
    discounted_rewards = np.sum([gamma**i * s[-1] for i, s in enumerate(self.transitions)])

    old_q_value = self.q_table[origin_state_ix, ACTIONS.index(action)]

    q_remainder_estimate = 0.0 if end_state_ix is None else self.q_table[end_state_ix].max()

    updated_q_value = \
        old_q_value + alpha * (discounted_rewards
                               + gamma**len(self.transitions) * q_remainder_estimate
                               - old_q_value)

    self.q_table[origin_state_ix, ACTIONS.index(action)] = updated_q_value
