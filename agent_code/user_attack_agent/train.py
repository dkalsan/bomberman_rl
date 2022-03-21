from collections import namedtuple, deque

import pickle
from typing import List
from pathlib import Path

import numpy as np

import events as e
from .callbacks import ACTIONS
from .callbacks import game_state_to_feature


# Events
OPTIMAL_CRATE_EVENT = "OPTIMAL_CRATE"
WAY_TO_COMPASS_NP_BOMBED_EVENT = "WAY_TO_COMPASS_NP_BOMBED"
FOLLOWED_COMPASS_DIRECTIONS_EVENT = "FOLLOWED_COMPASS_DIRECTIONS"
COMPASS_NP_NEXT_TO_ENEMY_EVENT = "COMPASS_NP_NEXT_TO_ENEMY"

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

    # Rewarded for following the compass directions
    action_to_compass_direction = {
        "UP": "N",
        "DOWN": "S",
        "LEFT": "W",
        "RIGHT": "E"
    }

    if (
        self_action in action_to_compass_direction.keys() and
        self_action == action_to_compass_direction[self_action]
    ):
        events.append(FOLLOWED_COMPASS_DIRECTIONS_EVENT)

    if self_action == "BOMB" and old_agent_state is not None:

        # TODO: Add reward proportional to boxes destroyed?
        #       we would need a feature describing this.
        ...

        # Rewarded for destroying optimal location
        if (
            self.state_space.get_state(old_agent_state)["compass"] == "NP" and
            self.state_space.get_state(old_agent_state)["compass_mode"] == "crate"
        ):
            events.append(OPTIMAL_CRATE_EVENT)

        # Rewarded for destroying boxes to get to the optimal location
        # (this can be compass_mode "crate", "attack")
        # This makes sense for compass modes where Mannhattan distance is used.
        if (
            self.state_space.get_state(old_agent_state)["compass_mode"] == "coin" or
            self.state_space.get_state(old_agent_state)["compass_mode"] == "attack"
        ):
            for compass_direction, shift in zip(["N", "S", "E", "W"], [[0, -1], [0, 1], [1, 0], [-1, 0]]):
                x, y = np.array(old_game_state["self"][3]) + shift
                if (
                    self.state_space.get_state(old_agent_state)["compass"] == compass_direction and
                    old_game_state["field"][x, y] == 1
                ):
                    events.append(WAY_TO_COMPASS_NP_BOMBED_EVENT)

        # Rewarded for destroying boxes to get to the coin
        # (with current BFS, it leads you closest to the coin, then returns "NP")
        # This never happens when you pickup the coin
        if (
            self.state_space.get_state(old_agent_state)["compass_mode"] == "coin" and
            self.state_space.get_state(old_agent_state)["compass"] == "NP"
        ):
            events.append(WAY_TO_COMPASS_NP_BOMBED_EVENT)

        # Rewarded for laying a bomb close to an enemy as indicated by compass "NP"
        if (
            self.state_space.get_state(old_agent_state)["compass_mode"] == "attack" and
            self.state_space.get_state(old_agent_state)["compass"] == "NP"
        ):
            events.append(COMPASS_NP_NEXT_TO_ENEMY_EVENT)

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

    # Create a checkpoint every 1000 rounds
    if last_game_state["round"] > 0 and last_game_state["round"] % 1000 == 0:
        Path("checkpoints").mkdir(exist_ok=True)

        if hasattr(self, "checkpoint_rounds"):
            filename = f"checkpoints/q_table_checkpoint_{last_game_state['round'] + self.checkpoint_rounds}.pt"
        else:
            filename = f"checkpoints/q_table_checkpoint_{last_game_state['round']}.pt"
        with open(filename, "wb") as file:
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
        e.BOMB_DROPPED: -0.5,
        e.WAITED: -0.4,
        e.CRATE_DESTROYED: 1.5,
        e.GOT_KILLED: -5,
        e.KILLED_SELF: -3,
        e.SURVIVED_ROUND: 1.5,
        e.KILLED_OPPONENT: 5,
        OPTIMAL_CRATE_EVENT: 2.0,
        WAY_TO_COMPASS_NP_BOMBED_EVENT: 1.0,
        FOLLOWED_COMPASS_DIRECTIONS_EVENT: 0.4,  # Negates penalty for moving, makes the sum 0.0
        COMPASS_NP_NEXT_TO_ENEMY_EVENT: 1.0  # Must be bigger than penalty for setting bomb
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
    rotated_state_90 = dict(origin_state)
    rotated_state_180 = dict(origin_state)
    rotated_state_270 = dict(origin_state)

    # Rotate 90° clockwise the agent and define the rotated state space
    # There is a tile on the board whose neighboring tiles have exactly
    # the same value but different order than the origin state
    rotated_state_90['tile_up'] = origin_state['tile_left']
    rotated_state_90['tile_down'] = origin_state['tile_right']
    rotated_state_90['tile_left'] = origin_state['tile_down']
    rotated_state_90['tile_right'] = origin_state['tile_up']

    # Rotate 180° clockwise the agent and define the rotated state space
    rotated_state_180['tile_up'] = origin_state['tile_down']
    rotated_state_180['tile_down'] = origin_state['tile_up']
    rotated_state_180['tile_left'] = origin_state['tile_right']
    rotated_state_180['tile_right'] = origin_state['tile_left']

    # Rotate 270° clockwise the agent and define the rotated state space
    rotated_state_270['tile_up'] = origin_state['tile_right']
    rotated_state_270['tile_down'] = origin_state['tile_left']
    rotated_state_270['tile_left'] = origin_state['tile_up']
    rotated_state_270['tile_right'] = origin_state['tile_down']

    # Match "compass" in origin state to the one in rotated state
    if origin_state['compass'] == "N":
        rotated_state_90['compass'] = "E"
        rotated_state_180['compass'] = "S"
        rotated_state_270['compass'] = "W"
    elif origin_state['compass'] == "S":
        rotated_state_90['compass'] = "W"
        rotated_state_180['compass'] = "N"
        rotated_state_270['compass'] = "E"
    elif origin_state['compass'] == "W":
        rotated_state_90['compass'] = "N"
        rotated_state_180['compass'] = "E"
        rotated_state_270['compass'] = "S"
    elif origin_state['compass'] == "E":
        rotated_state_90['compass'] = "S"
        rotated_state_180['compass'] = "W"
        rotated_state_270['compass'] = "N"
    else:
        rotated_state_90['compass'] = "NP"
        rotated_state_180['compass'] = "NP"
        rotated_state_270['compass'] = "NP"

    # Match "action" in origin state to the one in rotated state
    if action == "UP":
        rotated_action_90 = "RIGHT"
        rotated_action_180 = "DOWN"
        rotated_action_270 = "LEFT"
    elif action == "DOWN":
        rotated_action_90 = "LEFT"
        rotated_action_180 = "UP"
        rotated_action_270 = "RIGHT"
    elif action == "LEFT":
        rotated_action_90 = "UP"
        rotated_action_180 = "RIGHT"
        rotated_action_270 = "DOWN"
    elif action == "RIGHT":
        rotated_action_90 = "DOWN"
        rotated_action_180 = "LEFT"
        rotated_action_270 = "UP"
    elif action == "WAIT":
        rotated_action_90 = "WAIT"
        rotated_action_180 = "WAIT"
        rotated_action_270 = "WAIT"
    else:
        rotated_action_90 = "BOMB"
        rotated_action_180 = "BOMB"
        rotated_action_270 = "BOMB"

    # TODO: Update only unique states?
    all_states = [origin_state, rotated_state_90, rotated_state_180, rotated_state_270]
    all_actions = [action, rotated_action_90, rotated_action_180, rotated_action_270]

    # Update all rotated values
    for s, a in zip(all_states, all_actions):
        s_ix = self.state_space.get_index(s)
        update_q_table(self, s_ix, a)

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

    alpha = 0.08
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
