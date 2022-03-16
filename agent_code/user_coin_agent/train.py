from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import ACTIONS
from .callbacks import game_state_to_feature


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# This is a hyperparameter to our TD-learning
# For simplicity, we implement only =1 in this example 
TRANSITION_HISTORY_SIZE = 1


def setup_training(self):
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    
    old_agent_state = None
    new_agent_state = None

    if old_game_state is not None or new_game_state is None:
        old_agent_state = game_state_to_feature(self, old_game_state)
        new_agent_state = game_state_to_feature(self, new_game_state)

    self.transitions.append(Transition(old_agent_state,
                                       self_action,
                                       new_agent_state,
                                       reward_from_events(self, events)))

    update_q_table(self)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    
    last_agent_state = game_state_to_feature(self, last_game_state)
    
    self.transitions.append(Transition(last_agent_state,
                                       last_action,
                                       None,
                                       reward_from_events(self, events)))

    update_q_table(self)

    with open("q_table.pt", "wb") as file:
        pickle.dump(self.q_table, file)


def reward_from_events(self, events: List[str]) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.MOVED_UP: -0.1,
        e.MOVED_DOWN: -0.1,
        e.MOVED_RIGHT: -0.1,
        e.MOVED_LEFT: -0.1,
        e.INVALID_ACTION: -0.5,
        e.BOMB_DROPPED: -0.5,
        e.WAITED: -0.5
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def update_q_table(self):
    alpha = 0.02
    gamma = 0.9
    old_state, action, new_state, reward = self.transitions.popleft()

    # Temporary: ignore first and last timestep
    if old_state is not None or new_state is not None:
        old_q_value = self.q_table[old_state, ACTIONS.index(action)]
        self.logger.info(f"Q-value {old_q_value}")

        updated_q_value = \
            old_q_value + alpha * (reward + gamma * self.q_table[new_state].max() - old_q_value)
        self.q_table[old_state, ACTIONS.index(action)] = updated_q_value

        self.logger.info(f"Adjusted Q-Table at index {old_state} from {old_q_value} to {updated_q_value}.")
