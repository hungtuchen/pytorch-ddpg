import numpy as np
import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch = zip(*random.sample(self.memory, batch_size))

        state_batch = np.concatenate(state_batch, axis=0)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        next_state_batch = np.concatenate(next_state_batch, axis=0)

        return state_batch, action_batch, reward_batch, next_state_batch

    def __len__(self):
        return len(self.memory)
