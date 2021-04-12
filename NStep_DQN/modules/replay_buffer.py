from collections import deque
import random
import numpy as np


class ReplayBuffer():
    def __init__(self, bsize):
        self.buffer = deque(maxlen=bsize)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, bs):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, bs))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)