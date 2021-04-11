from collections import deque
import numpy as np
import random
import matplotlib.pyplot as plt


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


def curve_plot(path_result, frame_idx, rewards, losses):
    plt.figure(figsize=(20, 8))
    plt.subplot(121)
    plt.title('frame: %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(122)
    plt.title("loss")
    plt.plot(losses)
    plt.savefig(path_result)
    plt.close()