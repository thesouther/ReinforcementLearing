import os
import torch
import math

from torch import autograd


class Config:
    def __init__(self) -> None:
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.env_name = "CartPole-v0"
        self.epsilon_start = 1.0
        self.epsilon_final = 0.005
        self.epsilon_decay = 500
        self.gamma = 0.99
        self.batch_size = 32
        self.num_frames = 10000
        self.lr = 1e-4
        self.target_upfreq = 1000
        self.buffer_size = 10000
        self.log_freq = self.num_frames

        self.epsilon_by_frame = lambda frame_idx: self.epsilon_final + (
            self.epsilon_start - self.epsilon_final) * math.exp(-1. * frame_idx / self.epsilon_decay)

        self.exp_name = "DQN_" + self.env_name
        self.path_plot = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "plots")

        file_game_scan = "game_scans"
        self.path_game_scan = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results",
                                           file_game_scan)

        if not os.path.isdir(self.path_game_scan):
            os.mkdir(self.path_game_scan)