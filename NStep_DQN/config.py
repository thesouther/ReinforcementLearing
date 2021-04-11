import os
import torch
import math

from torch import autograd


class Config:
    def __init__(self) -> None:
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.env_name = "PongNoFrameskip-v4"  # "CartPole-v0"
        self.env_module = "img"  # ["img", "vect"]
        self.save_curve = True

        self.epsilon_start = 1.0
        self.epsilon_final = 0.005
        self.epsilon_decay = 30000
        self.buffer_size = 100000

        self.target_upfreq = 1000
        self.log_freq = 200
        self.learn_start = 1000
        self.update_freq = 1
        self.n_steps = 3
        self.action_selection_count_frequency = 1000

        self.gamma = 0.99
        self.batch_size = 32
        self.max_train_steps = 1000000
        self.lr = 1e-4

        self.epsilon_by_frame = lambda frame_idx: self.epsilon_final + (self.epsilon_start \
                               - self.epsilon_final) * math.exp(-1. * frame_idx / self.epsilon_decay)

        root_path = os.path.dirname(os.path.dirname(__file__))
        self.alg_name = "NStep_DQN"
        self.exp_name = self.alg_name + "_" + self.env_name
        self.path_plot = os.path.join(root_path, "results", "plots")
        self.path_models = os.path.join(root_path, "results", "models", self.exp_name + "_model.dump")
        self.path_optim = os.path.join(root_path, "results", "models", self.exp_name + "_optim.dump")
        self.path_memory = os.path.join(root_path, "results", "models", self.exp_name + "_memory.dump")
        self.path_sig_param = os.path.join(root_path, "results", "models", self.exp_name + "_sig_param.csv")
        self.path_td = os.path.join(root_path, "results", "models", self.exp_name + "_td.csv")
        self.path_action_log = os.path.join(root_path, "results", "models", self.exp_name + "_action_log.csv")

        self.path_game_scan = os.path.join(root_path, "logs", self.alg_name)

        if not os.path.isdir(self.path_game_scan):
            os.mkdir(self.path_game_scan)