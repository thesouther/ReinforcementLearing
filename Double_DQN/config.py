import os
import torch
import math
import time

from torch import autograd


class Config:
    def __init__(self) -> None:
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.env_name = "PongNoFrameskip-v4"  # "CartPole-v0"
        self.alg_name = "NStep_DQN"
        self.env_module = "img"  # ["img", "vect"]
        self.save_curve = True

        self.epsilon_start = 1.0
        self.epsilon_final = 0.005
        self.epsilon_decay = 30000
        self.buffer_size = 100000

        self.target_upfreq = 1000
        self.log_freq = 200
        self.learn_start = 3000
        self.update_freq = 1
        self.n_steps = 4
        self.action_selection_count_frequency = 1000

        self.gamma = 0.99
        self.batch_size = 32
        self.max_train_steps = 1000000
        self.lr = 1e-4

        self.epsilon_by_frame = lambda frame_idx: self.epsilon_final + (self.epsilon_start \
                               - self.epsilon_final) * math.exp(-1. * frame_idx / self.epsilon_decay)

        root_path = os.path.dirname(os.path.dirname(__file__))
        exp_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        self.exp_name = self.alg_name + "_" + self.env_name + "_" + exp_time

        self.path_game_scan = os.path.join(root_path, "results", self.alg_name)
        if not os.path.isdir(self.path_game_scan):
            os.mkdir(self.path_game_scan)

        self.path_plot = os.path.join(self.path_game_scan, "plot_" + self.exp_name + ".png")
        self.path_models = os.path.join(self.path_game_scan, "model_" + self.exp_name + ".dump")
        self.path_optim = os.path.join(self.path_game_scan, "optim_" + self.exp_name + ".dump")
        self.path_memory = os.path.join(self.path_game_scan, "buffer_" + self.exp_name + ".dump")
        self.path_sig_param = os.path.join(self.path_game_scan, "sig_param_" + self.exp_name + ".csv")
        self.path_td = os.path.join(self.path_game_scan, "td_" + self.exp_name + ".csv")
        self.path_action_log = os.path.join(self.path_game_scan, "actions_" + self.exp_name + ".csv")