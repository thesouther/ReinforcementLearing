import pickle
import os, csv
import os.path as osp
import numpy as np

import torch


class BaseAgent:
    def __init__(self, conf, env, log_path):
        self.model = None
        self.target_model = None
        self.optimizer = None

        self.n_actions = env.action_space.n

        self.log_path = log_path
        self.all_rewards = []
        self.conf = conf
        self.action_log_frequency = conf.action_selection_count_frequency
        self.action_selections = [0 for _ in range(self.n_actions)]

    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1.0 - cond)

    def MSE(self, x):
        return 0.5 * x.pow(2)

    def save_w(self):
        torch.save(self.model.state_dict(), self.conf.path_models)
        torch.seve(self.optimizer.state_dict(), self.conf.path_optim)

    def load_w(self):
        if osp.isfile(self.conf.path_models):
            self.model.load_state_dict(torch.load(self.conf.path_models))
            self.target_model.load_state_dict(torch.load(self.conf.path_models))
        if osp.isfile(self.conf.path_optim):
            self.optimizer.load_state_dict(torch.load(self.conf.path_optim))

    def save_replay(self):
        pickle.dump(self.memory, open(self.conf.path_memory, "wb"))

    def load_replay(self):
        if osp.isfile(self.conf.path_memory):
            self.memory = pickle.load(open(self.conf.path_memory, "rb"))

    def save_sigma_param_magnitudes(self, tstep):
        with torch.no_grad():
            sum_, count = 0.0, 0.0
            for name, param in self.model.named_parameters():
                if param.requires_grad and 'sigma' in name:
                    sum_ += torch.sum(param.abs()).item()
                    count += np.prod(param.shape)

            if count > 0:
                with open(self.conf.path_sig_param, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow((tstep, sum_ / count))

    def save_td(self, td, tstep):
        with open(self.conf.path_td, 'a') as f:
            writer = csv.writer(f)
            writer.writerow((tstep, td))

    def save_reward(self, reward):
        self.all_rewards.append(reward)

    def save_action(self, action, tstep):
        self.action_selections[int(action)] += 1.0 / self.action_log_frequency
        if (tstep + 1) % self.action_log_frequency == 0:
            with open(self.conf.path_action_log, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(list([tstep] + self.action_selections))
            self.action_selections = [0 for _ in range(len(self.action_selections))]


class Agent(BaseAgent):
    def __init__(self, conf=None, env=None, log_path=""):
        super(Agent, self).__init__(conf=conf, env=env, log_path=log_path)
        self.device = conf.device
