import os
import os.path as osp
import numpy as np
import random

import torch
import torch.optim as optim
from .model import DQN
from .replay_buffer import ReplayBuffer
from .base_agent import BaseAgent


class Agent(BaseAgent):
    def __init__(self, conf=None, env=None, test=False):
        super(Agent, self).__init__(conf=conf, env=env)
        self.device = conf.device
        self.conf = conf
        self.env = env
        self.test = test
        self.input_shape = env.observation_space.shape
        self.n_actions = env.action_space.n

        # DQN model
        self.model = DQN(env, self.device)
        self.target_model = DQN(env, self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.model, self.target_model = self.model.to(self.device), self.target_model.to(self.device)

        if self.test:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.train()

        self.optimizer = optim.Adam(self.model.parameters(), lr=conf.lr)
        self.replay_buffer = ReplayBuffer(conf.buffer_size)

    def compute_loss(self, batch_size):
        s, a, r, s_, d = self.replay_buffer.sample(batch_size)

        fea_shape = (-1, ) + self.input_shape
        # [32, 1, 84, 84]
        s = torch.tensor(s, device=self.device, dtype=torch.float).view(fea_shape)
        s_ = torch.tensor(s_, device=self.device, dtype=torch.float).view(fea_shape)
        # [32, 1]
        a = torch.tensor(a, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        r = torch.tensor(r, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        d = torch.tensor(d, device=self.device, dtype=torch.float).squeeze().view(-1, 1)

        # [32, 1]
        q_value = self.model(s).gather(1, a)

        # td target [32, 1]
        with torch.no_grad():
            max_next_a = self.model(s_).max(1)[1].view(-1, 1)

            max_next_q_values = self.target_model(s_).gather(1, max_next_a)

            expected_q_value = r + self.conf.gamma * max_next_q_values * (1 - d)
            expected_q_value.to(self.device)

        loss = self.MSE(q_value - expected_q_value).mean()

        return loss

    def update(self, state, action, reward, next_state, done, test=False, frame=0):
        # test 不用前传
        if test:
            return None

        # 填充经验池
        self.replay_buffer.push(state, action, reward, next_state, done)

        if frame < self.conf.learn_start or frame % self.conf.update_freq != 0:
            return None

        # 更新loss
        loss = self.compute_loss(self.conf.batch_size)
        # optimizer
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # update target net
        if frame % self.conf.target_upfreq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        # save data
        self.save_td(loss.item(), frame)
        # self.save_sigma_param_magnitudes(frame)
        return loss

    def act(self, state, epsilon, test=False):
        """
        state: [1, 84, 84]
        """
        with torch.no_grad():
            if random.random() >= epsilon or test:
                # [1, 1, 84, 84]
                state = torch.tensor(state, device=self.device, dtype=torch.float).unsqueeze(0)
                q_value = self.model(state)
                action = q_value.max(1)[1].view(1, 1)
                return action.item()
            else:
                return random.randrange(self.n_actions)