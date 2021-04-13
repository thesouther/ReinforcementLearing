import pickle
import os, csv
import os.path as osp
import numpy as np
import random

import torch
import torch.optim as optim
from .model import DQN
from .replay_buffer import ReplayBuffer


class BaseAgent:
    def __init__(self, conf, env):
        self.model = None
        self.target_model = None
        self.optimizer = None

        self.n_actions = env.action_space.n

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
        torch.save(self.optimizer.state_dict(), self.conf.path_optim)

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
    def __init__(self, conf=None, env=None, test=False):
        super(Agent, self).__init__(conf=conf, env=env)
        self.device = conf.device
        self.conf = conf
        self.env = env
        self.test = test
        self.input_shape = env.observation_space.shape
        self.n_actions = env.action_space.n

        # DQN model
        self.model = DQN(env, self.device, noisy=False, sigma_init=0.0)
        self.target_model = DQN(env, self.device, noisy=False, sigma_init=0.0)
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
        self.n_steps = conf.n_steps
        self.nstep_buffer = []

    def append_to_replay(self, s, a, r, s_, d):
        self.nstep_buffer.append((s, a, r, s_, d))

        if len(self.nstep_buffer) < self.n_steps:
            return

        R = sum([self.nstep_buffer[i][2] * (self.conf.gamma**i) for i in range(self.n_steps)])
        state, action, _, _, _ = self.nstep_buffer.pop(0)
        self.replay_buffer.push(state, action, R, s_, d)

    def compute_loss(self, batch_size):
        s, a, r, s_, d = self.replay_buffer.sample(batch_size)

        fea_shape = (-1, ) + self.input_shape
        # [32, 1, 84, 84]
        s = torch.tensor(s, device=self.device, dtype=torch.float).view(fea_shape)
        # [32, 1]
        a = torch.tensor(a, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        r = torch.tensor(r, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        d = torch.tensor(d, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        # 计算非最终状态的掩码并连接batch元素
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, s_)),
                                      device=self.device,
                                      dtype=torch.bool)
        try:  #sometimes all next states are false [32, 1, 84, 84]
            non_final_s_ = torch.tensor([s for s in s_ if s is not None],
                                        device=self.device,
                                        dtype=torch.float).view(fea_shape)
            empty_next_state_values = False
        except:
            non_final_s_ = None
            empty_next_state_values = True

        # [32, 1]
        q_value = self.model(s).gather(1, a)

        # td target [32, 1]
        with torch.no_grad():
            max_next_q_values = torch.zeros(batch_size, device=self.device, dtype=torch.float).unsqueeze(1)
            if not empty_next_state_values:
                max_next_a = self.target_model(non_final_s_).max(1)[1].view(-1, 1)
                max_next_q_values[non_final_mask] = self.target_model(non_final_s_).gather(1, max_next_a)
            expected_q_value = r + ((self.conf.gamma**self.n_steps) * max_next_q_values)
            expected_q_value.to(self.device)

        loss = self.MSE(q_value - expected_q_value).mean()

        return loss

    def update(self, state, action, reward, next_state, done, test=False, frame=0):
        # test 不用前传
        if test:
            return None

        # 填充经验池
        self.append_to_replay(state, action, reward, next_state, done)

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
        self.save_sigma_param_magnitudes(frame)
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

    def finish_nstep(self):
        while len(self.nstep_buffer) > 0:
            R = sum([self.nstep_buffer[i][2] * (self.conf.gamma**i) for i in range(len(self.nstep_buffer))])
            state, action, _, _, d = self.nstep_buffer.pop(0)

            self.replay_buffer.push(state, action, R, None, d)

    def reset_hx(self):
        pass