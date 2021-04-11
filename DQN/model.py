import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, env, conf):
        super(DQN, self).__init__()

        self.env = env
        self.num_inputs = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        self.conf = conf
        self.device = conf.device
        self.layes = nn.Sequential(nn.Linear(self.num_inputs, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                                   nn.Linear(128, self.num_actions))

    def forward(self, x):
        return self.layes(x)

    def act(self, state, epsilon):
        with torch.no_grad():
            if random.random() > epsilon:
                state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
                q_value = self.forward(state)
                action = q_value.max(1)[1].view(1, 1)
                return action.item()
            else:
                return random.randrange(self.num_actions)
