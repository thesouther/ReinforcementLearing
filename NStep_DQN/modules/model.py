import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DQN(nn.Module):
    def __init__(self, env, device):
        super(DQN, self).__init__()
        self.input_shape = env.observation_space.shape
        self.n_actions = env.action_space.n
        self.device = device

        self.features = nn.Sequential(nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4), nn.ReLU(),
                                      nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                                      nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.n_actions),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        tmp = torch.zeros(1, *self.input_shape)
        return self.features(tmp).view(1, -1).size(1)