import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DQN(nn.Module):
    def __init__(self, env, device, noisy=False, sigma_init=0.5):
        super(DQN, self).__init__()
        self.input_shape = env.observation_space.shape
        self.n_actions = env.action_space.n
        self.device = device
        self.noisy = noisy

        self.features = nn.Sequential(nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4), nn.ReLU(),
                                      nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                                      nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512) if not noisy else NoisyLinear(
                self.feature_size(), 512, sigma_init),
            nn.ReLU(),
            nn.Linear(512, self.n_actions) if not noisy else NoisyLinear(512, self.n_actions, sigma_init),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        tmp = torch.zeros(1, *self.input_shape)
        return self.features(tmp).view(1, -1).size(1)

    def act(self, state, epsilon, test=False):
        with torch.no_grad():
            if random.random() >= epsilon or test:
                state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
                q_value = self.forward(state)
                action = q_value.max(1)[1].view(1, 1)
                return action.item()
            else:
                return random.randrange(self.n_actions)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4, factorised_noise=True):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.factorised_noise = factorised_noise
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_spsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def sample_noise(self):
        if self.factorised_noise:
            epsilon_in = self._scale_noise(self.in_features)
            epsilon_out = self._scale_noise(self.out_features)
            self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
            self.bias_epsilon.copy_(epsilon_out)
        else:
            self.weight_epsilon.copy_(torch.randn((self.out_features, self.in_features)))
            self.bias_epsilon.copy_(torch.randn(self.out_features))

    def forward(self, inp):
        if self.training:
            return F.linear(inp, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(inp, self.weight_mu, self.bias_mu)