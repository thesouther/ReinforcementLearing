import os
import gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from config import Config
from utils import ReplayBuffer, curve_plot
from model import DQN

conf = Config()
device = conf.device


def train():
    env = gym.make(conf.env_name)

    # Instantiate
    model = DQN(env, conf)
    target_model = DQN(env, conf)
    target_model.load_state_dict(model.state_dict())
    model, target_model = model.to(device), target_model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=conf.lr)
    replay_buffer = ReplayBuffer(conf.buffer_size)

    # cal td loss
    def cal_td_loss(model, batch_size):
        s, a, r, s_, d = replay_buffer.sample(batch_size)
        s = torch.tensor(np.float32(s), dtype=torch.float).to(device)
        s_ = torch.tensor(np.float32(s_), dtype=torch.float).to(device)
        a = torch.tensor(a, dtype=torch.long).to(device)
        r = torch.tensor(r, dtype=torch.float).to(device)
        d = torch.tensor(d, dtype=torch.float).to(device)

        q_value = model(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_value = target_model(s_).max(1)[0]
            expected_q_value = r + conf.gamma * next_q_value * (1 - d)

        loss = (q_value - conf.Variable(expected_q_value.data)).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        for param in model.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        return loss

    episode_reward = 0
    losses = []
    all_rewards = []
    state = env.reset()
    for frame_idx in range(1, conf.num_frames + 1):
        epsilon = conf.epsilon_by_frame(frame_idx)
        action = model.act(state, epsilon)

        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0

        if len(replay_buffer) > conf.batch_size:
            loss = cal_td_loss(model, conf.batch_size)
            losses.append(loss.item())

        if frame_idx % conf.target_upfreq == 0:
            target_model.load_state_dict(model.state_dict())

        if frame_idx % conf.log_freq == 0:
            print("frame: {}, loss: {}, reward: {}.".format(frame_idx, loss, episode_reward))
            curve_name = "res_" + conf.exp_name + ".png"
            curve_path = os.path.join(conf.path_plot, curve_name)
            curve_plot(curve_path, frame_idx, all_rewards, losses)


if __name__ == "__main__":
    train()