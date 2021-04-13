import os
import gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from config import Config
from utils.utils import curve_plot
from utils.wrappers import *
from modules.agent import Agent

conf = Config()
device = conf.device


def train():
    env = make_atari(conf.env_name)
    env = bench.Monitor(env, os.path.join(conf.path_game_scan, conf.env_name))
    env = wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=True)
    env = WrapPyTorch(env)
    agent = Agent(conf=conf, env=env, test=False)

    episode_reward = 0
    losses = []
    state = env.reset()  # (1, 84, 84)
    for frame_idx in range(1, conf.max_train_steps + 1):
        epsilon = conf.epsilon_by_frame(frame_idx)

        action = agent.act(state, epsilon, test=False)
        # agent.save_action(action, frame_idx)

        next_state, reward, done, _ = env.step(action)
        loss = agent.update(state, action, reward, next_state, done, test=False, frame=frame_idx)

        # state = next_state
        episode_reward += reward

        if done:
            state = env.reset()
            agent.save_reward(episode_reward)
            episode_reward = 0

        if frame_idx % conf.log_freq == 0 and loss:
            print("frame: {}, reward: {}.".format(frame_idx, episode_reward))

    if conf.save_curve:
        curve_plot(conf.path_plot, frame_idx, agent.all_rewards, losses)
        # agent.save_w()


if __name__ == "__main__":
    train()