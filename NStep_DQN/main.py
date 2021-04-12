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
    all_rewards = []
    state = env.reset()  # (1, 84, 84)
    for frame_idx in range(1, conf.max_train_steps + 1):
        epsilon = conf.epsilon_by_frame(frame_idx)

        # action = model.act(state, epsilon)

        # next_state, reward, done, _ = env.step(action)
        # replay_buffer.push(state, action, reward, next_state, done)

        # state = next_state
        # episode_reward += reward

        # if done:
        #     state = env.reset()
        #     all_rewards.append(episode_reward)
        #     episode_reward = 0

        # if len(replay_buffer) > conf.batch_size:
        #     loss = cal_td_loss(model, conf.batch_size)
        #     losses.append(loss.item())

        # if frame_idx % conf.target_upfreq == 0:
        #     target_model.load_state_dict(model.state_dict())

        if frame_idx % conf.log_freq == 0:
            print("frame: {}, loss: {}, reward: {}.".format(frame_idx, loss, episode_reward))

    if conf.save_curve:
        curve_name = "res_" + conf.exp_name + ".png"
        curve_path = os.path.join(conf.path_plot, curve_name)
        curve_plot(curve_path, frame_idx, all_rewards, losses)


if __name__ == "__main__":
    train()