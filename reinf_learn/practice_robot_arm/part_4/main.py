#-*- coding:utf-8 –*-

# 完成运行程序

from env import ArmEnv
from rl import DDPG


MAX_EPISODES = 500
MAX_EP_STEPS = 200
ON_TRAIN = True

#set env
env = ArmEnv()
s_dim= env.state_dim
a_dim=env.action_dim       #
a_bound = env.action_bound #action 动作范围,在此项目中可以是角度

#set RL method
rl = DDPG(a_dim, s_dim, a_bound)

def train():
    # start training
    for i in range(MAX_EPISODES):
        s = env.reset()
        for j in range(MAX_EP_STEPS):
            env.render()

            a = rl.choose_action(s)

            s_, r, done = env.step(a)

            rl.store_transition(s, a, r, s_)

            if rl.memory_full:
                rl.learn()
                s = s_

    rl.save()

def eval():
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    while True:
        s = env.reset()
        for _ in range(200):
            env.render()
            a = rl.choose_action(s)
            s, r, done = env.step(a)
            if done:
                break

if ON_TRAIN:
    train()
else:
    eval()


# #start training
# for i in range(MAX_EPISODES):
#     s=env.reset()
#     for j in range(MAX_EP_STEPS):
#         env.render()
#
#         a=rl.choose_action(s)
#
#         s_,r,done = env.step(a)
#
#         rl.store_transition(s,a,r,s_)
#
#         if rl.memory_full():
#             rl.learn()
#
#         s = s_

# summary
#   env should have at least:
#     env.reset()
#     env.render()
#     env.step()
#
#   while RL should have at least:
#     rl.choose_action()
#     rl.store_transition()
#     rl.learn()
#     rl.memry_full()