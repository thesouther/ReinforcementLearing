#-*- coding:utf-8 –*-

# 存取网络
import tensorflow as tf
import numpy as np

# 超参数
LR_A = 0.001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GAMMA = 0.9   # reward discount
TAU = 0.01    # spft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32


class DDPG(object):

    def __init__(self):
        pass

    def choose_action(self,s):
        pass

    def learn(self):
        pass

    def store_transition(self, s, a, r, s_):
        pass

    def _build_a(self, s, scope, trainable):
        pass

    def _build_c(self, s, a, scope, trainable):
        pass

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, 'params', write_meta_graph=False)

    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, 'params')
