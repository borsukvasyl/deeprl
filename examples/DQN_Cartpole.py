from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import tensorflow as tf
import random

from deeprl.algorithms.qlearning import DQN, DQNConfig
from deeprl.models import AbstractDQN


RANDOM_SEED = 40

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)


class DQNetwork(AbstractDQN):
    def __init__(self, sess, s_size, a_size):
        self.sess = sess
        self.s_size = s_size
        self.a_size = a_size

        self.states = tf.placeholder(shape=[None, self.s_size], dtype=tf.float32)
        self.q_target = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None, ], dtype=tf.uint8)

        self.dense = tf.layers.dense(inputs=self.states, units=20, activation=tf.nn.tanh)
        self.q_value = tf.layers.dense(inputs=self.dense, units=self.a_size)

        actions_oh = tf.one_hot(self.actions, self.a_size)
        used_q = tf.reduce_sum(tf.multiply(self.q_value, actions_oh), axis=1)
        self.loss = tf.reduce_mean(tf.square(self.q_target - used_q))
        trainer = tf.train.AdamOptimizer(learning_rate=0.01)
        self.optimize = trainer.minimize(self.loss)

    def get_q(self, states):
        return self.sess.run(self.q_value, feed_dict={self.states: states})

    def train(self, batch_states, batch_actions, q_target):
        loss, _ = self.sess.run([self.loss, self.optimize], feed_dict={
            self.states: batch_states,
            self.actions: batch_actions,
            self.q_target: q_target
        })
        return loss


env = gym.make("CartPole-v0")

a_size = env.action_space.n
s_size = env.observation_space.shape[0]
print("Action space size: {}".format(a_size))
print("State space size: {}".format(s_size))


env = gym.make("CartPole-v0")
sess = tf.Session()
model = DQNetwork(sess, s_size, a_size)
config = DQNConfig()
algorithm = DQN(config, env, model)

sess.run(tf.global_variables_initializer())

print("Starting training")
algorithm.train(300)
print("Trained")


s = env.reset()
done = False
while not done:
    env.render()
    a = algorithm.choose_action(s)
    s, r, done, _ = env.step(a)
env.close()
