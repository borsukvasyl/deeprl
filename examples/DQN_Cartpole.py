from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import tensorflow as tf
import random

from deeprl.algorithms.qlearning import DQN, DQNConfig
from deeprl.models.qlearning import BaseDQN


RANDOM_SEED = 40

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)


class DQNetwork(BaseDQN):
    def build(self, s_size, a_size):
        self.s_size = s_size
        self.a_size = a_size

        self.states = tf.placeholder(shape=[None, self.s_size], dtype=tf.float32)
        self.q_target = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None, ], dtype=tf.uint8)

        self.dense = tf.layers.dense(inputs=self.states, units=20, activation=tf.nn.tanh)
        self.q_value = tf.layers.dense(inputs=self.dense, units=self.a_size)

        self.loss = self.calculate_loss(self.q_value, self.q_target, self.actions, self.a_size)
        trainer = tf.train.AdamOptimizer(learning_rate=0.01)
        self.optimize = trainer.minimize(self.loss)

    def get_q(self, states):
        return self.session.run(self.q_value, feed_dict={self.states: states})

    def train(self, batch_states, batch_actions, q_target):
        loss, _ = self.session.run([self.loss, self.optimize], feed_dict={
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


sess = tf.Session()
model = DQNetwork("main", sess, s_size=s_size, a_size=a_size)
config = DQNConfig()
algorithm = DQN(config, env, model)

sess.run(tf.global_variables_initializer())

algorithm.train(300)


s = env.reset()
done = False
while not done:
    env.render()
    a = algorithm.choose_action(s)
    s, r, done, _ = env.step(a)
env.close()
