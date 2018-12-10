from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import tensorflow as tf
import random

from deeprl.agents import PolicyAgent
from deeprl.trainers.policy import PolicyGradientTrainer, PolicyGradientConfig
from deeprl.models.policy import BasePolicyNet


RANDOM_SEED = 40

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)


class PolicyNet(BasePolicyNet):
    def build(self, s_size, a_size):
        self.s_size = s_size
        self.a_size = a_size

        self.states = tf.placeholder(shape=[None, self.s_size], dtype=tf.float32)
        self.q_target = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None, ], dtype=tf.uint8)
        self.discounted_episode_rewards = tf.placeholder(shape=[None, ], dtype=tf.float32)

        self.dense1 = tf.layers.dense(inputs=self.states, units=20, activation=tf.nn.relu)
        self.dense2 = tf.layers.dense(inputs=self.states, units=16, activation=tf.nn.relu)
        self.policy = tf.layers.dense(inputs=self.dense2, units=self.a_size, activation=tf.nn.softmax)

        self.loss = self.calculate_loss(self.policy, self.actions, self.discounted_episode_rewards, self.a_size)
        trainer = tf.train.AdamOptimizer(learning_rate=0.01)
        self.optimize = trainer.minimize(self.loss)

    def get_policy(self, states):
        return self.session.run(self.policy, feed_dict={self.states: states})

    def train(self, batch_states, batch_actions, discounted_episode_rewards):
        loss, _ = self.session.run([self.loss, self.optimize], feed_dict={
            self.states: batch_states,
            self.actions: batch_actions,
            self.discounted_episode_rewards: discounted_episode_rewards
        })
        return loss


env = gym.make("CartPole-v0")

a_size = env.action_space.n
s_size = env.observation_space.shape[0]
print("Action space size: {}".format(a_size))
print("State space size: {}".format(s_size))


sess = tf.Session()
model = PolicyNet("main", sess, s_size=s_size, a_size=a_size)
agent = PolicyAgent(model)
config = PolicyGradientConfig()
trainer = PolicyGradientTrainer(config, agent, env)

sess.run(tf.global_variables_initializer())

trainer.train(200)


s = env.reset()
done = False
while not done:
    env.render()
    a = agent.choose_action(s)
    s, r, done, _ = env.step(a)
env.close()
