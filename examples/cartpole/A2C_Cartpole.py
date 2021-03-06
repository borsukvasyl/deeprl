from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import tensorflow as tf
import random

from deeprl.agents import ActorCriticAgent
from deeprl.callbacks.Saver import Saver
from deeprl.trainers.actorcritic import A2CTrainer, ACConfig
from deeprl.models.actorcritic import BaseACNet
from deeprl.utils import record_video


RANDOM_SEED = 40

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)


class ACNet(BaseACNet):
    def build(self, s_size, a_size):
        self.s_size = s_size
        self.a_size = a_size

        self.states = tf.placeholder(shape=[None, self.s_size], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.uint8)
        self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)
        self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)

        self.dense = tf.layers.dense(inputs=self.states, units=32, activation=tf.nn.relu)
        self.policy = tf.layers.dense(inputs=self.dense, units=self.a_size, activation=tf.nn.softmax)
        self.v = tf.layers.dense(inputs=self.dense, units=1, activation=None)

        self.policy_loss = self.calculate_policy_loss(self.policy, self.actions, self.advantages, self.a_size)
        self.value_loss = self.calculate_value_loss(self.v, self.target_v)
        self.loss = self.policy_loss + 0.5 * self.value_loss
        trainer = tf.train.AdamOptimizer(learning_rate=0.01)
        self.optimize = trainer.minimize(self.loss)

    def get_policy(self, states):
        return self.session.run(self.policy, feed_dict={self.states: states})

    def get_v(self, states):
        return self.session.run(self.v, feed_dict={self.states: states})[:, 0]

    def get_policy_and_value(self, states):
        policy, v = self.session.run([self.policy, self.v], feed_dict={self.states: states})
        return policy, v

    def train(self, batch_states, batch_actions, batch_advantages, batch_target_v):
        loss, _ = self.session.run([self.loss, self.optimize], feed_dict={
            self.states: batch_states,
            self.actions: batch_actions,
            self.advantages: batch_advantages,
            self.target_v: batch_target_v
        })
        return loss


env = gym.make("CartPole-v0")
# env._max_episode_steps = 800

a_size = env.action_space.n
s_size = env.observation_space.shape[0]
print("Action space size: {}".format(a_size))
print("State space size: {}".format(s_size))


sess = tf.Session()
model = ACNet("main", sess, s_size=s_size, a_size=a_size)
agent = ActorCriticAgent(model)
sess.run(tf.global_variables_initializer())

checkpoint = tf.train.get_checkpoint_state("model_chkp")
if checkpoint and checkpoint.model_checkpoint_path:
    print("Loading: {}".format(checkpoint.model_checkpoint_path))
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, model.name))
    saver.restore(sess, checkpoint.model_checkpoint_path)
else:
    config = ACConfig()
    config.discount_factor = 0.9
    trainer = A2CTrainer(config, agent, env)
    # trainer.callbacks.append(Saver(model, step=20))

    trainer.train(300)

record_video(agent, env)
