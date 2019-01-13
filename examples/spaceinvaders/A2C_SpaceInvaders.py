from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import tensorflow as tf
import random

from deeprl.agents import ActorCriticAgent
from deeprl.callbacks.Saver import Saver, BaseCallback
from deeprl.trainers.actorcritic import A2CTrainer, ACConfig
from deeprl.models.actorcritic import BaseACNet
from deeprl.utils import record_video


RANDOM_SEED = 40
N_EPISODES = 5000

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)


class LstmResetter(BaseCallback):
    def __init__(self, models):
        self.models = models

    def on_episode_begin(self, episode, logs=None, **kwargs):
        for _model in self.models:
            _model.reset()


class ACNet(BaseACNet):
    def build(self, s_size, a_size):
        self.s_size = s_size
        self.a_size = a_size

        self.states = tf.placeholder(shape=[None, *self.s_size], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.uint8)
        self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)
        self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)

        self.lstm_state0 = tf.placeholder(tf.float32, shape=[1, 256])
        self.lstm_state1 = tf.placeholder(tf.float32, shape=[1, 256])
        self.lstm_state = tf.contrib.rnn.LSTMStateTuple(self.lstm_state0, self.lstm_state1)
        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)

        conv1 = tf.layers.conv2d(inputs=self.states, filters=32, kernel_size=[6, 6], strides=3, activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[4, 4], strides=2, activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(inputs=conv2, filters=96, kernel_size=[4, 4], strides=2, activation=tf.nn.relu)
        conv4 = tf.layers.conv2d(inputs=conv3, filters=128, kernel_size=[4, 4], strides=2, activation=tf.nn.relu)
        flat = tf.layers.flatten(conv4)
        print("Flatten size:", flat.get_shape())

        dense1 = tf.layers.dense(inputs=flat, units=256, activation=tf.nn.relu)

        step_size = tf.shape(dense1)[:1]
        lstm_input_reshaped = tf.reshape(dense1, [1, -1, 256])
        lstm_outputs, self.base_lstm_state = tf.nn.dynamic_rnn(self.lstm_cell,
                                                               lstm_input_reshaped,
                                                               initial_state=self.lstm_state,
                                                               sequence_length=step_size,
                                                               time_major=False)
        lstm_outputs = tf.reshape(lstm_outputs, [-1, 256])

        self.policy = tf.layers.dense(inputs=lstm_outputs, units=self.a_size, activation=tf.nn.softmax)
        self.v = tf.layers.dense(inputs=lstm_outputs, units=1, activation=None)

        self.policy_loss = self.calculate_policy_loss(self.policy, self.actions, self.advantages, self.a_size)
        self.value_loss = self.calculate_value_loss(self.v, self.target_v)
        self.loss = self.policy_loss + 0.5 * self.value_loss
        trainer = tf.train.AdamOptimizer(learning_rate=0.01)
        self.optimize = trainer.minimize(self.loss)

    def get_policy(self, states):
        policy, self.base_lstm_state_out = self.session.run([self.policy, self.base_lstm_state], feed_dict={
            self.states: states,
            self.lstm_state0: self.base_lstm_state_out[0],
            self.lstm_state1: self.base_lstm_state_out[1]
        })
        return policy

    def get_v(self, states):
        return self.session.run(self.v, feed_dict={
            self.states: states,
            self.lstm_state0: self.base_lstm_state_out[0],
            self.lstm_state1: self.base_lstm_state_out[1]
        })[:, 0]

    def train(self, batch_states, batch_actions, batch_advantages, batch_target_v):
        loss, _ = self.session.run([self.loss, self.optimize], feed_dict={
            self.states: batch_states,
            self.actions: batch_actions,
            self.advantages: batch_advantages,
            self.target_v: batch_target_v,
            self.lstm_state0: np.zeros([1, 256]),
            self.lstm_state1: np.zeros([1, 256])
        })
        return loss

    def reset(self):
        self.base_lstm_state_out = tf.contrib.rnn.LSTMStateTuple(np.zeros([1, 256]), np.zeros([1, 256]))


env = gym.make("SpaceInvaders-v0")

a_size = env.action_space.n
s_size = env.observation_space.shape
print("Action space size: {}".format(a_size))
print("State space size: {}".format(s_size))


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
sess = tf.Session(config=config)
model = ACNet("main", sess, s_size=s_size, a_size=a_size)
agent = ActorCriticAgent(model)

config = ACConfig()
config.min_sample_size = 32
trainer = A2CTrainer(config, agent, env)
trainer.callbacks.append(Saver(model, step=20))
trainer.callbacks.append(LstmResetter([model]))

sess.run(tf.global_variables_initializer())

start_episode = 0
checkpoint = tf.train.get_checkpoint_state("model_chkp")
if checkpoint and checkpoint.model_checkpoint_path:
    print("Loading: {}".format(checkpoint.model_checkpoint_path))
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, model.name))
    saver.restore(sess, checkpoint.model_checkpoint_path)
    start_episode = int(checkpoint.model_checkpoint_path.split("-")[-1])


trainer.train(N_EPISODES, start_episode=start_episode)


# visualize(agent, env)
record_video(agent, env)
