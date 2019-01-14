from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import tensorflow as tf
import random

from deeprl.agents import QAgent
from deeprl.callbacks import Tensorboard, Saver
from deeprl.trainers.qlearning import DoubleDQNTrainer, DQNConfig
from deeprl.models.qlearning import BaseDuelingDQN
from deeprl.utils import record_video
from deeprl.environments import make_wrapper


RANDOM_SEED = 40
N_EPISODES = 10000

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)


class DQNetwork(BaseDuelingDQN):
    def build(self, s_size, a_size):
        self.s_size = s_size
        self.a_size = a_size

        self.states = tf.placeholder(shape=[None, *self.s_size], dtype=tf.float32)
        self.q_target = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None, ], dtype=tf.uint8)

        conv1 = tf.layers.conv2d(inputs=self.states, filters=32, kernel_size=[6, 6], strides=3, activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[4, 4], strides=2, activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=[4, 4], strides=2, activation=tf.nn.relu)
        flat = tf.layers.flatten(conv3)
        print("Flatten size:", flat.get_shape())

        dense1 = tf.layers.dense(inputs=flat, units=512, activation=tf.nn.relu)

        self.value = tf.layers.dense(inputs=dense1, units=1)
        self.advantage = tf.layers.dense(inputs=dense1, units=self.a_size)
        self.q_value = self.calculate_q_value(self.value, self.advantage)

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


env = gym.make("SpaceInvaders-v0")
make_wrapper(env, stack_frames=True)

a_size = env.action_space.n
s_size = env.observation_space.shape
print("Action space size: {}".format(a_size))
print("State space size: {}".format(s_size))


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
sess = tf.Session(config=config)
model = DQNetwork("main", sess, s_size=s_size, a_size=a_size)
agent = QAgent(model)

config = DQNConfig()
config.experience_size = 100000
config.batch_size = 32
trainer = DoubleDQNTrainer(config, agent, env)
trainer.callbacks.append(Tensorboard(sess, ["r_total"]))
trainer.callbacks.append(Saver(model, step=20))

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
