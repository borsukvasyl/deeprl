from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod

from deeprl.callbacks import CallbackList, ConsoleLogger


class BaseTrainer(object):
    __metaclass__ = ABCMeta

    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

        self.callbacks = CallbackList([
            ConsoleLogger()
        ])

    @abstractmethod
    def run_episode(self):
        pass

    def train(self, num_episodes, start_episode=0):
        self.callbacks.on_train_begin(start_episode=start_episode)

        for episode in range(start_episode, num_episodes):
            self.callbacks.on_episode_begin(episode)
            logs = self.run_episode()
            self.callbacks.on_episode_end(episode, logs)

        self.callbacks.on_train_end()
