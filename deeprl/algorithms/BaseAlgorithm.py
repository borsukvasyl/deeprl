from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod

from deeprl.callbacks import CallbackList, ConsoleLogger


class BaseAlgorithm(object):
    __metaclass__ = ABCMeta

    def __init__(self, env, model, policy):
        self.env = env
        self.model = model

        self.policy = policy
        self.callbacks = CallbackList([
            ConsoleLogger()
        ])

    @abstractmethod
    def run_episode(self):
        pass

    @abstractmethod
    def choose_action(self, state):
        pass

    def train(self, num_episodes):
        self.callbacks.on_train_begin()

        for episode in range(num_episodes):
            self.callbacks.on_episode_begin(episode)
            logs = self.run_episode()
            self.callbacks.on_episode_end(episode, logs)

        self.callbacks.on_train_end()
