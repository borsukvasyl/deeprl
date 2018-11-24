from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod
import numpy as np


class BaseQLearning(object):
    __metaclass__ = ABCMeta

    def __init__(self, config, env, model):
        self.env = env
        self.model = model

        self.discount_factor = config.discount_factor

        self.callbacks = []

    def get_q_target(self, batch_rewards, batch_next_states, batch_dones):
        q_next_state = self.model.get_q(batch_next_states)

        q_target = np.zeros((len(batch_rewards),))
        for i in range(len(batch_rewards)):
            target = batch_rewards[i]
            if not batch_dones[i]:
                target += self.discount_factor * np.max(q_next_state[i])
            q_target[i] = target
        return q_target

    def run_callbacks_on_episode(self):
        for callback in self.callbacks:
            callback.on_episode()
