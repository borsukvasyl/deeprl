from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod


class BaseDQN(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_q(self, states):
        pass

    @abstractmethod
    def train(self, batch_states, batch_actions, q_target):
        pass

    def get_one_q(self, state):
        return self.get_q([state])[0]
