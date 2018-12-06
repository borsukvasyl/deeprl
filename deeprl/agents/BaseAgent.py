from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod


class BaseAgent(object):
    __metaclass__ = ABCMeta

    def __init__(self, model, policy):
        self.model = model
        self.policy = policy

    @abstractmethod
    def choose_action(self, state):
        pass
