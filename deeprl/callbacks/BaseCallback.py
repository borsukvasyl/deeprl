from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta


class BaseCallback(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.model = None

    def set_model(self, model):
        self.model = model

    def on_episode_begin(self, episode, logs=None):
        pass

    def on_episode_end(self, episode, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass
