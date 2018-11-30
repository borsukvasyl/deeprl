from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta


class BaseCallback(object):
    __metaclass__ = ABCMeta

    def on_episode_begin(self, episode, logs=None):
        """
        Method which is executed when new episode is started
        :param episode: int, episode number
        :param logs: dict, episode data
        :return: None
        """
        pass

    def on_episode_end(self, episode, logs=None):
        """
        Method which is executed when episode is finished
        :param episode: int, episode number
        :param logs: dict, episode data
        :return: None
        """
        pass

    def on_train_begin(self, logs=None):
        """
        Method which is executed when training is started
        :param logs: training data
        :return: None
        """
        pass

    def on_train_end(self, logs=None):
        """
        Method which is executed when training is finished
        :param logs: training data
        :return: None
        """
        pass
