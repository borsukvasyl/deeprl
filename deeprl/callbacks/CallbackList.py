from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class CallbackList(object):
    def __init__(self, callbacks=None):
        """
        Initializes CallbacksList instance
        :param callbacks: list or None, list of callbacks
        """
        self.callbacks = callbacks.copy() if callbacks is not None else []

    def append(self, callback):
        """
        Adds new callback to list
        :param callback: BaseCallback, new callback
        :return: None
        """
        self.callbacks.append(callback)

    def on_episode_begin(self, episode, logs=None):
        """
        Runs on_episode_begin methods for all callbacks when new episode is started
        :param episode: int, episode number
        :param logs: dict, episode data
        :return: None
        """
        for callback in self.callbacks:
            callback.on_episode_begin(episode, logs)

    def on_episode_end(self, episode, logs=None):
        """
        Runs on_episode_end methods for all callbacks when episode is finished
        :param episode: int, episode number
        :param logs: dict, episode data
        :return: None
        """
        for callback in self.callbacks:
            callback.on_episode_end(episode, logs)

    def on_train_begin(self, logs=None):
        """
        Runs on_train_begin methods for all callbacks when training is started
        :param logs: training data
        :return: None
        """
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        """
        Runs on_train_end methods for all callbacks when training is finished
        :param logs: training data
        :return: None
        """
        for callback in self.callbacks:
            callback.on_train_end(logs)
