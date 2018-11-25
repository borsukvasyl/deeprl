from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class CallbackList(object):
    def __init__(self, callbacks=None):
        self.callbacks = callbacks.copy() if callbacks is not None else []

    def append(self, callback):
        self.callbacks.append(callback)

    def set_model(self, model):
        for callback in self.callbacks:
            callback.model = model

    def on_episode_begin(self, episode, logs=None):
        for callback in self.callbacks:
            callback.on_episode_begin(episode, logs)

    def on_episode_end(self, episode, logs=None):
        for callback in self.callbacks:
            callback.on_episode_end(episode, logs)

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)
