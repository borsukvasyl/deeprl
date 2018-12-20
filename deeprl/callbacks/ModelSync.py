from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deeprl.callbacks.BaseCallback import BaseCallback


class ModelSync(BaseCallback):
    def __init__(self, model, source_model, step=1):
        """
        Initializes ModelSync instance
        :param model: BaseModel, model instance
        :param source_model: BaseModel, source model instance
        :param step: int, number of episode after which synchronization is run
        """
        super(ModelSync, self).__init__()

        self.model = model
        self.source_model = source_model

        self.step = step
        self.last_episode = 0

    def on_train_begin(self, logs=None, **kwargs):
        """
        Makes initial synchronisation
        :param logs: dict, episode data
        :return: None
        """
        self.model.sync(self.source_model)

    def on_episode_end(self, episode, logs=None, **kwargs):
        """
        Synchronizes model if last update was self.step episodes ago
        :param episode: int, episode number
        :param logs: dict, episode data
        :return: None
        """
        if episode - self.last_episode >= self.step:
            self.model.sync(self.source_model)
            self.last_episode = episode
