from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deeprl.callbacks.BaseCallback import BaseCallback


class ModelSync(BaseCallback):
    def __init__(self, model, source_model, step=1):
        super(ModelSync, self).__init__()

        self.model = model
        self.source_model = source_model

        self.step = step
        self.last_episode = 0

    def on_episode_end(self, episode, logs=None):
        if episode - self.last_episode >= self.step:
            self.model.sync(self.source_model)
            self.last_episode = episode
