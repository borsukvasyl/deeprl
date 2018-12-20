from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from deeprl.callbacks import BaseCallback


class Saver(BaseCallback):
    def __init__(self, model, step=1, filename="model_chkp/model", max_to_keep=0):
        super(Saver, self).__init__()

        self.model = model
        self.step = step
        self.filename = filename
        self.last_episode = -1

        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.model.name),
                                    max_to_keep=max_to_keep)

    def on_episode_end(self, episode, logs=None, **kwargs):
        """
        Saves model when episode is ended and last save was some number of steps ago
        :param episode: int, episode number
        :param logs: dict, episode data
        :return: None
        """
        if episode - self.last_episode >= self.step:
            self.saver.save(self.model.session, self.filename, global_step=episode)
            self.last_episode = episode
