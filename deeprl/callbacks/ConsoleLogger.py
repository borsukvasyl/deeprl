from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from deeprl.callbacks import BaseCallback


class ConsoleLogger(BaseCallback):
    FORMAT = "%(asctime)s %(levelname)s %(message)s"

    def __init__(self):
        super(ConsoleLogger, self).__init__()

        logging.basicConfig(level=logging.INFO, format=ConsoleLogger.FORMAT)

    def on_episode_begin(self, episode, logs=None):
        pass

    def on_episode_end(self, episode, logs=None):
        if logs is None:
            logs = {}
        logging.info("EPISODE {}: total_reward: {}, loss: {}".format(episode, logs.get("r_total"), logs.get("loss")))

    def on_train_begin(self, logs=None):
        logging.info("Starting training\n")

    def on_train_end(self, logs=None):
        logging.info("Training finished\n")
