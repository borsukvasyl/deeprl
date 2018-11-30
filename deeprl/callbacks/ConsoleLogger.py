from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from deeprl.callbacks import BaseCallback


FORMAT = "%(asctime)s %(levelname)s %(message)s"


class ConsoleLogger(BaseCallback):
    def __init__(self, level=logging.INFO, head_format=FORMAT):
        """
        Initializes ConsoleLogger instance
        :param level: int, logging level
        :param head_format: str, line header format
        """
        super(ConsoleLogger, self).__init__()

        logging.basicConfig(level=level, format=head_format)

    def on_episode_end(self, episode, logs=None):
        """
        Prints episode total reward and loss
        :param episode: int, episode number
        :param logs: dict, episode data
        :return: None
        """
        if logs is None:
            logs = {}
        logging.info("EPISODE {}: total_reward: {}, loss: {}".format(episode, logs.get("r_total"), logs.get("loss")))

    def on_train_begin(self, logs=None):
        """
        Prints that training has started
        :param logs: training data
        :return: None
        """
        logging.info("Starting training\n")

    def on_train_end(self, logs=None):
        """
        Prints that training has finished
        :param logs: training data
        :return: None
        """
        logging.info("Training finished\n")
