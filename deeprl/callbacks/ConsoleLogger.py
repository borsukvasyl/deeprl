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

    def on_episode_end(self, episode, logs=None, **kwargs):
        """
        Prints episode total reward and loss
        :param episode: int, episode number
        :param logs: dict, episode data
        :return: None
        """
        if logs is None:
            logs = {}
        logging.info("EPISODE {}: total_reward: {}, loss: {}".format(episode, logs.get("r_total"), logs.get("loss")))

    def on_train_begin(self, logs=None, **kwargs):
        """
        Prints that training has started
        :param logs: training data
        :return: None
        """
        first_episode = kwargs.get("start_episode", None)
        logging.info("Starting training from episode {}\n".format(first_episode if first_episode is not None else 0))

    def on_train_end(self, logs=None, **kwargs):
        """
        Prints that training has finished
        :param logs: training data
        :return: None
        """
        logging.info("Training finished\n")
