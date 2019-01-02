from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from deeprl.callbacks import BaseCallback


class Tensorboard(BaseCallback):
    def __init__(self, session, log_names=("r_total", "loss"), log_dir="model_chkp"):
        """
        :param session: tf.Session
        :param log_names: list of strings, names of logs
        """
        self.session = session

        self.summary_placeholders = {log_name: tf.placeholder(tf.float32) for log_name in log_names}
        summary_scalars = [tf.summary.scalar(log_name, log_placeholder)
                           for log_name, log_placeholder in self.summary_placeholders.items()]
        self.summary_op = tf.summary.merge(summary_scalars)
        self.summary_writer = tf.summary.FileWriter(log_dir, self.session.graph)

    def on_episode_end(self, episode, logs=None, **kwargs):
        """
        Prints episode total reward and loss
        :param episode: int, episode number
        :param logs: dict, episode data
        :return: None
        """
        if logs is not None:
            feed_dict = {log_placeholder: logs[log_name]
                         for log_name, log_placeholder in self.summary_placeholders.items()}
            episode_summary = self.session.run(self.summary_op, feed_dict=feed_dict)
            self.summary_writer.add_summary(episode_summary, episode)
            self.summary_writer.flush()
