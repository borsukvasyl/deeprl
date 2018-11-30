from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
import tensorflow as tf

from deeprl.models.qlearning import BaseDQN


class BaseDuelingDQN(BaseDQN):
    __metaclass__ = ABCMeta

    @staticmethod
    def calculate_q_value(value, advantage):
        return value + tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keepdims=True))
