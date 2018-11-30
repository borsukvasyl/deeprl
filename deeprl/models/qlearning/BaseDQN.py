from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod
import tensorflow as tf

from deeprl.models import BaseModel


class BaseDQN(BaseModel):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_q(self, states):
        pass

    @abstractmethod
    def train(self, batch_states, batch_actions, q_target):
        pass

    def get_one_q(self, state):
        return self.get_q([state])[0]

    @staticmethod
    def calculate_loss(q_value, q_target, actions, a_size=-1):
        """
        Calculates loss for DQN model
        :param q_value:
        :param q_target:
        :param actions:
        :param a_size: int, size of one-hot encoding, if is negative then encoding is not performed
        :return:
        """
        if a_size != -1:
            actions = tf.one_hot(actions, a_size)
        used_q = tf.reduce_sum(tf.multiply(q_value, actions), axis=1)
        return tf.reduce_mean(tf.square(q_target - used_q))
