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
        """
        Returns Q-values for given states
        :param states: list, list of states
        :return: list, list of Q-values
        """
        pass

    @abstractmethod
    def train(self, batch_states, batch_actions, q_target):
        """
        Trains model
        :param batch_states: list, batch of states
        :param batch_actions: list, batch of actions
        :param q_target: list, batch of Q-value targets
        :return: int, loss
        """
        pass

    def get_one_q(self, state):
        """
        Returns Q-value for one state
        :param state: state
        :return: Q-value for given state
        """
        return self.get_q([state])[0]

    @staticmethod
    def calculate_loss(q_value, q_target, actions, a_size=-1):
        """
        Calculates loss for DQN model
        :param q_value: tensor [batch_size, a_size]
        :param q_target: tensor [batch_size,]
        :param actions: tensor [batch_size,] or [batch_size, a_size]
        :param a_size: int, size of one-hot encoding, if is negative then encoding is not performed
        :return: tensor [1]
        """
        if a_size != -1:
            actions = tf.one_hot(actions, a_size)

        used_q = tf.reduce_sum(tf.multiply(q_value, actions), axis=1)
        return tf.reduce_mean(tf.square(q_target - used_q))
