from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod
import tensorflow as tf

from deeprl.models import BaseModel


class BaseACNet(BaseModel):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_policy(self, states):
        pass

    def get_one_policy(self, state):
        return self.get_policy([state])[0]

    @abstractmethod
    def get_v(self, states):
        pass

    @abstractmethod
    def get_policy_and_value(self, states):
        pass

    @abstractmethod
    def train(self, batch_states, batch_actions, batch_advantages, batch_target_v):
        pass

    @staticmethod
    def calculate_policy_loss(policy, actions, advantages, a_size=-1, entropy=False):
        if a_size != -1:
            actions = tf.one_hot(actions, depth=a_size)

        log_prob = tf.log(tf.clip_by_value(policy, 0.000001, 0.999999))
        neg_log_responsible_policy = -tf.reduce_sum(tf.multiply(log_prob, actions), axis=1)
        policy_loss = tf.reduce_mean(neg_log_responsible_policy * advantages)
        if entropy:
            return policy_loss, tf.reduce_sum(tf.multiply(policy, -log_prob))
        return policy_loss

    @staticmethod
    def calculate_value_loss(value, target_value):
        return tf.reduce_mean(tf.square(value - target_value))
