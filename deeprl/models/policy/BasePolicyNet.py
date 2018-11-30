from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod
import tensorflow as tf

from deeprl.models import BaseModel


class BasePolicyNet(BaseModel):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_policy(self, states):
        pass

    @abstractmethod
    def train(self, batch_states, batch_actions, discounted_episode_rewards):
        pass

    def get_one_policy(self, state):
        return self.get_policy([state])[0]

    @staticmethod
    def calculate_loss(policy, actions, discounted_episode_rewards, a_size=-1):
        if a_size != -1:
            actions = tf.one_hot(actions, depth=a_size)

        log_prob = tf.log(tf.clip_by_value(policy, 0.000001, 0.999999))
        neg_log_responsible_policy = -tf.reduce_sum(tf.multiply(log_prob, actions), axis=1)
        return tf.reduce_mean(neg_log_responsible_policy * discounted_episode_rewards)
