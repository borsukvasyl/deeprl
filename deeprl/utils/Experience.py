from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque
import random


class Experience(object):
    """
    Data structure to store experience frames
    """

    def __init__(self, maxlen):
        """
        :param maxlen: int, maximum length of experience
        """
        self.experience = deque(maxlen=maxlen)

    @property
    def size(self):
        """
        Gets current number of experience frames
        :return: int
        """
        return len(self.experience)

    def is_empty(self):
        """
        Checks whether experience is empty
        :return: bool
        """
        return self.size == 0

    def is_full(self):
        """
        Checks whether experience is full
        :return: bool
        """
        return self.size == self.experience.maxlen - 1

    def _add(self, data):
        self.experience.append(data)

    def add(self, state, action, reward, next_state, done):
        self._add([state, action, reward, next_state, done])

    def clear(self):
        """
        Clears experience
        :return: None
        """
        self.experience.clear()

    def sample(self, batch_size):
        """
        Samples random experience.
        :param batch_size: int, number of experience frames to return
        :return: list or ValueError, if batch_size is negative ot bigger than experience length
        """
        return random.sample(self.experience, batch_size)
