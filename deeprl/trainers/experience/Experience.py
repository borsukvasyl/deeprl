from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque

from deeprl.trainers.experience.samplers import generate_sampler


class Experience(object):
    """
    Data structure to store experience frames
    """

    def __init__(self, config):
        """
        :param maxlen: int, maximum length of experience
        """
        self.config = config
        self.experience = deque(maxlen=config.experience_size)

        self.sampler = generate_sampler(self, config)

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

    def sample(self, done=False):
        """
        Samples random experience.
        :param done: bool, whether episode is finished
        :return: list or ValueError, if batch_size is negative ot bigger than experience length
        """
        return self.sampler.sample(done)

    def __iter__(self):
        for exp in self.experience:
            yield exp
