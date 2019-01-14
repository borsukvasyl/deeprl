from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from abc import ABCMeta, abstractmethod


class BaseSampler(object):
    __metaclass__ = ABCMeta

    def __init__(self, experience, config, prev_sampler=None):
        self.experience = experience
        self.config = config
        self.prev_sampler = prev_sampler

    @abstractmethod
    def sample(self, done):
        pass


class RandomSampler(BaseSampler):
    def __init__(self, experience, config, prev_sampler=None):
        super(RandomSampler, self).__init__(experience, config, prev_sampler)
        self.last_update = -1

    def sample(self, done):
        self.last_update = 0 if done else self.last_update + 1
        if self.last_update % self.config.train_freq == 0 and self.experience.size >= self.config.batch_size:
            return random.sample(self.experience.experience, self.config.batch_size)
        return None


class OrderedSampler(BaseSampler):
    def sample(self, done):
        if done or self.experience.size >= self.config.min_sample_size:
            return list(self.experience)
        return None


class OnDoneClear(BaseSampler):
    def sample(self, done):
        sampled = self.prev_sampler.sample(done)
        if done: self.experience.clear()
        return sampled


class OnSampleClear(BaseSampler):
    def sample(self, done):
        sampled = self.prev_sampler.sample(done)
        if sampled: self.experience.clear()
        return sampled


def generate_sampler(experience, config):
    sampler = None

    if "random" in config.experience_sampler:
        sampler = RandomSampler(experience, config, prev_sampler=sampler)
    elif "ordered" in config.experience_sampler:
        sampler = OrderedSampler(experience, config, prev_sampler=sampler)

    if "on_episode_clear" in config.experience_sampler:
        sampler = OnDoneClear(experience, config, prev_sampler=sampler)
    elif "on_sample_clear" in config.experience_sampler:
        sampler = OnSampleClear(experience, config, prev_sampler=sampler)

    if sampler is None:
        raise KeyError("invalid configuration for experience sampler")
    return sampler
