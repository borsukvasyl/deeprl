from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class PolicyGradientConfig(object):
    def __init__(self):
        self.experience_sampler = ["ordered", "on_episode_clear"]
        self.experience_size = None
        self.min_sample_size = 1

        self.discount_factor = 0.99
