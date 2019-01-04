from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ACConfig(object):
    def __init__(self):
        # experience replay
        self.experience_sampler = ["ordered", "on_sample_clear"]
        self.experience_size = None
        self.min_sample_size = 4

        self.discount_factor = 0.99
