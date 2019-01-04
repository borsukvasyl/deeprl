from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class DQNConfig(object):
    def __init__(self):
        # experience replay
        self.experience_sampler = ["random"]
        self.experience_size = 2000
        self.batch_size = 40

        # For epsilon greedy policy
        self.e = 1
        self.e_decay = 0.995
        self.e_min = 0.1

        self.discount_factor = 0.99
