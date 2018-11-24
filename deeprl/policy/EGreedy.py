from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class EGreedyPolicy(object):
    def __init__(self, e):
        self.e = e

    def choose_action(self, q_values):
        if np.random.random() < self.e:
            return np.random.randint(len(q_values))
        return np.argmax(q_values)
