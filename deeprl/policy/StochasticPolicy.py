from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class StochasticPolicy(object):
    def choose_action(self, pi):
        return np.random.choice(len(pi), p=pi)
