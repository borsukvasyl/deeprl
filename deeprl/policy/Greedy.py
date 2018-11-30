from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class GreedyPolicy(object):
    def choose_actions(self, q_values):
        return [self.choose_action(q_value) for q_value in q_values]

    def choose_action(self, q_values):
        return np.argmax(q_values)
