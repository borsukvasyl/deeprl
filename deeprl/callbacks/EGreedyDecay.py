from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deeprl.callbacks.Callback import Callback


class EGreedyDecay(Callback):
    def __init__(self, e_greedy, e_min, e_decay):
        self.e_greedy = e_greedy
        self.e_min = e_min
        self.e_decay = e_decay

    def on_episode(self):
        if self.e_greedy.e > self.e_min:
            self.e_greedy.e *= self.e_decay
