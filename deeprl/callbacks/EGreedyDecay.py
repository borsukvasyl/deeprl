from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deeprl.callbacks.BaseCallback import BaseCallback


class EGreedyDecay(BaseCallback):
    def __init__(self, e_greedy, e_min, e_decay):
        """
        Initializes EGreedyDecay instance
        :param e_greedy: EGreedyPolicy, policy instance
        :param e_min: float, minimum value of e
        :param e_decay: float, e decay multiplier
        """
        super(EGreedyDecay, self).__init__()

        self.e_greedy = e_greedy
        self.e_min = e_min
        self.e_decay = e_decay

    def on_episode_end(self, episode, logs=None):
        """
        Decreases e if it is bigger than e_min
        :param episode: int, episode number
        :param logs: dict, episode data
        :return: None
        """
        if self.e_greedy.e > self.e_min:
            self.e_greedy.e *= self.e_decay
