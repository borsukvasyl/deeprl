from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deeprl.agents import BaseAgent
from deeprl.policy import GreedyPolicy


class QAgent(BaseAgent):
    def __init__(self, model):
        super(QAgent, self).__init__(model, GreedyPolicy())

    def choose_action(self, state):
        q_value = self.model.get_one_q(state)
        return self.policy.choose_action(q_value)
